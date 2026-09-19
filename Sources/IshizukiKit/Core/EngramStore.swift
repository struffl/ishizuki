// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// An n-gram embedding table kept on disk and fetched before it is needed.

import Foundation
import MLX

/// How an n-gram table is cut into files, and how an n-gram becomes a row of it.
///
/// Each head hashes the same n-gram modulo its own prime, so a collision in one head does not
/// travel to the others and the heads disagree independently. A head's rows are a contiguous
/// block of the table, and the blocks sit end to end, so the address is arithmetic and no index
/// has to be consulted or held.
public struct EngramLayout: Codable, Sendable, Equatable {
  public var ngramSize: Int
  public var heads: Int
  public var headDim: Int
  /// One prime per head, and where that head's block starts.
  public var vocabSizes: [Int]
  public var offsets: [Int]
  public var parts: Int
  public var rowsPerPart: Int
  public var dtype: String

  public init(
    ngramSize: Int, heads: Int, headDim: Int, vocabSizes: [Int], offsets: [Int],
    parts: Int, rowsPerPart: Int, dtype: String
  ) {
    self.ngramSize = ngramSize
    self.heads = heads
    self.headDim = headDim
    self.vocabSizes = vocabSizes
    self.offsets = offsets
    self.parts = parts
    self.rowsPerPart = rowsPerPart
    self.dtype = dtype
  }

  /// The heads laid end to end, each starting where the last one ended.
  public static func blocked(
    ngramSize: Int, headDim: Int, vocabSizes: [Int], parts: Int, dtype: String
  ) -> EngramLayout {
    var offsets: [Int] = []
    var running = 0
    for size in vocabSizes {
      offsets.append(running)
      running += size
    }
    return EngramLayout(
      ngramSize: ngramSize, heads: vocabSizes.count, headDim: headDim, vocabSizes: vocabSizes,
      offsets: offsets, parts: parts,
      rowsPerPart: (running + parts - 1) / parts, dtype: dtype)
  }

  public var type: DType {
    get throws {
      guard let type = DType.allCases.first(where: { "\($0)" == dtype }) else {
        throw BonsaiError.unsupportedModel("\(dtype) is not a type this runtime reads")
      }
      return type
    }
  }

  public var totalRows: Int { (offsets.last ?? 0) + (vocabSizes.last ?? 0) }
  /// What one token's heads come to once they are laid side by side.
  public var width: Int { heads * headDim }
  public var rowBytes: Int {
    get throws { headDim * (try type.size) }
  }

  public static func fileName(part: Int) -> String {
    "engrams/part_\(String(format: "%03d", part)).bin"
  }

  public func place(row: Int) -> (part: Int, index: Int) {
    (part: row / rowsPerPart, index: row % rowsPerPart)
  }

  /// Where one head keeps the n-gram that hashed to `hash`.
  public func address(head: Int, hash: Int) -> Int {
    let size = vocabSizes[head]
    var folded = hash % size
    if folded < 0 { folded += size }
    return offsets[head] + folded
  }
}

/// One n-gram table, read a context's worth of rows at a time.
///
/// This is the half of streaming that routed experts could not be: an n-gram's address is the
/// tokens themselves, so every row a step needs is known before the step begins. Nothing has to
/// come back from the GPU first, the reads can all go out at once, and a fetch for the next
/// token overlaps the work of the current one. An expert cache is a guess about what will be
/// asked for; this is not a cache at all.
public final class EngramStore: @unchecked Sendable {
  public let layout: EngramLayout
  public let capacity: Int

  private let descriptors: [Int32]
  /// Two buffers, so a prefetch can land while the rows already fetched are still being read.
  /// A single buffer is what forces a streamed expert to evaluate before it reads again.
  private var buffers: [ResidentBuffer]
  private var front = 0

  private let queue = DispatchQueue(
    label: "studio.ishizuki.engrams", qos: .userInitiated, attributes: .concurrent)
  private let lock = NSLock()
  private var pending: (group: DispatchGroup, rows: Int, tokens: Int?, error: Error?)?

  public private(set) var rowsRead = 0

  public init(directory: URL, layout: EngramLayout, capacity: Int = 8192) throws {
    self.layout = layout
    self.capacity = capacity

    var opened: [Int32] = []
    for part in 0..<layout.parts {
      let url = directory.appending(path: EngramLayout.fileName(part: part))
      let handle = open(url.path, O_RDONLY)
      guard handle >= 0 else {
        for previous in opened { close(previous) }
        throw BonsaiError.missingWeight("cannot open \(url.lastPathComponent)")
      }
      opened.append(handle)
    }
    self.descriptors = opened

    let bytes = capacity * (try layout.rowBytes)
    self.buffers = [try ResidentBuffer(byteCount: bytes), try ResidentBuffer(byteCount: bytes)]
  }

  deinit { for descriptor in descriptors { close(descriptor) } }

  /// Starts fetching `rows` into the buffer that is not in use. Returns at once; the rows are
  /// there once `take()` returns.
  public func prefetch(_ rows: [Int], tokens: Int? = nil) throws {
    let width = try layout.rowBytes

    lock.lock()
    guard pending == nil else {
      lock.unlock()
      throw BonsaiError.missingComponent("a prefetch is already in flight")
    }
    guard rows.count <= capacity else {
      lock.unlock()
      throw BonsaiError.shapeMismatch(
        "\(rows.count) rows asked for, but this store holds \(capacity)")
    }
    for row in rows where row < 0 || row >= layout.totalRows {
      lock.unlock()
      throw BonsaiError.shapeMismatch(
        "n-gram row \(row) is not in a \(layout.totalRows) table")
    }
    let target = 1 - front
    let group = DispatchGroup()
    pending = (group: group, rows: rows.count, tokens: tokens, error: nil)
    lock.unlock()

    let buffer = buffers[target]

    // Each row is its own read at its own offset into a disjoint slice of the buffer, so they
    // can all be in flight together. A token's heads are scattered the length of the table by
    // design, and a disk answers scattered reads far better in parallel than one at a time.
    for (slot, row) in rows.enumerated() {
      queue.async(group: group) { [self] in
        let placed = layout.place(row: row)
        do {
          try buffer.read(
            from: descriptors[placed.part], offset: placed.index * width,
            into: (slot * width)..<((slot + 1) * width))
        } catch {
          lock.lock()
          if pending?.error == nil { pending?.error = error }
          lock.unlock()
        }
      }
    }
  }

  /// Waits for the prefetch to land, makes it current, and hands back its rows. Rows fetched
  /// for whole tokens come back as one row per token, every head laid side by side.
  public func take() throws -> MLXArray {
    lock.lock()
    guard let inFlight = pending else {
      lock.unlock()
      throw BonsaiError.missingComponent("nothing was prefetched")
    }
    lock.unlock()

    inFlight.group.wait()

    lock.lock()
    let failure = pending?.error
    pending = nil
    front = 1 - front
    rowsRead += inFlight.rows
    lock.unlock()

    if let failure { throw failure }
    let shape =
      inFlight.tokens.map { [$0, layout.width] } ?? [inFlight.rows, layout.headDim]
    return buffers[front].array(shape: shape, dtype: try layout.type)
  }

  /// Fetches and waits, for the caller that has nothing to overlap with.
  public func rows(_ rows: [Int]) throws -> MLXArray {
    try prefetch(rows)
    return try take()
  }

  /// The addresses one context's n-grams resolve to, every head of a token in a run so the
  /// fetched rows come back already laid out as `[token, heads * headDim]`.
  public func addresses(hashes: [[Int]]) throws -> [Int] {
    var rows: [Int] = []
    rows.reserveCapacity(hashes.count * layout.heads)
    for token in hashes {
      guard token.count == layout.heads else {
        throw BonsaiError.shapeMismatch(
          "a token hashed to \(token.count) heads, but the table has \(layout.heads)")
      }
      for (head, hash) in token.enumerated() {
        rows.append(layout.address(head: head, hash: hash))
      }
    }
    return rows
  }

  /// One embedding per token, fetched and assembled.
  public func embeddings(hashes: [[Int]]) throws -> MLXArray {
    try prefetch(try addresses(hashes: hashes), tokens: hashes.count)
    return try take()
  }
}
