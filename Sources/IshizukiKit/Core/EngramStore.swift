// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// An n-gram embedding table kept on disk and fetched before it is needed.

import Foundation
import MLX

/// How an n-gram table is cut into files and what one row of it looks like.
///
/// The table is far larger than the model it belongs to and a context touches almost none of
/// it: one row per token. What matters is not how much fits in memory but how quickly a row
/// arrives, so the file is a flat array of fixed-width rows and the address is arithmetic.
public struct EngramLayout: Codable, Sendable, Equatable {
  public var ngramSize: Int
  public var vocabSize: Int
  /// The table is split across this many files, striped by row so that a context's rows land
  /// on every file rather than crowding one.
  public var parts: Int
  public var embedDim: Int
  public var dtype: String

  public init(ngramSize: Int, vocabSize: Int, parts: Int, embedDim: Int, dtype: String) {
    self.ngramSize = ngramSize
    self.vocabSize = vocabSize
    self.parts = parts
    self.embedDim = embedDim
    self.dtype = dtype
  }

  public var type: DType {
    get throws {
      guard let type = DType.allCases.first(where: { "\($0)" == dtype }) else {
        throw BonsaiError.unsupportedModel("\(dtype) is not a type this runtime reads")
      }
      return type
    }
  }

  public var rowBytes: Int {
    get throws { embedDim * (try type.size) }
  }

  public static func fileName(part: Int) -> String {
    "engrams/part_\(String(format: "%03d", part)).bin"
  }

  /// Which file a row lives in, and where in it. Striping by remainder keeps a batch of rows
  /// spread over every file, so the reads go out in parallel instead of queueing on one.
  public func place(row: Int) -> (part: Int, index: Int) {
    (part: row % parts, index: row / parts)
  }

  public func rowsIn(part: Int) -> Int {
    (vocabSize - part + parts - 1) / parts
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

  private let descriptors: [Int32]
  /// Two buffers, so a prefetch can land while the rows already fetched are still being read.
  /// A single buffer is what forces a streamed expert to evaluate before it reads again.
  private var buffers: [ResidentBuffer]
  private var front = 0
  private var capacity: Int

  private let queue = DispatchQueue(
    label: "studio.ishizuki.engrams", qos: .userInitiated, attributes: .concurrent)
  private let lock = NSLock()
  private var pending: (group: DispatchGroup, count: Int, error: Error?)?

  public private(set) var rowsRead = 0

  public init(directory: URL, layout: EngramLayout, capacity: Int = 512) throws {
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
  public func prefetch(_ rows: [Int]) throws {
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
    for row in rows where row < 0 || row >= layout.vocabSize {
      lock.unlock()
      throw BonsaiError.shapeMismatch("n-gram row \(row) is not in a \(layout.vocabSize) table")
    }
    let target = 1 - front
    let group = DispatchGroup()
    pending = (group: group, count: rows.count, error: nil)
    lock.unlock()

    let width = try layout.rowBytes
    let buffer = buffers[target]

    // Each row is its own read at its own offset into a disjoint slice of the buffer, so they
    // can all be in flight together. A context's rows are scattered over the table, and a disk
    // answers scattered reads far better in parallel than one after another.
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

  /// Waits for the prefetch to land, makes it current, and hands back its rows.
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
    rowsRead += inFlight.count
    lock.unlock()

    if let failure { throw failure }
    return buffers[front].array(
      shape: [inFlight.count, layout.embedDim], dtype: try layout.type)
  }

  /// Fetches and waits, for the caller that has nothing to overlap with.
  public func rows(_ rows: [Int]) throws -> MLXArray {
    try prefetch(rows)
    return try take()
  }
}
