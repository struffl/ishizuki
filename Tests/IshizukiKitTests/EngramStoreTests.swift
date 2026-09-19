// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// An n-gram table read by address, ahead of the work that needs it.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// The table is far larger than memory and every address is known before the step runs, so what
/// has to be true is narrower than for an expert cache: each head reaches its own block, every
/// row lands in the slot it was asked for, and a fetch can be in flight while the last one is
/// still being read.
@Suite("Engram store")
struct EngramStoreTests {
  /// The shape of the real table in miniature: distinct primes per head, heads laid end to end,
  /// cut into more files than there are heads.
  private let primes = [1009, 1013, 1019, 1021]
  private let headDim = 8
  private let parts = 6

  private func layout() -> EngramLayout {
    EngramLayout.blocked(
      ngramSize: 3, headDim: headDim, vocabSizes: primes, parts: parts, dtype: "float32")
  }

  /// Every value in row `r` is `r`, so a row read from the wrong file or the wrong block shows
  /// up as the wrong number rather than as plausible noise.
  private func write(to directory: URL) throws {
    let layout = layout()
    try FileManager.default.createDirectory(
      at: directory.appending(path: "engrams"), withIntermediateDirectories: true)

    for part in 0..<layout.parts {
      var values: [Float] = []
      for index in 0..<layout.rowsPerPart {
        let row = part * layout.rowsPerPart + index
        values.append(contentsOf: [Float](repeating: Float(row), count: headDim))
      }
      let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
      try data.write(to: directory.appending(path: EngramLayout.fileName(part: part)))
    }
  }

  private func temporary() -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "engrams-\(UUID().uuidString)")
  }

  @Test("lays the heads end to end and folds each by its own prime")
  func addressing() {
    let layout = layout()
    #expect(layout.heads == 4)
    #expect(layout.width == 4 * headDim)
    #expect(layout.offsets == [0, 1009, 2022, 3041])
    #expect(layout.totalRows == 1009 + 1013 + 1019 + 1021)

    #expect(layout.address(head: 0, hash: 0) == 0)
    #expect(layout.address(head: 1, hash: 0) == 1009)
    // A hash past a head's prime wraps inside that head, never into its neighbour.
    #expect(layout.address(head: 1, hash: 1013) == 1009)
    #expect(layout.address(head: 1, hash: 1012) == 1009 + 1012)
    #expect(layout.address(head: 3, hash: -1) == 3041 + 1020)
  }

  @Test("fetches every row to the slot it was asked for")
  func fetches() throws {
    let directory = temporary()
    defer { try? FileManager.default.removeItem(at: directory) }
    try write(to: directory)

    let store = try EngramStore(directory: directory, layout: layout(), capacity: 64)
    let wanted = [0, 4061, 1, 2022, 1008, 1009, 3041]
    let got = try store.rows(wanted)
    eval(got)

    #expect(got.shape == [wanted.count, headDim])
    for (slot, row) in wanted.enumerated() {
      #expect(got[slot, 0].item(Float.self) == Float(row))
      #expect(got[slot, headDim - 1].item(Float.self) == Float(row))
    }
    #expect(store.rowsRead == wanted.count)
  }

  /// What a layer actually asks for: one embedding per token, every head side by side.
  @Test("assembles a token's heads into one row")
  func embeddings() throws {
    let directory = temporary()
    defer { try? FileManager.default.removeItem(at: directory) }
    try write(to: directory)

    let layout = layout()
    let store = try EngramStore(directory: directory, layout: layout, capacity: 64)
    let hashes = [[0, 0, 0, 0], [5, 6, 7, 8], [1008, 1012, 1018, 1020]]
    let got = try store.embeddings(hashes: hashes)
    eval(got)

    #expect(got.shape == [hashes.count, layout.width])
    for (token, heads) in hashes.enumerated() {
      for (head, hash) in heads.enumerated() {
        let expected = Float(layout.address(head: head, hash: hash))
        #expect(got[token, head * headDim].item(Float.self) == expected)
        #expect(got[token, (head + 1) * headDim - 1].item(Float.self) == expected)
      }
    }
  }

  /// The point of the two buffers: the rows a step is still reading must survive the fetch for
  /// the next step. A single buffer would have the second fetch land on top of the first.
  @Test("a fetch in flight does not disturb the rows already handed out")
  func doubleBuffers() throws {
    let directory = temporary()
    defer { try? FileManager.default.removeItem(at: directory) }
    try write(to: directory)

    let store = try EngramStore(directory: directory, layout: layout(), capacity: 64)
    let first = try store.rows([10, 11, 12])

    try store.prefetch([900, 901, 902])
    let second = try store.take()
    eval(first, second)

    #expect(first[0, 0].item(Float.self) == 10)
    #expect(first[2, 0].item(Float.self) == 12)
    #expect(second[0, 0].item(Float.self) == 900)
    #expect(second[2, 0].item(Float.self) == 902)
  }

  @Test("a whole context's tokens arrive in one fetch")
  func wholeContext() throws {
    let directory = temporary()
    defer { try? FileManager.default.removeItem(at: directory) }
    try write(to: directory)

    let layout = layout()
    let store = try EngramStore(directory: directory, layout: layout, capacity: 4096)
    let hashes = (0..<512).map { token in (0..<4).map { token * 7 + $0 * 131 } }
    let got = try store.embeddings(hashes: hashes)
    eval(got)

    #expect(got.shape == [512, layout.width])
    #expect(store.rowsRead == 512 * 4)
    for token in stride(from: 0, to: 512, by: 53) {
      let expected = Float(layout.address(head: 2, hash: hashes[token][2]))
      #expect(got[token, 2 * headDim].item(Float.self) == expected)
    }
  }

  @Test("refuses an address the table does not have, and an overlong fetch")
  func refusesBadAddresses() throws {
    let directory = temporary()
    defer { try? FileManager.default.removeItem(at: directory) }
    try write(to: directory)

    let layout = layout()
    let store = try EngramStore(directory: directory, layout: layout, capacity: 8)
    #expect(throws: BonsaiError.self) { try store.prefetch([layout.totalRows]) }
    #expect(throws: BonsaiError.self) { try store.prefetch([-1]) }
    #expect(throws: BonsaiError.self) { try store.prefetch(Array(0..<9)) }
    #expect(throws: BonsaiError.self) { _ = try store.take() }
    #expect(throws: BonsaiError.self) { _ = try store.embeddings(hashes: [[1, 2]]) }
  }
}
