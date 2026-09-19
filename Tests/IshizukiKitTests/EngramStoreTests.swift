// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// An n-gram table read by address, ahead of the work that needs it.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// The table is bigger than memory and the addresses are known before the step runs, so what
/// has to be true is narrower than for an expert cache: every row lands where it was asked for,
/// a fetch can be in flight while the last one is still being read, and a bad address is caught
/// rather than served.
@Suite("Engram store")
struct EngramStoreTests {
  private let vocab = 997
  private let parts = 8
  private let dim = 16

  private func layout() -> EngramLayout {
    EngramLayout(
      ngramSize: 3, vocabSize: vocab, parts: parts, embedDim: dim, dtype: "float32")
  }

  /// Writes the table so that every value in row `r` is `r`. A row read from the wrong file or
  /// the wrong offset then shows up as the wrong number rather than as plausible noise.
  private func write(to directory: URL) throws {
    let layout = layout()
    try FileManager.default.createDirectory(
      at: directory.appending(path: "engrams"), withIntermediateDirectories: true)

    var blobs = [[Float]](repeating: [], count: parts)
    for row in 0..<vocab {
      let placed = layout.place(row: row)
      blobs[placed.part].append(contentsOf: [Float](repeating: Float(row), count: dim))
    }
    for part in 0..<parts {
      let data = blobs[part].withUnsafeBufferPointer { Data(buffer: $0) }
      try data.write(to: directory.appending(path: EngramLayout.fileName(part: part)))
    }
  }

  private func temporary() -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "engrams-\(UUID().uuidString)")
  }

  @Test("fetches every row to the slot it was asked for")
  func fetches() throws {
    let directory = temporary()
    defer { try? FileManager.default.removeItem(at: directory) }
    try write(to: directory)

    let store = try EngramStore(directory: directory, layout: layout(), capacity: 64)
    let wanted = [0, 996, 1, 500, 7, 8, 63, 64]
    let got = try store.rows(wanted)
    eval(got)

    #expect(got.shape == [wanted.count, dim])
    for (slot, row) in wanted.enumerated() {
      #expect(got[slot, 0].item(Float.self) == Float(row))
      #expect(got[slot, dim - 1].item(Float.self) == Float(row))
    }
    #expect(store.rowsRead == wanted.count)
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

  @Test("a whole context's rows arrive in one fetch")
  func wholeContext() throws {
    let directory = temporary()
    defer { try? FileManager.default.removeItem(at: directory) }
    try write(to: directory)

    let store = try EngramStore(directory: directory, layout: layout(), capacity: 512)
    let wanted = (0..<512).map { ($0 * 7) % vocab }
    let got = try store.rows(wanted)
    eval(got)

    for slot in stride(from: 0, to: 512, by: 37) {
      #expect(got[slot, 0].item(Float.self) == Float(wanted[slot]))
    }
  }

  @Test("refuses an address the table does not have, and an overlong fetch")
  func refusesBadAddresses() throws {
    let directory = temporary()
    defer { try? FileManager.default.removeItem(at: directory) }
    try write(to: directory)

    let store = try EngramStore(directory: directory, layout: layout(), capacity: 8)
    #expect(throws: BonsaiError.self) { try store.prefetch([vocab]) }
    #expect(throws: BonsaiError.self) { try store.prefetch([-1]) }
    #expect(throws: BonsaiError.self) { try store.prefetch(Array(0..<9)) }
    #expect(throws: BonsaiError.self) { _ = try store.take() }
  }
}
