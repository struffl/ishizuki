// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// Memory the runtime owns, used by MLX in place.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// The one property streaming depends on: a buffer filled by `pread` becomes an `MLXArray`
/// without the bytes being copied. If this ever starts copying, a streamed layer pays twice for
/// every weight and the whole design stops being worth having.
@Suite("Resident buffer")
struct ResidentBufferTests {
  private func temporary() -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "resident-\(UUID().uuidString).bin")
  }

  @Test("reads a slice off disk and hands it to MLX in place")
  func readsAndWraps() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }

    let values = (0..<1024).map { Float($0) * 0.5 }
    var data = Data()
    for value in values { withUnsafeBytes(of: value) { data.append(contentsOf: $0) } }
    try data.write(to: url)

    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    // Only the second half, read straight into the buffer at a non-zero offset.
    let buffer = try ResidentBuffer(byteCount: 512 * 4)
    try buffer.read(from: handle.fileDescriptor, offset: 512 * 4, into: 0..<(512 * 4))

    let array = buffer.array(shape: [512], dtype: .float32)
    eval(array)
    #expect(array.shape == [512])
    #expect(array[0].item(Float.self) == values[512])
    #expect(array[511].item(Float.self) == values[1023])

    // Writing through the pointer changes what MLX sees, which is what "no copy" means.
    buffer.pointer.assumingMemoryBound(to: Float.self)[0] = -1
    let again = buffer.array(shape: [512], dtype: .float32)
    eval(again)
    #expect(again[0].item(Float.self) == -1)
  }

  @Test("the buffer outlives every array cut from it")
  func lifetime() throws {
    var array: MLXArray?
    do {
      let buffer = try ResidentBuffer(byteCount: 64 * 4)
      buffer.pointer.assumingMemoryBound(to: Float.self)[7] = 42
      array = buffer.array(shape: [64], dtype: .float32)
    }
    // The buffer went out of scope; the array holds the only reference left.
    let held = try #require(array)
    eval(held)
    #expect(held[7].item(Float.self) == 42)
  }

  @Test("a read past the end is refused rather than scribbling")
  func bounds() throws {
    let buffer = try ResidentBuffer(byteCount: 64)
    #expect(throws: BonsaiError.self) {
      try buffer.read(from: 0, offset: 0, into: 0..<(buffer.byteCount + 1))
    }
  }
}
