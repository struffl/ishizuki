// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
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

  /// Only a disk's own failure is worth waiting out: a file that ends early or a descriptor
  /// that is no good fails the first time, and says which it was.
  @Test("a read the file cannot answer fails at once and says why")
  func failures() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }
    try Data(count: 16).write(to: url)
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let buffer = try ResidentBuffer(byteCount: 64)

    let start = Date()
    let short = #expect(throws: BonsaiError.self) {
      try buffer.read(from: handle.fileDescriptor, offset: 0, into: 0..<64)
    }
    let closed = #expect(throws: BonsaiError.self) {
      try buffer.read(from: -1, offset: 0, into: 0..<64)
    }
    #expect(short?.description == "Missing weight: read 16 of 64 bytes at 0: the file ends first")
    #expect(closed?.description.hasSuffix("Bad file descriptor") == true)
    #expect(-start.timeIntervalSinceNow < 0.25)
  }

  /// A file offset and a destination that share no phase within a page, pieces larger than the
  /// scratch memory they pass through, and a read that ends where the file does.
  @Test("an uncached read lands the same bytes wherever the offsets fall")
  func readsUncached() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }
    let bytes = (0..<(9 << 20) + 4321).map { UInt8(truncatingIfNeeded: $0 &* 2_654_435_761 >> 7) }
    try Data(bytes).write(to: url)
    let descriptor = ExpertStore.openUncached(url)
    defer { close(descriptor) }

    for (offset, count, at) in [
      (777, 40_000, 5), (0, 16_384, 0), (123, 8_600_000, 16_390), (bytes.count - 30_001, 30_001, 3),
    ] {
      let buffer = try ResidentBuffer(byteCount: at + count)
      try buffer.readUncached(from: descriptor, offset: offset, into: at..<(at + count))
      let landed = UnsafeRawBufferPointer(start: buffer.pointer.advanced(by: at), count: count)
      #expect(Array(landed) == Array(bytes[offset..<(offset + count)]), "\(count) at \(offset)")
    }
    let buffer = try ResidentBuffer(byteCount: 64)
    #expect(throws: BonsaiError.self) {
      try buffer.readUncached(from: descriptor, offset: bytes.count - 10, into: 0..<64)
    }
  }

  @Test("a read past the end is refused rather than scribbling")
  func bounds() throws {
    let buffer = try ResidentBuffer(byteCount: 64)
    #expect(throws: BonsaiError.self) {
      try buffer.read(from: 0, offset: 0, into: 0..<(buffer.byteCount + 1))
    }
  }
}
