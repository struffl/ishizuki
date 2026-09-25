// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Memory this runtime allocates and fills itself, handed to MLX without a copy.

import Cmlx
import Foundation
import MLX

/// A page-aligned allocation that can be read into and then read from as an `MLXArray`.
///
/// Streaming a model off disk needs a buffer the runtime fills with `pread` and MLX then uses
/// in place. Every other way into an `MLXArray` copies, which for a weight that is read once
/// per token costs more than the read did. `mlx_array_new_data_managed` takes ownership of a
/// pointer instead, so the bytes are written exactly once.
public final class ResidentBuffer: @unchecked Sendable {
  public let pointer: UnsafeMutableRawPointer
  public let byteCount: Int
  private let alignment: Int
  /// Whether the pages are locked in memory, where the system can neither swap nor compress
  /// them. A lock that fails leaves an ordinary buffer rather than no buffer.
  public let isLocked: Bool

  /// Metal wants its allocations page aligned; 16 KiB is the page size on Apple silicon.
  public static let pageSize = 16384

  public init(
    byteCount: Int, alignment: Int = ResidentBuffer.pageSize, locked: Bool = false
  ) throws {
    guard byteCount > 0 else {
      throw BonsaiError.shapeMismatch("a resident buffer needs a size")
    }
    var raw: UnsafeMutableRawPointer?
    let rounded = (byteCount + alignment - 1) / alignment * alignment
    guard posix_memalign(&raw, alignment, rounded) == 0, let raw else {
      throw BonsaiError.missingComponent("could not allocate \(rounded) bytes")
    }
    self.pointer = raw
    self.byteCount = rounded
    self.alignment = alignment
    self.isLocked = locked && mlock(raw, rounded) == 0
  }

  deinit {
    if isLocked { munlock(pointer, byteCount) }
    free(pointer)
  }

  /// Fills `range` of this buffer from `descriptor` at `offset`, in one read.
  ///
  /// A read the disk fails is tried again, waiting longer each time, before it gives up. A USB
  /// drive pushed hard can reset and report I/O errors for several seconds — reads and writes
  /// together did it to the one DeepSeek-V4.1 was streamed from — and answers again once it is
  /// back, so the first `EIO` is no reason to lose a whole generation.
  @discardableResult
  public func read(
    from descriptor: Int32, offset: Int, into range: Range<Int>
  ) throws -> Int {
    guard range.lowerBound >= 0, range.upperBound <= byteCount else {
      throw BonsaiError.shapeMismatch("read of \(range) runs past a \(byteCount)-byte buffer")
    }
    return try Self.fill(
      pointer.advanced(by: range.lowerBound), from: descriptor, offset: offset,
      count: range.count, needed: range.count)
  }

  /// Fills `range` as `read` does from a `descriptor` opened with `F_NOCACHE`, without the
  /// bytes staying in the file cache as well.
  ///
  /// The system reads around its cache only where the file offset and the memory it lands in
  /// sit at the same place in a page. A tensor in a shard and the slot it goes to almost never
  /// do, and a read that misses that goes through the cache and stays there: DeepSeek-V4.1's
  /// experts left 20 GB of shards cached within one prompt, and the system compressed other
  /// programs to keep them. So each piece is read as whole pages into page-aligned scratch
  /// memory, and copied across from there.
  public func readUncached(
    from descriptor: Int32, offset: Int, into range: Range<Int>
  ) throws {
    guard range.lowerBound >= 0, range.upperBound <= byteCount else {
      throw BonsaiError.shapeMismatch("read of \(range) runs past a \(byteCount)-byte buffer")
    }
    let page = Self.pageSize
    let piece = Scratch.bytes - page
    let failure = FirstFailure()
    DispatchQueue.concurrentPerform(iterations: (range.count + piece - 1) / piece) { index in
      let done = index * piece
      let count = min(piece, range.count - done)
      let head = (offset + done) % page
      do {
        try Scratch.borrow { scratch in
          try Self.fill(
            scratch, from: descriptor, offset: offset + done - head,
            count: (head + count + page - 1) / page * page, needed: head + count)
          memcpy(pointer.advanced(by: range.lowerBound + done), scratch.advanced(by: head), count)
        }
      } catch {
        failure.record(error)
      }
    }
    if let error = failure.error { throw error }
  }

  /// Reads up to `count` bytes into `target`, trying again as `read` describes, and fails only
  /// if the file ends before `needed` of them.
  @discardableResult
  private static func fill(
    _ target: UnsafeMutableRawPointer, from descriptor: Int32, offset: Int, count: Int,
    needed: Int
  ) throws -> Int {
    var done = 0
    var failures = 0
    while done < needed {
      let got = pread(descriptor, target.advanced(by: done), count - done, off_t(offset + done))
      if got > 0 {
        done += got
        continue
      }
      let code = got < 0 ? errno : 0
      if code == EINTR { continue }
      if code == EIO || code == EAGAIN, failures < readRetries {
        Thread.sleep(forTimeInterval: 0.5 * Double(1 << failures))
        failures += 1
        continue
      }
      let reason = code == 0 ? "the file ends first" : String(cString: strerror(code))
      throw BonsaiError.missingWeight("read \(done) of \(needed) bytes at \(offset): \(reason)")
    }
    return done
  }

  /// How many times a failed read is tried again: after half a second, then doubling, about
  /// sixteen seconds in all on top of the kernel's own retries of the transfer.
  static let readRetries = 5

  /// An array over a slice of this buffer. The buffer outlives the array: MLX is handed a
  /// retain rather than the allocation, so a slot can be wrapped many times and freed once.
  public func array(byteOffset: Int = 0, shape: [Int], dtype: DType) -> MLXArray {
    let retained = Unmanaged.passRetained(self).toOpaque()
    let dims = shape.map { Int32($0) }
    let ctx = dims.withUnsafeBufferPointer { dimensions in
      mlx_array_new_data_managed_payload(
        pointer.advanced(by: byteOffset), dimensions.baseAddress, Int32(shape.count),
        dtype.cmlxDtype, retained,
        { payload in
          guard let payload else { return }
          Unmanaged<ResidentBuffer>.fromOpaque(payload).release()
        })
    }
    return MLXArray(ctx)
  }
}

/// Page-aligned memory an uncached read lands in before it is copied to where it belongs, kept
/// from one read to the next: a reader holds one piece at a time, and only as many readers run
/// as a batch of reads has at once.
private enum Scratch {
  static let bytes = 4 << 20
  private static let lock = NSLock()
  nonisolated(unsafe) private static var idle: [UnsafeMutableRawPointer] = []

  static func borrow<R>(_ body: (UnsafeMutableRawPointer) throws -> R) throws -> R {
    lock.lock()
    let kept = idle.popLast()
    lock.unlock()
    let scratch = try kept ?? allocate()
    defer {
      lock.lock()
      idle.append(scratch)
      lock.unlock()
    }
    return try body(scratch)
  }

  private static func allocate() throws -> UnsafeMutableRawPointer {
    var raw: UnsafeMutableRawPointer?
    guard posix_memalign(&raw, ResidentBuffer.pageSize, bytes) == 0, let raw else {
      throw BonsaiError.missingComponent("could not allocate \(bytes) bytes to read into")
    }
    return raw
  }
}
