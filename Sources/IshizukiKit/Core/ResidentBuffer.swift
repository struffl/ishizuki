// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
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

  /// Metal wants its allocations page aligned; 16 KiB is the page size on Apple silicon.
  public static let pageSize = 16384

  public init(byteCount: Int, alignment: Int = ResidentBuffer.pageSize) throws {
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
  }

  deinit { free(pointer) }

  /// Fills `range` of this buffer from `descriptor` at `offset`, in one read.
  @discardableResult
  public func read(
    from descriptor: Int32, offset: Int, into range: Range<Int>
  ) throws -> Int {
    guard range.lowerBound >= 0, range.upperBound <= byteCount else {
      throw BonsaiError.shapeMismatch("read of \(range) runs past a \(byteCount)-byte buffer")
    }
    var done = 0
    while done < range.count {
      let got = pread(
        descriptor, pointer.advanced(by: range.lowerBound + done), range.count - done,
        off_t(offset + done))
      guard got > 0 else {
        throw BonsaiError.missingWeight(
          "read \(done) of \(range.count) bytes at \(offset)")
      }
      done += got
    }
    return done
  }

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
