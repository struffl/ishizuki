// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Several residual streams rather than one, and the gates that read and write them.

import Foundation
import MLX
import MLXNN

/// RMS over each stream on its own, then the whole width scaled by one weight.
///
/// The streams are normalised apart so that one of them running hot cannot quieten the others,
/// which is the whole point of carrying more than one. The arithmetic is float32: these values
/// are summed the depth of the model and the widths are small enough that narrowing buys nothing.
func groupedRMSNorm(
  _ x: MLXArray, weight: MLXArray, groups: Int, width: Int, eps: Float
) -> MLXArray {
  let leading = Array(x.shape.dropLast())
  let grouped = x.reshaped(leading + [groups, width]).asType(.float32)
  let scaled = grouped * rsqrt(grouped.square().mean(axis: -1, keepDims: true) + eps)
  let weighted = scaled.reshaped(leading + [groups * width]) * weight.asType(.float32)
  return weighted.asType(x.dtype)
}

/// The residual stream, widened.
///
/// A plain decoder layer reads one stream and adds to it. This reads `count` of them, mixes them
/// into the single width a block expects, and writes the block's answer back into every stream
/// with its own gate. The streams are what the layer carries forward; the mixed view exists only
/// for the length of one block.
///
/// The mixing weight is low rank, so widening the residual costs a rank rather than a square.
public struct GatedResidual: @unchecked Sendable {
  /// One stream's worth of what a block reads, the streams it came from, and how much of the
  /// block's answer each stream takes back.
  public struct Opened: @unchecked Sendable {
    public var mixed: MLXArray
    public var streams: MLXArray
    public var injection: MLXArray?
  }

  let norm: MLXArray
  let down: any Projection
  let up: any Projection
  let inject: (any Projection)?
  let count: Int
  let width: Int
  let eps: Float

  public init(
    norm: MLXArray, down: any Projection, up: any Projection, inject: (any Projection)?,
    count: Int, width: Int, eps: Float
  ) {
    self.norm = norm
    self.down = down
    self.up = up
    self.inject = inject
    self.count = count
    self.width = width
    self.eps = eps
  }

  /// Normalises each stream on its own, then scales the lot. The norm runs in float32: the
  /// streams are summed many times over a deep model and the widths here are small enough that
  /// keeping them narrow buys nothing.
  func normalized(_ streams: MLXArray) -> MLXArray {
    groupedRMSNorm(streams, weight: norm, groups: count, width: width, eps: eps)
  }

  /// Opens the streams for one block: what it should read, and what it will need to write back.
  public func callAsFunction(_ streams: MLXArray) -> Opened {
    let leading = Array(streams.shape.dropLast())
    let normed = normalized(streams)

    // Dividing by the stream count before the gate keeps the mix the same size however many
    // streams a model carries.
    var mix = silu(down(normed) / Float(count))
    mix = sigmoid(up(mix))

    let mixed =
      (mix.reshaped(leading + [count, width]) * normed.reshaped(leading + [count, width]))
      .mean(axis: -2)

    guard let inject else {
      return Opened(mixed: mixed, streams: streams, injection: nil)
    }
    // Twice a sigmoid, so a stream can take more of the answer than it leaves behind.
    let injection = 2 * sigmoid(inject(normed) / Float(count))
    return Opened(mixed: mixed, streams: streams, injection: injection)
  }

  /// Writes a block's answer back into the streams it was opened from.
  public static func close(_ opened: Opened, with answer: MLXArray) -> MLXArray {
    guard let injection = opened.injection else { return answer }
    let spread = answer.expandedDimensions(axis: -2) * injection.expandedDimensions(axis: -1)
    let leading = Array(spread.shape.dropLast(2))
    let flat = spread.reshaped(leading + [spread.dim(-2) * spread.dim(-1)])
    return opened.streams + flat
  }
}
