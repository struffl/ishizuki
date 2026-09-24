// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DeepSeek-ViT and the aligner that folds its patches into the language model's width.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXFast
import MLXNN

/// A picture as V4.1 reads it: patches of 14 pixels, and the grid the aligner makes of them.
public struct DeepSeekImage: @unchecked Sendable {
  public var patches: MLXArray
  public var patchRows: Int
  public var patchColumns: Int

  public var rows: Int { (patchRows + 2) / 3 }
  public var columns: Int { (patchColumns + 2) / 3 }
  /// The span the picture takes in the prompt: a start, every row of the grid ended by a
  /// newline, and an end.
  public var spanLength: Int { rows * (columns + 1) + 2 }
}

/// A ViT trained from scratch with 2D rope and no class token, then a 3x3 pixel-unshuffle and a
/// two-layer MLP, so nine patches become one token of the language model.
public final class DeepSeekVision: @unchecked Sendable {
  struct Block {
    let norm1: MLXArray
    let qkv: DenseLinear
    let out: DenseLinear
    let norm2: MLXArray
    let up: DenseLinear
    let down: DenseLinear
  }

  let patchEmbed: DenseLinear
  let blocks: [Block]
  let norm: MLXArray
  let alignerUp: DenseLinear
  let alignerDown: DenseLinear
  let start: MLXArray
  let end: MLXArray
  let newline: MLXArray
  let heads: Int
  let theta: Float
  let ratio: Int
  public let patchSize: Int
  public let minPixels: Int
  public let maxTokens: Int

  /// The tower a release ships, sized by the `vision_config` its config.json carries.
  public init(weights: DeepSeekWeights) throws {
    let raw = try Data(contentsOf: weights.checkpoint.directory.appending(path: "config.json"))
    guard let root = try JSONSerialization.jsonObject(with: raw) as? [String: Any],
      let vision = root["vision_config"] as? [String: Any]
    else { throw BonsaiError.missingComponent("this release has no vision_config") }
    func dense(_ prefix: String, bias: Bool = true) throws -> DenseLinear {
      let weight = try weights.array(prefix + ".weight")
      return DenseLinear(weight: weight, bias: bias ? try weights.array(prefix + ".bias") : nil)
    }
    let layers = (vision["num_hidden_layers"] as? NSNumber)?.intValue ?? 0
    self.heads = (vision["num_attention_heads"] as? NSNumber)?.intValue ?? 16
    self.theta = (vision["rope_theta"] as? NSNumber)?.floatValue ?? 10000
    self.ratio = (vision["downsample_ratio"] as? NSNumber)?.intValue ?? 3
    self.patchSize = (vision["patch_size"] as? NSNumber)?.intValue ?? 14
    self.minPixels = (vision["min_pixels"] as? NSNumber)?.intValue ?? 544 * 544
    self.maxTokens = (vision["max_image_tokens"] as? NSNumber)?.intValue ?? 1024
    self.patchEmbed = try dense("vision.patch_embed.proj")
    self.blocks = try (0..<layers).map { layer in
      let prefix = "vision.blocks.\(layer)"
      return Block(
        norm1: try weights.array(prefix + ".norm1.weight"), qkv: try dense(prefix + ".attn.wqkv"),
        out: try dense(prefix + ".attn.wo"), norm2: try weights.array(prefix + ".norm2.weight"),
        up: try dense(prefix + ".mlp.w1", bias: false),
        down: try dense(prefix + ".mlp.w2", bias: false))
    }
    self.norm = try weights.array("vision.norm.weight")
    self.alignerUp = try dense("aligner.w1")
    self.alignerDown = try dense("aligner.w2")
    self.start = try weights.array("image_start")
    self.end = try weights.array("image_end")
    self.newline = try weights.array("image_newline")
  }

  /// 2D rotation for a `rows x columns` grid: the first half of each head's rotary channels
  /// turn with the row, the second half with the column.
  private func rotation(rows: Int, columns: Int, headDim: Int) -> (MLXArray, MLXArray) {
    let dims = headDim / 2
    let rates = (0..<(dims / 2)).map { 1 / Foundation.pow(theta, Float(2 * $0) / Float(dims)) }
    var angles: [Float] = []
    angles.reserveCapacity(rows * columns * dims)
    for row in 0..<rows {
      for column in 0..<columns {
        for rate in rates { angles.append(Float(row) * rate) }
        for rate in rates { angles.append(Float(column) * rate) }
      }
    }
    let table = MLXArray(angles, [rows * columns, 1, dims])
    return (cos(table), sin(table))
  }

  private func rotate(_ x: MLXArray, _ cosine: MLXArray, _ sine: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let a = x[.ellipsis, ..<half].asType(.float32)
    let b = x[.ellipsis, half...].asType(.float32)
    return concatenated([a * cosine - b * sine, b * cosine + a * sine], axis: -1).asType(x.dtype)
  }

  /// The aligner's rows for one picture, `[rows * columns, width]`, in reading order.
  public func encode(_ image: DeepSeekImage, dtype: DType) -> MLXArray {
    let count = image.patchRows * image.patchColumns
    var x = patchEmbed(image.patches.reshaped([count, -1]).asType(dtype))
    let width = x.dim(-1)
    let headDim = width / heads
    let (cosine, sine) = rotation(
      rows: image.patchRows, columns: image.patchColumns, headDim: headDim)
    for block in blocks {
      let normed = MLXFast.rmsNorm(x, weight: block.norm1.asType(x.dtype), eps: 1e-6)
      let parts = split(block.qkv(normed), parts: 3, axis: -1)
      let q = rotate(parts[0].reshaped([count, heads, headDim]), cosine, sine)
      let k = rotate(parts[1].reshaped([count, heads, headDim]), cosine, sine)
      let v = parts[2].reshaped([count, heads, headDim])
      let attended = MLXFast.scaledDotProductAttention(
        queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
        keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
        values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
        scale: 1 / Float(headDim).squareRoot(), mask: nil)
      x = x + block.out(attended[0].transposed(1, 0, 2).reshaped([count, width]))
      let gated = split(
        block.up(MLXFast.rmsNorm(x, weight: block.norm2.asType(x.dtype), eps: 1e-6)), parts: 2,
        axis: -1)
      x = x + block.down(silu(gated[0]) * gated[1])
    }
    x = MLXFast.rmsNorm(x, weight: norm.asType(x.dtype), eps: 1e-6)

    let rows = image.rows * ratio
    let columns = image.columns * ratio
    var grid = x.reshaped([image.patchRows, image.patchColumns, width]).transposed(2, 0, 1)
    if rows > image.patchRows || columns > image.patchColumns {
      grid = padded(
        grid,
        widths: [.init((0, 0)), .init((0, rows - image.patchRows)),
          .init((0, columns - image.patchColumns))])
    }
    let folded = grid.reshaped([width, image.rows, ratio, image.columns, ratio])
      .transposed(1, 3, 0, 2, 4)
      .reshaped([image.rows * image.columns, width * ratio * ratio])
    return alignerDown(gelu(alignerUp(folded)))
  }

  /// A picture's whole span in the prompt, delimiters included, `[spanLength, width]`.
  public func span(_ image: DeepSeekImage, dtype: DType) -> MLXArray {
    let features = encode(image, dtype: dtype)
    var pieces: [MLXArray] = [start.reshaped([1, -1]).asType(dtype)]
    for row in 0..<image.rows {
      pieces.append(features[(row * image.columns)..<((row + 1) * image.columns)])
      pieces.append(newline.reshaped([1, -1]).asType(dtype))
    }
    pieces.append(end.reshaped([1, -1]).asType(dtype))
    return concatenated(pieces.map { $0.asType(dtype) }, axis: 0)
  }

  /// The pixel size a picture is padded to: the smallest multiple of the patch that holds it
  /// once it is at least `minPixels`, shrunk until its span fits `maxTokens`.
  public func plan(width: Int, height: Int) -> (width: Int, height: Int) {
    Self.plan(
      width: width, height: height, patchSize: patchSize, ratio: ratio, minPixels: minPixels,
      maxTokens: maxTokens)
  }

  public static func plan(
    width: Int, height: Int, patchSize: Int, ratio: Int, minPixels: Int, maxTokens: Int
  ) -> (width: Int, height: Int) {
    var w = Double(width)
    var h = Double(height)
    if w * h > 0, w * h < Double(minPixels) {
      let scale = (Double(minPixels) / (w * h)).squareRoot()
      w = Double(Int(w * scale))
      h = Double(Int(h * scale))
    }
    let p = Double(patchSize)
    var bestWidth = Int((w / p).rounded(.up) * p)
    var bestHeight = Int((h / p).rounded(.up) * p)
    func tokens(_ bw: Int, _ bh: Int) -> Int {
      let rows = Int((Double(bh / patchSize) / Double(ratio)).rounded(.up))
      let columns = Int((Double(bw / patchSize) / Double(ratio)).rounded(.up))
      return rows * (columns + 1) + 2
    }
    if tokens(bestWidth, bestHeight) > maxTokens {
      let aspect = h / w
      let maxW = ((Double(maxTokens) - 2) / aspect + 0.25).squareRoot() - 0.5
      let maxH = maxW * aspect
      let cell = patchSize * ratio
      if maxW < 1 {
        bestHeight = (maxTokens - 2) / 2 * cell
        bestWidth = cell
      } else if maxH < 1 {
        bestHeight = cell
        bestWidth = (maxTokens - 3) * cell
      } else {
        let beta = min(
          Foundation.floor(maxW) * Double(cell) / w, Foundation.floor(maxH) * Double(cell) / h)
        bestHeight = Int(Foundation.floor(h * beta / p)) * patchSize
        bestWidth = Int(Foundation.floor(w * beta / p)) * patchSize
      }
    }
    return (bestWidth, bestHeight)
  }

  /// A picture fitted inside its planned size, letterboxed in mid grey, scaled to [-1, 1] and cut
  /// into patches in reading order.
  public func prepare(_ image: CGImage) throws -> DeepSeekImage {
    let (width, height) = plan(width: image.width, height: image.height)
    let scale = min(Double(width) / Double(image.width), Double(height) / Double(image.height))
    let drawn = (
      width: max(1, Int((Double(image.width) * scale).rounded())),
      height: max(1, Int((Double(image.height) * scale).rounded()))
    )
    let left = Int((Double(width - drawn.width) * 0.5).rounded(.toNearestOrEven))
    let top = Int((Double(height - drawn.height) * 0.5).rounded(.toNearestOrEven))
    let bytesPerRow = width * 4
    var buffer = [UInt8](repeating: 127, count: bytesPerRow * height)
    guard
      let context = CGContext(
        data: &buffer, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else {
      throw BonsaiError.imageProcessing("could not create a \(width)x\(height) bitmap")
    }
    context.interpolationQuality = .high
    context.draw(
      image,
      in: CGRect(
        x: left, y: height - top - drawn.height, width: drawn.width, height: drawn.height))
    var planar = [Float](repeating: 0, count: 3 * width * height)
    for y in 0..<height {
      for x in 0..<width {
        for channel in 0..<3 {
          let value = Float(buffer[y * bytesPerRow + x * 4 + channel]) / 255
          planar[(channel * height + y) * width + x] = (value - 0.5) / 0.5
        }
      }
    }
    return Self.patchify(MLXArray(planar, [3, height, width]), patchSize: patchSize)
  }

  public func prepare(contentsOf url: URL) throws -> DeepSeekImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      throw BonsaiError.imageProcessing("could not decode an image from \(url.path)")
    }
    return try prepare(image)
  }

  /// `[3, height, width]` pixels as patches, one row of `3 * patch * patch` per patch.
  public static func patchify(_ pixels: MLXArray, patchSize: Int) -> DeepSeekImage {
    let rows = pixels.dim(1) / patchSize
    let columns = pixels.dim(2) / patchSize
    let patches = pixels.reshaped([3, rows, patchSize, columns, patchSize])
      .transposed(1, 3, 0, 2, 4)
      .reshaped([rows * columns, 3 * patchSize * patchSize])
    return DeepSeekImage(patches: patches, patchRows: rows, patchColumns: columns)
  }
}
