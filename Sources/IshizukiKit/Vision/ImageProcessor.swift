// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import CoreGraphics
import Foundation
import ImageIO
import MLX

#if canImport(UniformTypeIdentifiers)
  import UniformTypeIdentifiers
#endif

public struct ProcessedImage: @unchecked Sendable {
  public var patches: MLXArray
  public var grid: (t: Int, h: Int, w: Int)
  /// The same picture as DeepSeek-V4.1's tower reads it, when that is the model it is for.
  public var deepseek: DeepSeekImage?

  public init(patches: MLXArray, grid: (t: Int, h: Int, w: Int), deepseek: DeepSeekImage? = nil) {
    self.patches = patches
    self.grid = grid
    self.deepseek = deepseek
  }

  public var tokenCount: Int { deepseek?.spanLength ?? (grid.t * grid.h * grid.w) / 4 }
}

public struct ImageProcessor: Sendable {
  public let patchSize: Int
  public let temporalPatchSize: Int
  public let mergeSize: Int
  public let mean: [Float]
  public let standardDeviation: [Float]
  public var maxPixels: Int
  public var minPixels: Int

  public init(
    config: BonsaiConfig.VisionConfig,
    maxPixels: Int = 1024 * 32 * 32,
    minPixels: Int = 65536,
    mean: [Float] = [0.5, 0.5, 0.5],
    standardDeviation: [Float] = [0.5, 0.5, 0.5]
  ) {
    self.patchSize = config.patchSize
    self.temporalPatchSize = config.temporalPatchSize
    self.mergeSize = config.spatialMergeSize
    self.maxPixels = maxPixels
    self.minPixels = minPixels
    self.mean = mean
    self.standardDeviation = standardDeviation
  }

  public func targetSize(width: Int, height: Int) -> (width: Int, height: Int) {
    let factor = patchSize * mergeSize
    func round(_ value: Double) -> Int {
      max(factor, Int((value / Double(factor)).rounded()) * factor)
    }
    var h = round(Double(height))
    var w = round(Double(width))

    let pixels = Double(height) * Double(width)
    if h * w > maxPixels {
      let beta = (pixels / Double(maxPixels)).squareRoot()
      h = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
      w = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
    } else if h * w < minPixels {
      let beta = (Double(minPixels) / pixels).squareRoot()
      h = max(factor, Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor)
      w = max(factor, Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor)
    }
    return (w, h)
  }

  public func process(contentsOf url: URL) throws -> ProcessedImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      throw BonsaiError.imageProcessing("could not decode an image from \(url.path)")
    }
    return try process(image: image)
  }

  public func process(image: CGImage) throws -> ProcessedImage {
    let (width, height) = targetSize(width: image.width, height: image.height)

    let bytesPerRow = width * 4
    var buffer = [UInt8](repeating: 0, count: bytesPerRow * height)
    guard
      let context = CGContext(
        data: &buffer, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else {
      throw BonsaiError.imageProcessing("could not create a \(width)×\(height) bitmap")
    }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    var planar = [Float](repeating: 0, count: 3 * height * width)
    let plane = height * width
    for y in 0..<height {
      for x in 0..<width {
        let source = y * bytesPerRow + x * 4
        let destination = y * width + x
        for channel in 0..<3 {
          let value = Float(buffer[source + channel]) / 255.0
          planar[channel * plane + destination] =
            (value - mean[channel]) / standardDeviation[channel]
        }
      }
    }

    let gridHeight = height / patchSize
    let gridWidth = width / patchSize
    let image = MLXArray(planar, [1, 3, height, width])

    let frames = tiled(image, repetitions: [temporalPatchSize, 1, 1, 1])

    let patches =
      frames
      .reshaped([
        1, temporalPatchSize, 3,
        gridHeight / mergeSize, mergeSize, patchSize,
        gridWidth / mergeSize, mergeSize, patchSize,
      ])
      .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
      .reshaped([
        gridHeight * gridWidth,
        3 * temporalPatchSize * patchSize * patchSize,
      ])

    return ProcessedImage(
      patches: patches.asType(.float16), grid: (t: 1, h: gridHeight, w: gridWidth))
  }
}
