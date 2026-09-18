// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public struct MultimodalPrompt: @unchecked Sendable {
  public var tokens: [Int]
  public var embeddings: MLXArray
  public var positions: MLXArray
}

extension BonsaiModel {
  public func prepareMultimodal(tokens: [Int], images: [ProcessedImage]) throws
    -> MultimodalPrompt
  {
    guard let vision else {
      throw BonsaiError.missingComponent("this pack has no vision tower")
    }
    guard let imageToken = tokenizer.imageTokenId else {
      throw BonsaiError.missingComponent("the tokenizer has no <|image_pad|> token")
    }

    let placeholders = tokens.filter { $0 == imageToken }.count
    guard placeholders == images.count else {
      throw BonsaiError.imageProcessing(
        "prompt has \(placeholders) image placeholder(s) but \(images.count) image(s) "
          + "were supplied")
    }

    var expanded: [Int] = []
    expanded.reserveCapacity(tokens.count)
    var imageIndex = 0
    for token in tokens {
      if token == imageToken {
        expanded.append(
          contentsOf: Array(repeating: imageToken, count: images[imageIndex].tokenCount))
        imageIndex += 1
      } else {
        expanded.append(token)
      }
    }

    let ids = MLXArray(expanded.map { Int32($0) }).reshaped([1, expanded.count])
    let embeddings = text.embedTokens(ids)

    var cursor = 0
    for image in images {
      guard
        let start = expanded[cursor...].firstIndex(where: { $0 == imageToken })
      else { break }
      let features = vision(patches: image.patches, grid: image.grid)
        .asType(embeddings.dtype)
      embeddings[0..., start..<(start + image.tokenCount), 0...] =
        features.expandedDimensions(axis: 0)
      cursor = start + image.tokenCount
    }
    eval(embeddings)

    return MultimodalPrompt(
      tokens: expanded,
      embeddings: embeddings,
      positions: mropePositions(
        tokens: expanded, images: images, imageToken: imageToken))
  }

  func mropePositions(tokens: [Int], images: [ProcessedImage], imageToken: Int) -> MLXArray {
    let merge = config.visionConfig?.spatialMergeSize ?? 2
    var time = [Int32]()
    var height = [Int32]()
    var width = [Int32]()
    time.reserveCapacity(tokens.count)
    height.reserveCapacity(tokens.count)
    width.reserveCapacity(tokens.count)

    var next: Int32 = 0
    var index = 0
    var imageIndex = 0

    while index < tokens.count {
      if tokens[index] == imageToken, imageIndex < images.count {
        let image = images[imageIndex]
        let gridT = image.grid.t
        let gridH = image.grid.h / merge
        let gridW = image.grid.w / merge
        let start = next

        for t in 0..<gridT {
          for h in 0..<gridH {
            for w in 0..<gridW {
              time.append(start + Int32(t))
              height.append(start + Int32(h))
              width.append(start + Int32(w))
            }
          }
        }
        next = start + Int32(max(gridT, max(gridH, gridW)))
        index += image.tokenCount
        imageIndex += 1
      } else {
        time.append(next)
        height.append(next)
        width.append(next)
        next += 1
        index += 1
      }
    }

    return stacked(
      [MLXArray(time), MLXArray(height), MLXArray(width)], axis: 0)
  }
}
