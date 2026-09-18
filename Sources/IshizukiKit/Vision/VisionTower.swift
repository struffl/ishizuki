// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXNN

public final class VisionTower: @unchecked Sendable {
  public let config: BonsaiConfig.VisionConfig

  private let patchEmbedWeight: MLXArray
  private let patchEmbedBias: MLXArray
  private let positionEmbedding: MLXArray
  private let blocks: [VisionBlock]
  private let merger: PatchMerger
  private let gridPerSide: Int

  public init(config: BonsaiConfig.VisionConfig, store: WeightStore) throws {
    self.config = config

    let rawWeight = try store("vision_tower.patch_embed.proj.weight")
    let patchVolume =
      config.temporalPatchSize * config.patchSize * config.patchSize * config.inChannels
    self.patchEmbedWeight = rawWeight.reshaped([config.hiddenSize, patchVolume])
    self.patchEmbedBias = try store("vision_tower.patch_embed.proj.bias")
    self.positionEmbedding = try store("vision_tower.pos_embed.weight")

    let side = Int(Double(config.numPositionEmbeddings).squareRoot().rounded())
    guard side * side == config.numPositionEmbeddings else {
      throw BonsaiError.unsupportedModel(
        "num_position_embeddings \(config.numPositionEmbeddings) is not a square grid")
    }
    self.gridPerSide = side

    var built: [VisionBlock] = []
    built.reserveCapacity(config.depth)
    for index in 0..<config.depth {
      built.append(
        try VisionBlock(
          config: config, prefix: "vision_tower.blocks.\(index)", store: store))
    }
    self.blocks = built
    self.merger = try PatchMerger(config: config, store: store)
  }

  public func callAsFunction(patches: MLXArray, grid: (t: Int, h: Int, w: Int)) -> MLXArray {
    var hidden = embedPatches(patches)
    hidden = hidden + interpolatedPositionEmbedding(grid: grid)

    let rotary = rotaryPositionEmbedding(grid: grid)
    for block in blocks {
      hidden = block(hidden, rotary: rotary)
    }
    return merger(hidden)
  }

  private func embedPatches(_ patches: MLXArray) -> MLXArray {
    let count = patches.dim(0)
    let reordered =
      patches
      .reshaped([
        count, config.inChannels, config.temporalPatchSize, config.patchSize,
        config.patchSize,
      ])
      .transposed(0, 2, 3, 4, 1)
      .reshaped([count, -1])
    return matmul(reordered, patchEmbedWeight.T) + patchEmbedBias
  }

  private func interpolatedPositionEmbedding(grid: (t: Int, h: Int, w: Int)) -> MLXArray {
    let merge = config.spatialMergeSize
    let last = Float(gridPerSide - 1)

    let rows = linspace(0, last, count: grid.h)
    let columns = linspace(0, last, count: grid.w)

    let rowFloor = floor(rows)
    let columnFloor = floor(columns)
    let rowCeil = minimum(rowFloor + 1, last)
    let columnCeil = minimum(columnFloor + 1, last)
    let rowFrac = rows - rowFloor
    let columnFrac = columns - columnFloor

    let rowFloorIndex = rowFloor.asType(.int32)
    let rowCeilIndex = rowCeil.asType(.int32)
    let columnFloorIndex = columnFloor.asType(.int32)
    let columnCeilIndex = columnCeil.asType(.int32)

    let baseFloor = (rowFloorIndex * Int32(gridPerSide)).expandedDimensions(axis: 1)
    let baseCeil = (rowCeilIndex * Int32(gridPerSide)).expandedDimensions(axis: 1)
    let cFloor = columnFloorIndex.expandedDimensions(axis: 0)
    let cCeil = columnCeilIndex.expandedDimensions(axis: 0)

    let corners = [
      (
        baseFloor + cFloor,
        (1 - rowFrac).expandedDimensions(axis: 1)
          * (1 - columnFrac).expandedDimensions(axis: 0)
      ),
      (
        baseFloor + cCeil,
        (1 - rowFrac).expandedDimensions(axis: 1)
          * columnFrac.expandedDimensions(axis: 0)
      ),
      (
        baseCeil + cFloor,
        rowFrac.expandedDimensions(axis: 1)
          * (1 - columnFrac).expandedDimensions(axis: 0)
      ),
      (
        baseCeil + cCeil,
        rowFrac.expandedDimensions(axis: 1)
          * columnFrac.expandedDimensions(axis: 0)
      ),
    ]

    var accumulated: MLXArray?
    for (index, weight) in corners {
      let gathered = positionEmbedding[index.reshaped([-1])]
      let weighted = gathered * weight.reshaped([-1, 1]).asType(positionEmbedding.dtype)
      accumulated = accumulated.map { $0 + weighted } ?? weighted
    }
    var embedding = accumulated!

    if grid.t > 1 {
      embedding = tiled(embedding, repetitions: [grid.t, 1])
    }

    let hidden = embedding.dim(-1)
    return
      embedding
      .reshaped([grid.t, grid.h / merge, merge, grid.w / merge, merge, hidden])
      .transposed(0, 1, 3, 2, 4, 5)
      .reshaped([-1, hidden])
  }

  private func rotaryPositionEmbedding(grid: (t: Int, h: Int, w: Int)) -> MLXArray {
    let merge = config.spatialMergeSize
    let headDim = config.hiddenSize / config.numHeads
    let rotaryDim = headDim / 2

    let exponents =
      MLXArray(stride(from: 0, to: rotaryDim, by: 2).map { Float($0) })
      / Float(rotaryDim)
    let invFreq = 1.0 / pow(MLXArray(Float(10000)), exponents)
    let maxSide = max(grid.h, grid.w)
    let table =
      MLXArray(0..<maxSide).asType(.float32)
      .expandedDimensions(axis: 1) * invFreq

    var rowIndices = [Int32]()
    var columnIndices = [Int32]()
    rowIndices.reserveCapacity(grid.h * grid.w)
    columnIndices.reserveCapacity(grid.h * grid.w)
    for blockRow in 0..<(grid.h / merge) {
      for blockColumn in 0..<(grid.w / merge) {
        for intraRow in 0..<merge {
          for intraColumn in 0..<merge {
            rowIndices.append(Int32(blockRow * merge + intraRow))
            columnIndices.append(Int32(blockColumn * merge + intraColumn))
          }
        }
      }
    }
    if grid.t > 1 {
      rowIndices = Array(repeating: rowIndices, count: grid.t).flatMap { $0 }
      columnIndices = Array(repeating: columnIndices, count: grid.t).flatMap { $0 }
    }

    let rowFrequencies = table[MLXArray(rowIndices)]
    let columnFrequencies = table[MLXArray(columnIndices)]
    return concatenated([rowFrequencies, columnFrequencies], axis: -1)
  }
}

final class VisionBlock: @unchecked Sendable {
  private let norm1: DenseLayerNorm
  private let norm2: DenseLayerNorm
  private let qkv: DenseLinear
  private let proj: DenseLinear
  private let fc1: DenseLinear
  private let fc2: DenseLinear
  private let numHeads: Int
  private let headDim: Int
  private let scale: Float

  init(config: BonsaiConfig.VisionConfig, prefix: String, store: WeightStore) throws {
    self.numHeads = config.numHeads
    self.headDim = config.hiddenSize / config.numHeads
    self.scale = 1.0 / Float(headDim).squareRoot()
    self.norm1 = try DenseLayerNorm(store: store, prefix: prefix + ".norm1")
    self.norm2 = try DenseLayerNorm(store: store, prefix: prefix + ".norm2")
    self.qkv = try DenseLinear(store: store, prefix: prefix + ".attn.qkv")
    self.proj = try DenseLinear(store: store, prefix: prefix + ".attn.proj")
    self.fc1 = try DenseLinear(store: store, prefix: prefix + ".mlp.linear_fc1")
    self.fc2 = try DenseLinear(store: store, prefix: prefix + ".mlp.linear_fc2")
  }

  func callAsFunction(_ x: MLXArray, rotary: MLXArray) -> MLXArray {
    var hidden = x + attention(norm1(x), rotary: rotary)
    hidden = hidden + fc2(MLXNN.geluApproximate(fc1(norm2(hidden))))
    return hidden
  }

  private func attention(_ x: MLXArray, rotary: MLXArray) -> MLXArray {
    let length = x.dim(0)
    let projected = qkv(x).reshaped([length, 3, numHeads, headDim]).transposed(1, 0, 2, 3)

    var queries = applyVisionRope(projected[0], rotary: rotary)
    var keys = applyVisionRope(projected[1], rotary: rotary)
    var values = projected[2]

    queries = queries.transposed(1, 0, 2).expandedDimensions(axis: 0)
    keys = keys.transposed(1, 0, 2).expandedDimensions(axis: 0)
    values = values.transposed(1, 0, 2).expandedDimensions(axis: 0)

    let attended = MLXFast.scaledDotProductAttention(
      queries: queries, keys: keys, values: values, scale: scale, mask: nil)
    return proj(attended[0].transposed(1, 0, 2).reshaped([length, numHeads * headDim]))
  }

  private func applyVisionRope(_ x: MLXArray, rotary: MLXArray) -> MLXArray {
    let doubled = tiled(rotary, repetitions: [1, 2])
    let cosine = cos(doubled).expandedDimensions(axis: 1).asType(x.dtype)
    let sine = sin(doubled).expandedDimensions(axis: 1).asType(x.dtype)
    let half = x.dim(-1) / 2
    let rotated = concatenated(
      [-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    return x * cosine + rotated * sine
  }
}

final class PatchMerger: @unchecked Sendable {
  private let norm: DenseLayerNorm
  private let fc1: DenseLinear
  private let fc2: DenseLinear
  private let mergedSize: Int

  init(config: BonsaiConfig.VisionConfig, store: WeightStore) throws {
    self.mergedSize = config.hiddenSize * config.spatialMergeSize * config.spatialMergeSize
    self.norm = try DenseLayerNorm(store: store, prefix: "vision_tower.merger.norm")
    self.fc1 = try DenseLinear(store: store, prefix: "vision_tower.merger.linear_fc1")
    self.fc2 = try DenseLinear(store: store, prefix: "vision_tower.merger.linear_fc2")
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    fc2(MLXNN.geluApproximate(fc1(norm(x).reshaped([-1, mergedSize]))))
  }
}
