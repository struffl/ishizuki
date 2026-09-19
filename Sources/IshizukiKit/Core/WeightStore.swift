// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public final class WeightStore: @unchecked Sendable {
  public let arrays: [String: MLXArray]

  public init(arrays: [String: MLXArray]) {
    self.arrays = arrays
  }

  /// A pack is either one safetensors file or a set of shards named by an index. Both land in
  /// the same flat name table, so nothing downstream needs to know which it was.
  public convenience init(directory: URL, file: String = "model.safetensors") throws {
    let fm = FileManager.default
    let single = directory.appending(path: file)
    if fm.fileExists(atPath: single.path) {
      self.init(arrays: try loadArrays(url: single))
      return
    }

    let index = directory.appending(path: file + ".index.json")
    guard fm.fileExists(atPath: index.path) else {
      throw BonsaiError.missingWeight(
        "no \(file) and no \(file).index.json in \(directory.path)")
    }

    let object = try JSONSerialization.jsonObject(with: try Data(contentsOf: index))
    guard let map = (object as? [String: Any])?["weight_map"] as? [String: String] else {
      throw BonsaiError.missingWeight("\(file).index.json has no weight_map")
    }

    var arrays: [String: MLXArray] = [:]
    arrays.reserveCapacity(map.count)
    for shard in Set(map.values).sorted() {
      let url = directory.appending(path: shard)
      guard fm.fileExists(atPath: url.path) else {
        throw BonsaiError.missingWeight("\(file).index.json names \(shard), which is not here")
      }
      for (name, array) in try loadArrays(url: url) {
        arrays[name] = array
      }
    }

    let missing = map.keys.filter { arrays[$0] == nil }
    guard missing.isEmpty else {
      throw BonsaiError.missingWeight(
        "\(missing.count) tensor(s) named by the index are absent from the shards, "
          + "starting with \(missing.sorted()[0])")
    }
    self.init(arrays: arrays)
  }

  public func callAsFunction(_ name: String) throws -> MLXArray {
    guard let array = arrays[name] else {
      throw BonsaiError.missingWeight(name)
    }
    return array
  }

  public func optional(_ name: String) -> MLXArray? { arrays[name] }

  public func has(_ name: String) -> Bool { arrays[name] != nil }

  public func names(prefix: String) -> [String] {
    arrays.keys.filter { $0.hasPrefix(prefix) }.sorted()
  }
}

public struct PackedModuleFactory {
  public let store: WeightStore
  public let records: [String: BonsaiConfig.PackedModuleRecord]
  public let tensorPrefix: String
  public let quantization: BonsaiConfig.QuantizationConfig

  public init(store: WeightStore, config: BonsaiConfig, tensorPrefix: String) {
    self.store = store
    self.records = Dictionary(
      uniqueKeysWithValues: config.modules.map { ($0.path, $0) })
    self.tensorPrefix = tensorPrefix
    self.quantization = config.quantization
  }

  public var groupSize: Int { quantization.groupSize }
  public var bits: Int { quantization.bits }

  /// An imatrix pass records its overrides against the tensor name, so the lookup key is the
  /// prefixed one, not the bare module path the model builds with.
  public func quant(for path: String) -> BonsaiConfig.ModuleQuant {
    quantization.module(tensorPrefix + path)
  }

  public func linear(_ path: String) throws -> PackedLinear {
    if let record = records[path] {
      guard !record.embedding else {
        throw BonsaiError.shapeMismatch("\(path) is an embedding, not a linear")
      }
      return try packedLinear(path, block: record.block)
    }
    guard records.isEmpty else {
      throw BonsaiError.missingWeight("no packed-module record for \(path)")
    }
    return try packedLinear(path, block: 0)
  }

  public func tiedHead(_ path: String) throws -> PackedLinear {
    try packedLinear(path, block: records[path]?.block ?? 0)
  }

  public func embedding(_ path: String) throws -> PackedEmbedding {
    let block: Int
    if let record = records[path] {
      guard record.embedding else {
        throw BonsaiError.shapeMismatch("\(path) is not marked as an embedding")
      }
      block = record.block
    } else {
      guard records.isEmpty else {
        throw BonsaiError.missingWeight("no packed-module record for \(path)")
      }
      block = 0
    }
    let key = tensorPrefix + path
    let entry = quant(for: path)
    return try PackedEmbedding(
      weight: store(key + ".weight"),
      scales: store(key + ".scales"),
      biases: store(key + ".biases"),
      signs: store.optional(key + ".signs"),
      block: block,
      groupSize: entry.groupSize,
      bits: entry.bits)
  }

  private func packedLinear(_ path: String, block: Int) throws -> PackedLinear {
    let key = tensorPrefix + path
    let entry = quant(for: path)
    return try PackedLinear(
      weight: store(key + ".weight"),
      scales: store(key + ".scales"),
      biases: store(key + ".biases"),
      signs: store.optional(key + ".signs"),
      block: block,
      groupSize: entry.groupSize,
      bits: entry.bits)
  }
}

public final class DenseLinear: @unchecked Sendable {
  public let weight: MLXArray
  public let bias: MLXArray?

  public init(weight: MLXArray, bias: MLXArray?) {
    self.weight = weight
    self.bias = bias
  }

  public convenience init(store: WeightStore, prefix: String, bias: Bool = true) throws {
    self.init(
      weight: try store(prefix + ".weight"),
      bias: bias ? store.optional(prefix + ".bias") : nil)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var y = matmul(x, weight.T)
    if let bias { y = y + bias }
    return y
  }
}

public final class DenseLayerNorm: @unchecked Sendable {
  public let weight: MLXArray
  public let bias: MLXArray?
  public let eps: Float

  public init(weight: MLXArray, bias: MLXArray?, eps: Float = 1e-6) {
    self.weight = weight
    self.bias = bias
    self.eps = eps
  }

  public convenience init(store: WeightStore, prefix: String, eps: Float = 1e-6) throws {
    self.init(
      weight: try store(prefix + ".weight"),
      bias: store.optional(prefix + ".bias"),
      eps: eps)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    MLX.layerNorm(x, weight: weight, bias: bias, eps: eps)
  }
}
