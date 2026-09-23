// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// Which order a linear-attention layer's value heads sit in when there are more of them than
/// key heads.
///
/// A checkpoint stores them grouped by key head — `[K0V0, K0V1, K1V0, K1V1, ...]` — so key head
/// `i` serves value heads `i * r ..< (i + 1) * r`. llama.cpp's converter retiles them to
/// `[K0V0, K1V0, K0V1, K1V1, ...]`, where key head `i` serves every value head congruent to `i`,
/// because that turns its broadcast into a plain repeat. Neither order is more correct; the
/// weights carry no mark of which one they are in, so the store that produced them says.
public enum ValueHeadLayout: Sendable {
  case grouped
  case tiled
}

public final class WeightStore: @unchecked Sendable {
  public let arrays: [String: MLXArray]
  /// Tensors still in GGML blocks. A name appears here or in `arrays`, never both.
  public let ggmlArrays: [String: GGUFBlocks]
  public let valueHeadLayout: ValueHeadLayout
  private let expertStores: [Int: ExpertStore]
  /// The n-gram table, when the pack keeps one beside itself. A single table serves the one
  /// layer that carries a PLE block.
  public let engrams: EngramStore?

  public init(
    arrays: [String: MLXArray], ggml: [String: GGUFBlocks] = [:],
    valueHeadLayout: ValueHeadLayout = .grouped, experts: [Int: ExpertStore] = [:],
    engrams: EngramStore? = nil
  ) {
    self.arrays = arrays
    self.ggmlArrays = ggml
    self.valueHeadLayout = valueHeadLayout
    self.expertStores = experts
    self.engrams = engrams
  }

  public func ggml(_ name: String) -> GGUFBlocks? { ggmlArrays[name] }

  /// Where the language model's tensors sit: nested under `language_model.` in a multimodal
  /// pack, at the top in a text-only one. Asked of the embedding as well as the final norm,
  /// because a hyper-connected model folds its streams through a mixer and ships no norm.
  public var languageModelPrefix: String {
    has("language_model.model.norm.weight") || has("language_model.model.embed_tokens.weight")
      ? "language_model." : ""
  }

  public func canonical(zeroCentredNorms: Bool) -> WeightStore {
    let names = Array(arrays.keys)
    guard TensorNaming.isHuggingFaceLayout(names) else { return self }
    var renamed: [String: MLXArray] = [:]
    renamed.reserveCapacity(arrays.count)
    for (name, array) in arrays {
      var tensor = TensorNaming.relayout(name, array, zeroCentredNorms: zeroCentredNorms)
      if tensor.dtype != array.dtype { tensor = tensor.asType(.float16) }
      renamed[TensorNaming.canonical(name)] = tensor
    }
    return WeightStore(
      arrays: renamed, ggml: ggmlArrays, valueHeadLayout: valueHeadLayout,
      experts: expertStores, engrams: engrams)
  }

  /// The routed experts of one layer, when the pack keeps them beside itself rather than in
  /// the shards. Nil means every expert is already in `arrays`.
  public func experts(layer: Int) -> ExpertStore? { expertStores[layer] }

  /// What the streamed layers have read so far, summed. Nil for a pack that holds its experts,
  /// which is what tells a readout there is nothing to say.
  public var expertTraffic: ExpertStore.Summary? {
    ExpertStore.Summary(layers: Array(expertStores.values))
  }

  /// Opens the per-layer expert files a repacked sparse model ships, if there are any.
  public func openingExperts(at directory: URL, slots: Int) throws -> WeightStore {
    let layoutURL = directory.appending(path: ExpertRepack.layoutFile)
    guard FileManager.default.fileExists(atPath: layoutURL.path) else { return self }
    let layout = try JSONDecoder().decode(
      ExpertLayout.self, from: try Data(contentsOf: layoutURL))

    var stores: [Int: ExpertStore] = [:]
    let folder = directory.appending(path: "experts")
    for name in (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
    where name.hasPrefix("layer_") && name.hasSuffix(".bin") {
      let digits = name.dropFirst("layer_".count).dropLast(".bin".count)
      guard let layer = Int(digits) else { continue }
      let own = directory.appending(path: ExpertRepack.layerLayoutFile(layer))
      let layerLayout =
        FileManager.default.fileExists(atPath: own.path)
        ? try JSONDecoder().decode(ExpertLayout.self, from: try Data(contentsOf: own)) : layout
      stores[layer] = try ExpertStore(
        url: folder.appending(path: name), layout: layerLayout, slots: slots)
    }
    return WeightStore(
      arrays: arrays, ggml: ggmlArrays, valueHeadLayout: valueHeadLayout, experts: stores,
      engrams: engrams)
  }

  /// Opens the n-gram table a per-layer-embedding model ships, if there is one. `capacity` is
  /// the most rows a single fetch may ask for: one chunk of tokens times the head count.
  public func openingEngrams(at directory: URL, capacity: Int = 8192) throws -> WeightStore {
    let layoutURL = directory.appending(path: EngramLayout.layoutFile)
    guard FileManager.default.fileExists(atPath: layoutURL.path) else { return self }
    let layout = try JSONDecoder().decode(
      EngramLayout.self, from: try Data(contentsOf: layoutURL))
    let store = try EngramStore(directory: directory, layout: layout, capacity: capacity)
    return WeightStore(
      arrays: arrays, ggml: ggmlArrays, valueHeadLayout: valueHeadLayout,
      experts: expertStores, engrams: store)
  }

  /// Folds the implied one into every zero-centred norm, for a pack that left them as the
  /// checkpoint wrote them. `expected` is the count the pack declared, zero when it did not say;
  /// a mismatch means the naming rule and the pack disagree about which norms moved.
  public func foldingCentredNorms(expected: Int) throws -> WeightStore {
    var folded = arrays
    var count = 0
    for (name, array) in arrays where TensorNaming.isZeroCentredNorm(name) {
      folded[name] = array.asType(.float32) + 1
      count += 1
    }
    guard expected == 0 || count == expected else {
      throw BonsaiError.shapeMismatch(
        "pack declares \(expected) zero-centred norm(s), found \(count)")
    }
    return WeightStore(
      arrays: folded, ggml: ggmlArrays, valueHeadLayout: valueHeadLayout,
      experts: expertStores, engrams: engrams)
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

  public func has(_ name: String) -> Bool { arrays[name] != nil || ggmlArrays[name] != nil }

  public func names(prefix: String) -> [String] {
    arrays.keys.filter { $0.hasPrefix(prefix) }.sorted()
  }

  /// A safetensors tensor arrives as a promise: nothing is read off disk until something
  /// evaluates it. This redeems a whole tower at once, so its shard is read in one pass
  /// rather than a tensor at a time through the forward pass.
  public func warm(prefix: String) {
    let pending = names(prefix: prefix).compactMap { arrays[$0] }
    guard !pending.isEmpty else { return }
    eval(pending)
  }
}

public struct PackedModuleFactory {
  public let store: WeightStore
  public let records: [String: BonsaiConfig.PackedModuleRecord]
  public let tensorPrefix: String
  public let quantization: BonsaiConfig.QuantizationConfig
  /// When set, every module comes back as a dense projection over the raw weight in `store`
  /// (no scales, no biases, no Hadamard block) rather than a quantized one. Calibration uses
  /// this to run the exact forward pass a checkpoint about to be quantized would produce,
  /// through the same `Attention`/`GatedDeltaNet`/`MLP` code the real packs run — nothing
  /// downstream of this factory needs to know the difference.
  public let dense: Bool
  /// The width activations run at. Float16 is what a pack runs in; a golden test raises it so
  /// that what it measures is the architecture rather than the rounding.
  public let activationDType: DType
  private let collector: ActivationCollector?

  public init(
    store: WeightStore, config: BonsaiConfig, tensorPrefix: String,
    dense: Bool = false, collector: ActivationCollector? = nil,
    activationDType: DType = .float16
  ) {
    self.activationDType = activationDType
    self.store = store
    self.records = Dictionary(
      uniqueKeysWithValues: config.modules.map { ($0.path, $0) })
    self.tensorPrefix = tensorPrefix
    self.quantization = config.quantization
    self.dense = dense
    self.collector = collector
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
    let key = tensorPrefix + path
    if let blocks = store.ggml(key + ".weight") {
      return PackedEmbedding(ggml: blocks)
    }
    if dense || (!store.has(key + ".scales") && store.has(key + ".weight")) {
      return PackedEmbedding(dense: try store(key + ".weight"), dtype: activationDType)
    }
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

  /// The same module can arrive packed or dense depending on what the quantizer decided to
  /// leave alone, so the small delta-net projections resolve by what is actually on disk.
  public func projection(_ path: String) throws -> any Projection {
    if store.ggml(tensorPrefix + path + ".weight") != nil
      || store.has(tensorPrefix + path + ".trellis")
    {
      return try linear(path)
    }
    if dense || store.has(tensorPrefix + path + ".scales") {
      return try linear(path)
    }
    return try DenseLinear(store: store, prefix: tensorPrefix + path)
  }

  private func packedLinear(_ path: String, block: Int) throws -> PackedLinear {
    let key = tensorPrefix + path
    if let blocks = store.ggml(key + ".weight") {
      return PackedLinear(ggml: blocks)
    }
    if let tensor = try EXL3Tensor(store: store, key: key) {
      return PackedLinear(exl3: tensor)
    }
    if dense {
      let weight = try store(key + ".weight")
      guard let collector else { return PackedLinear(dense: weight) }
      return PackedLinear(dense: weight) { x in collector.record(path: path, x: x) }
    }
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

/// A linear layer, however its weights happen to be stored.
public protocol Projection: Sendable {
  func callAsFunction(_ x: MLXArray) -> MLXArray
  var inputDim: Int { get }
  var outputDim: Int { get }
}

extension PackedLinear: Projection {}

public final class DenseLinear: Projection, @unchecked Sendable {
  public let weight: MLXArray
  public let bias: MLXArray?

  public var outputDim: Int { weight.dim(0) }
  public var inputDim: Int { weight.dim(1) }

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
    var y = matmul(x, weight.T.asType(x.dtype))
    if let bias { y = y + bias.asType(x.dtype) }
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
