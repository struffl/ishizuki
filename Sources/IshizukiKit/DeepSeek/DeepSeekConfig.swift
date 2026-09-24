// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The shape of a DeepSeek-V4.1 checkpoint, read from the config it ships.

import Foundation

/// Every number the V4.1 forward pass needs, under the names this runtime uses.
///
/// A release config nests these under `text_config` with Hugging Face names, and DeepSeek's own
/// inference config writes them flat with its own. Both read here, so a fixture written by the
/// reference and a checkpoint pulled from the hub arrive at the same struct.
public struct DeepSeekConfig: Sendable, Equatable {
  public var vocabSize: Int
  public var dim: Int
  public var moeInterDim: Int
  public var layers: Int
  public var heads: Int
  public var headDim: Int
  public var ropeDim: Int
  public var qLoraRank: Int
  public var oLoraRank: Int
  public var oGroups: Int
  public var swigluLimit: Float
  public var normEps: Float

  public var ropeTheta: Float
  public var compressRopeTheta: Float
  public var ropeFactor: Float
  public var originalContext: Int
  public var betaFast: Float
  public var betaSlow: Float

  public var routedExperts: Int
  public var activatedExperts: Int
  public var routeScale: Float
  public var normTopkProb: Bool

  public var window: Int
  public var compressRatios: [Int]
  public var kvSourceLayers: [Int]
  public var indexSourceLayers: [Int]
  public var indexHeads: Int
  public var indexHeadDim: Int
  public var indexTopK: Int
  public var candidateSourceLayer: Int
  public var candidateTopKBlocks: Int
  public var candidateBlockSize: Int

  public var hcMult: Int
  public var hcSinkhornIterations: Int
  public var hcEps: Float

  public var engramLayers: [Int]
  public var engramRows: [Int]
  public var engramMaxNgram: Int
  public var engramBucketBase: Int
  public var engramHeads: Int
  public var engramHeadDim: Int
  public var engramPadTokenId: Int
  public var engramCompressedVocab: Int

  public var draftLayers: Int
  public var draftBlockSize: Int
  public var draftNoiseTokenId: Int
  public var draftTargetLayers: [Int]
  public var draftMarkovRank: Int
  public var draftRoutedExperts: Int
  public var draftActivatedExperts: Int

  public var bosTokenId: Int?
  public var eosTokenId: Int?
  public var imageTokenId: Int?

  public static let modelTypes: Set<String> = ["deepseek_v41", "deepseek_v41_text"]

  /// A layer at or past `layers` is one of the draft head's.
  public func experts(layer: Int) -> (routed: Int, activated: Int) {
    guard layer >= layers else { return (routedExperts, activatedExperts) }
    return (
      draftRoutedExperts > 0 ? draftRoutedExperts : routedExperts,
      draftActivatedExperts > 0 ? draftActivatedExperts : activatedExperts
    )
  }

  public func compressRatio(layer: Int) -> Int {
    layer < compressRatios.count ? compressRatios[layer] : 0
  }

  /// The most recent layer at or below `layer` that owns the compressed KV `layer` reads.
  public func kvSource(of layer: Int) -> Int? {
    kvSourceLayers.filter { $0 <= layer }.max()
  }

  /// The most recent layer at or below `layer` whose top-k selection `layer` attends through.
  public func indexSource(of layer: Int) -> Int? {
    indexSourceLayers.filter { $0 <= layer }.max()
  }

  public var hasEngram: Bool { !engramLayers.isEmpty }
  public var hasDraft: Bool { draftBlockSize > 0 && draftLayers > 0 }

  /// The config a directory ships, whichever of the two spellings it uses.
  public static func load(directory: URL) throws -> DeepSeekConfig {
    let data = try Data(contentsOf: directory.appending(path: "config.json"))
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw BonsaiError.unsupportedModel("config.json is not an object")
    }
    return try DeepSeekConfig(object)
  }

  /// Whether a raw config describes this architecture, without reading the rest of it.
  public static func describes(_ object: [String: Any]) -> Bool {
    if let type = object["model_type"] as? String, modelTypes.contains(type) { return true }
    if let text = object["text_config"] as? [String: Any], let type = text["model_type"] as? String {
      return modelTypes.contains(type)
    }
    return false
  }

  public init(_ root: [String: Any]) throws {
    let text = (root["text_config"] as? [String: Any]) ?? root
    let rope = (text["rope_scaling"] as? [String: Any]) ?? [:]

    func value(_ keys: String...) -> Any? {
      for key in keys {
        if let found = text[key] { return found }
        if let found = root[key] { return found }
      }
      return nil
    }
    func int(_ keys: String...) throws -> Int {
      for key in keys {
        if let n = (text[key] ?? root[key]) as? NSNumber { return n.intValue }
      }
      throw BonsaiError.unsupportedModel("config.json has no \(keys[0])")
    }
    func optionalInt(_ keys: String..., default fallback: Int) -> Int {
      for key in keys {
        if let n = (text[key] ?? root[key]) as? NSNumber { return n.intValue }
      }
      return fallback
    }
    func float(_ keys: String..., default fallback: Float) -> Float {
      for key in keys {
        if let n = (text[key] ?? root[key]) as? NSNumber { return n.floatValue }
      }
      return fallback
    }
    func ints(_ keys: String...) -> [Int] {
      for key in keys {
        if let list = (text[key] ?? root[key]) as? [NSNumber] { return list.map(\.intValue) }
      }
      return []
    }
    func ropeFloat(_ key: String, _ flat: String, default fallback: Float) -> Float {
      if let n = rope[key] as? NSNumber { return n.floatValue }
      return float(flat, default: fallback)
    }

    self.vocabSize = try int("vocab_size")
    self.dim = try int("hidden_size", "dim")
    self.moeInterDim = try int("moe_intermediate_size", "moe_inter_dim")
    self.layers = try int("num_hidden_layers", "n_layers")
    self.heads = try int("num_attention_heads", "n_heads")
    self.headDim = try int("head_dim")
    self.ropeDim = try int("qk_rope_head_dim", "rope_head_dim")
    self.qLoraRank = try int("q_lora_rank")
    self.oLoraRank = try int("o_lora_rank")
    self.oGroups = try int("o_groups")
    self.swigluLimit = float("swiglu_limit", default: 0)
    self.normEps = float("rms_norm_eps", "norm_eps", default: 1e-20)

    self.ropeTheta = float("rope_theta", default: 10000)
    self.compressRopeTheta = float("compress_rope_theta", default: 160000)
    self.ropeFactor = ropeFloat("factor", "rope_factor", default: 1)
    self.originalContext =
      (rope["original_max_position_embeddings"] as? NSNumber)?.intValue
      ?? optionalInt("original_seq_len", default: 0)
    self.betaFast = ropeFloat("beta_fast", "beta_fast", default: 32)
    self.betaSlow = ropeFloat("beta_slow", "beta_slow", default: 1)

    self.routedExperts = try int("n_routed_experts")
    self.activatedExperts = try int("num_experts_per_tok", "n_activated_experts")
    self.routeScale = float("routed_scaling_factor", "route_scale", default: 1)
    self.normTopkProb = (value("norm_topk_prob") as? Bool) ?? true
    if let scoring = value("scoring_func", "score_func") as? String, scoring != "sqrtsoftplus" {
      throw BonsaiError.unsupportedModel("DeepSeek-V4.1 routing by \(scoring) is not one this reads")
    }

    self.window = try int("sliding_window", "window_size")
    self.compressRatios = ints("compress_ratios")
    self.kvSourceLayers = ints("kv_source_layer_ids", "kv_source_layers")
    self.indexSourceLayers = ints("index_source_layer_ids", "index_source_layers")
    self.indexHeads = try int("index_n_heads")
    self.indexHeadDim = try int("index_head_dim")
    self.indexTopK = try int("index_topk")
    self.candidateSourceLayer = optionalInt(
      "candidate_source_layer_id", "candidate_source_layer", default: -1)
    self.candidateTopKBlocks = optionalInt("candidate_topk_blocks", default: 0)
    self.candidateBlockSize = optionalInt("candidate_block_size", default: 0)

    self.hcMult = try int("hc_mult")
    self.hcSinkhornIterations = optionalInt("hc_sinkhorn_iters", default: 20)
    self.hcEps = float("hc_eps", default: 1e-6)

    self.engramLayers = ints("engram_layer_ids")
    self.engramRows = ints("engram_num_embeddings")
    self.engramMaxNgram = optionalInt("engram_max_ngram_size", default: 1)
    self.engramBucketBase = optionalInt("engram_vocab_size", default: 0)
    self.engramHeads = optionalInt("engram_n_heads", default: 0)
    self.engramHeadDim = optionalInt("engram_head_dim", default: 0)
    self.engramPadTokenId = optionalInt("engram_pad_token_id", "engram_pad_id", default: 2)
    self.engramCompressedVocab = optionalInt("engram_compressed_vocab_size", default: 0)

    self.draftLayers = optionalInt("num_nextn_predict_layers", "n_mtp_layers", default: 0)
    self.draftBlockSize = optionalInt("dspark_block_size", default: 0)
    self.draftNoiseTokenId = optionalInt("dspark_noise_token_id", default: 0)
    self.draftTargetLayers = ints("dspark_target_layer_ids")
    self.draftMarkovRank = optionalInt("dspark_markov_rank", default: 256)
    self.draftRoutedExperts = optionalInt("dspark_n_routed_experts", default: 0)
    self.draftActivatedExperts = optionalInt(
      "dspark_num_experts_per_tok", "dspark_n_activated_experts", default: 0)

    self.bosTokenId = (root["bos_token_id"] as? NSNumber)?.intValue
    self.eosTokenId = (root["eos_token_id"] as? NSNumber)?.intValue
    self.imageTokenId = (root["image_token_id"] as? NSNumber)?.intValue

    try validate()
  }

  private func validate() throws {
    guard compressRatios.count >= layers else {
      throw BonsaiError.unsupportedModel(
        "compress_ratios names \(compressRatios.count) layers of \(layers)")
    }
    guard headDim > ropeDim, ropeDim % 2 == 0, headDim % 32 == 0 else {
      throw BonsaiError.unsupportedModel("head_dim \(headDim) with rope \(ropeDim) is not a V4.1 head")
    }
    guard heads % oGroups == 0 else {
      throw BonsaiError.unsupportedModel("\(heads) heads do not split into \(oGroups) groups")
    }
    for layer in 0..<layers where compressRatio(layer: layer) > 0 {
      guard kvSource(of: layer) != nil, indexSource(of: layer) != nil else {
        throw BonsaiError.unsupportedModel("layer \(layer) compresses but nothing above it owns a cache")
      }
    }
    guard engramLayers.count == engramRows.count else {
      throw BonsaiError.unsupportedModel("engram layers and table sizes disagree")
    }
  }
}
