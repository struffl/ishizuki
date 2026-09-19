// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

public struct BonsaiConfig: Codable, Sendable {
  public var schemaVersion: Int
  public var modelType: String
  public var baseModelType: String?
  public var textConfig: TextConfig
  public var visionConfig: VisionConfig?
  public var modules: [PackedModuleRecord]
  public var quantization: QuantizationConfig
  public var components: Components?
  public var tensorNamespace: String?
  public var gdnActivationLayout: String?
  public var requiresRuntime: String?

  public var imageTokenId: Int?
  public var videoTokenId: Int?
  public var visionStartTokenId: Int?
  public var visionEndTokenId: Int?

  enum CodingKeys: String, CodingKey {
    case schemaVersion = "schema_version"
    case modelType = "model_type"
    case baseModelType = "base_model_type"
    case textConfig = "text_config"
    case visionConfig = "vision_config"
    case modules, quantization, components
    case tensorNamespace = "tensor_namespace"
    case gdnActivationLayout = "gdn_activation_layout"
    case requiresRuntime = "requires_runtime"
    case imageTokenId = "image_token_id"
    case videoTokenId = "video_token_id"
    case visionStartTokenId = "vision_start_token_id"
    case visionEndTokenId = "vision_end_token_id"
  }

  public struct Components: Codable, Sendable {
    public var text: Bool
    public var vision: Bool
    public var mtp: Bool
  }

  /// One module's quantization, either the pack-wide default or a per-path override.
  public struct ModuleQuant: Codable, Sendable, Equatable {
    public var bits: Int
    public var groupSize: Int
    public var mode: String

    public init(bits: Int, groupSize: Int, mode: String = "affine") {
      self.bits = bits
      self.groupSize = groupSize
      self.mode = mode
    }

    enum CodingKeys: String, CodingKey {
      case bits
      case groupSize = "group_size"
      case mode
    }

    /// `mode` is optional in a pack's overrides: mlx_lm writes only the width and group size for
    /// a module that differs from the default, and a missing mode means the pack's own.
    public init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      self.bits = try container.decode(Int.self, forKey: .bits)
      self.groupSize = try container.decode(Int.self, forKey: .groupSize)
      self.mode = try container.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
    }
  }

  /// MLX writes the sensitivity-guided profiles oMLX calls oQ*e as one flat object: the
  /// pack-wide width at the top, then a nested object for every module the imatrix pass moved
  /// off it. Both forms decode here, so a uniform pack is just one with no overrides.
  public struct QuantizationConfig: Codable, Sendable {
    public var bits: Int
    public var groupSize: Int
    public var mode: String
    public var overrides: [String: ModuleQuant]

    public init(
      bits: Int, groupSize: Int, mode: String = "affine",
      overrides: [String: ModuleQuant] = [:]
    ) {
      self.bits = bits
      self.groupSize = groupSize
      self.mode = mode
      self.overrides = overrides
    }

    public var `default`: ModuleQuant {
      ModuleQuant(bits: bits, groupSize: groupSize, mode: mode)
    }

    public func module(_ path: String) -> ModuleQuant {
      overrides[path] ?? `default`
    }

    /// Every width the pack actually uses, for validation and for reporting.
    public var widths: Set<Int> {
      Set(overrides.values.map(\.bits)).union([bits])
    }

    private struct Key: CodingKey {
      var stringValue: String
      var intValue: Int? { nil }
      init?(stringValue: String) { self.stringValue = stringValue }
      init?(intValue: Int) { nil }
    }

    public init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: Key.self)
      var bits = 2
      var groupSize = 128
      var mode = "affine"
      var overrides: [String: ModuleQuant] = [:]

      for key in container.allKeys {
        switch key.stringValue {
        case "bits": bits = try container.decode(Int.self, forKey: key)
        case "group_size": groupSize = try container.decode(Int.self, forKey: key)
        case "mode": mode = try container.decode(String.self, forKey: key)
        default:
          // Anything else is a module path, or a scalar the runtime has no use for.
          if let entry = try? container.decode(ModuleQuant.self, forKey: key) {
            overrides[key.stringValue] = entry
          }
        }
      }

      self.init(bits: bits, groupSize: groupSize, mode: mode, overrides: overrides)
    }

    public func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: Key.self)
      try container.encode(bits, forKey: Key(stringValue: "bits")!)
      try container.encode(groupSize, forKey: Key(stringValue: "group_size")!)
      try container.encode(mode, forKey: Key(stringValue: "mode")!)
      for (path, entry) in overrides {
        try container.encode(entry, forKey: Key(stringValue: path)!)
      }
    }
  }

  public struct PackedModuleRecord: Codable, Sendable {
    public var path: String
    public var block: Int
    public var embedding: Bool
    public var dtype: String
  }

  public struct RopeParameters: Codable, Sendable {
    public var ropeType: String?
    public var ropeTheta: Float
    public var partialRotaryFactor: Float?
    public var mropeSection: [Int]?
    public var mropeInterleaved: Bool?
    public var factor: Float?
    public var originalMaxPositionEmbeddings: Int?

    enum CodingKeys: String, CodingKey {
      case ropeType = "rope_type"
      case ropeTheta = "rope_theta"
      case partialRotaryFactor = "partial_rotary_factor"
      case mropeSection = "mrope_section"
      case mropeInterleaved = "mrope_interleaved"
      case factor
      case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }
  }

  public struct TextConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var tieWordEmbeddings: Bool
    public var attnOutputGate: Bool?
    public var fullAttentionInterval: Int?
    public var layerTypes: [String]?

    public var linearNumValueHeads: Int
    public var linearNumKeyHeads: Int
    public var linearValueHeadDim: Int
    public var linearKeyHeadDim: Int
    public var linearConvKernelDim: Int

    public var ropeParameters: RopeParameters
    public var partialRotaryFactor: Float?
    public var outputGateType: String?
    public var mtpNumHiddenLayers: Int?

    /// Sparse feed-forward geometry. Absent, or an expert count of zero, means every layer's
    /// MLP is dense.
    public var numExperts: Int?
    public var numExpertsPerTok: Int?
    public var moeIntermediateSize: Int?
    public var sharedExpertIntermediateSize: Int?
    public var normTopkProb: Bool?
    /// Every `decoderSparseStep`-th layer is sparse; the rest are dense, as are any layer named
    /// in `mlpOnlyLayers`.
    public var decoderSparseStep: Int?
    public var mlpOnlyLayers: [Int]?
    public var bosTokenId: Int?
    public var eosTokenId: Int?

    enum CodingKeys: String, CodingKey {
      case modelType = "model_type"
      case hiddenSize = "hidden_size"
      case intermediateSize = "intermediate_size"
      case numHiddenLayers = "num_hidden_layers"
      case numAttentionHeads = "num_attention_heads"
      case numKeyValueHeads = "num_key_value_heads"
      case headDim = "head_dim"
      case rmsNormEps = "rms_norm_eps"
      case vocabSize = "vocab_size"
      case maxPositionEmbeddings = "max_position_embeddings"
      case tieWordEmbeddings = "tie_word_embeddings"
      case attnOutputGate = "attn_output_gate"
      case fullAttentionInterval = "full_attention_interval"
      case layerTypes = "layer_types"
      case linearNumValueHeads = "linear_num_value_heads"
      case linearNumKeyHeads = "linear_num_key_heads"
      case linearValueHeadDim = "linear_value_head_dim"
      case linearKeyHeadDim = "linear_key_head_dim"
      case linearConvKernelDim = "linear_conv_kernel_dim"
      case ropeParameters = "rope_parameters"
      case partialRotaryFactor = "partial_rotary_factor"
      case outputGateType = "output_gate_type"
      case mtpNumHiddenLayers = "mtp_num_hidden_layers"
      case numExperts = "num_experts"
      case numExpertsPerTok = "num_experts_per_tok"
      case moeIntermediateSize = "moe_intermediate_size"
      case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
      case normTopkProb = "norm_topk_prob"
      case decoderSparseStep = "decoder_sparse_step"
      case mlpOnlyLayers = "mlp_only_layers"
      case bosTokenId = "bos_token_id"
      case eosTokenId = "eos_token_id"
    }

    public var isFullAttention: [Bool] {
      if let types = layerTypes, types.count == numHiddenLayers {
        return types.map { $0 == "full_attention" }
      }
      let interval = fullAttentionInterval ?? 4
      return (0..<numHiddenLayers).map { ($0 + 1) % interval == 0 }
    }

    /// Which layers route through experts. A checkpoint can be sparse everywhere, sparse on a
    /// stride, or dense in named layers, so the three are resolved together rather than at each
    /// call site.
    public var isSparse: [Bool] {
      guard let experts = numExperts, experts > 0 else {
        return Array(repeating: false, count: numHiddenLayers)
      }
      let step = max(decoderSparseStep ?? 1, 1)
      let dense = Set(mlpOnlyLayers ?? [])
      return (0..<numHiddenLayers).map { !dense.contains($0) && $0 % step == 0 }
    }

    public var ropeDimensions: Int {
      let factor = ropeParameters.partialRotaryFactor ?? partialRotaryFactor ?? 1.0
      return Int((Float(headDim) * factor).rounded())
    }
  }

  public struct VisionConfig: Codable, Sendable {
    public var depth: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHeads: Int
    public var inChannels: Int
    public var patchSize: Int
    public var temporalPatchSize: Int
    public var spatialMergeSize: Int
    public var outHiddenSize: Int
    public var numPositionEmbeddings: Int
    public var hiddenAct: String?
    public var deepstackVisualIndexes: [Int]?

    enum CodingKeys: String, CodingKey {
      case depth
      case hiddenSize = "hidden_size"
      case intermediateSize = "intermediate_size"
      case numHeads = "num_heads"
      case inChannels = "in_channels"
      case patchSize = "patch_size"
      case temporalPatchSize = "temporal_patch_size"
      case spatialMergeSize = "spatial_merge_size"
      case outHiddenSize = "out_hidden_size"
      case numPositionEmbeddings = "num_position_embeddings"
      case hiddenAct = "hidden_act"
      case deepstackVisualIndexes = "deepstack_visual_indexes"
    }
  }

  public static func load(directory: URL) throws -> BonsaiConfig {
    let data = try Data(contentsOf: directory.appending(path: "config.json"))
    if let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      object["schema_version"] == nil
    {
      return try standard(object)
    }
    return try JSONDecoder().decode(BonsaiConfig.self, from: data)
  }

  /// An upstream MLX checkpoint, whose config nests the text and vision towers rather than
  /// flattening them the way a Bonsai pack does.
  static func standard(_ o: [String: Any]) throws -> BonsaiConfig {
    guard let nested = o["text_config"] as? [String: Any] else { return try flat(o) }
    return try assemble(o, text: nested)
  }

  private static func assemble(
    _ o: [String: Any], text nested: [String: Any]
  ) throws -> BonsaiConfig {
    var text = nested
    text["tie_word_embeddings"] =
      (nested["tie_word_embeddings"] as? Bool) ?? (o["tie_word_embeddings"] as? Bool) ?? false

    // MLX writes the rope family under `type`; the pack schema calls it `rope_type`.
    if var rope = nested["rope_parameters"] as? [String: Any] {
      if rope["rope_type"] == nil, let type = rope["type"] as? String {
        rope["rope_type"] = type
      }
      text["rope_parameters"] = rope
    } else {
      text["rope_parameters"] = ["rope_theta": nested["rope_theta"] ?? 1_000_000]
    }

    // Upstream is inconsistent about which of these sits in the text tower and which sits at
    // the top: a dense Qwen3.5 writes `attn_output_gate` beside the dimensions, the MoE writes
    // it at the root. Reading only the nested copy silently loses the query gate, which halves
    // the projection the attention expects and fails on a reshape far from here.
    for key in [
      "attn_output_gate", "full_attention_interval", "partial_rotary_factor",
      "num_experts", "num_experts_per_tok", "moe_intermediate_size",
      "shared_expert_intermediate_size", "norm_topk_prob", "decoder_sparse_step",
      "mlp_only_layers", "mtp_num_hidden_layers",
    ] where text[key] == nil {
      if let value = o[key] { text[key] = value }
    }

    // A dense model leaves the delta-net geometry out entirely.
    for key in [
      "linear_num_value_heads", "linear_num_key_heads", "linear_value_head_dim",
      "linear_key_head_dim", "linear_conv_kernel_dim",
    ] where text[key] == nil {
      text[key] = 0
    }

    let layers = (nested["num_hidden_layers"] as? NSNumber)?.intValue ?? 0
    if text["layer_types"] == nil, nested["full_attention_interval"] == nil {
      text["layer_types"] = Array(repeating: "full_attention", count: layers)
    }

    let quantization =
      o["quantization"] as? [String: Any]
      ?? o["quantization_config"] as? [String: Any]
      ?? ["bits": 16, "group_size": 64, "mode": "affine"]

    let vision = o["vision_config"] as? [String: Any]
    let mtpLayers = (nested["mtp_num_hidden_layers"] as? NSNumber)?.intValue ?? 0

    var pack: [String: Any] = [
      "schema_version": 0,
      "model_type": o["model_type"] as? String ?? "qwen3_5",
      "text_config": text,
      "modules": [],
      "quantization": quantization,
      "components": [
        "text": true, "vision": vision != nil, "mtp": mtpLayers > 0,
      ],
    ]
    if let vision { pack["vision_config"] = vision }
    for key in [
      "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
    ] {
      if let value = o[key] { pack[key] = value }
    }

    return try JSONDecoder().decode(
      BonsaiConfig.self, from: try JSONSerialization.data(withJSONObject: pack))
  }

  /// A plain single-tower checkpoint with its dimensions at the top level.
  static func flat(_ o: [String: Any]) throws -> BonsaiConfig {
    func int(_ key: String) -> Int? { (o[key] as? NSNumber)?.intValue }
    func double(_ key: String) -> Double? { (o[key] as? NSNumber)?.doubleValue }
    guard let hidden = int("hidden_size"), let layers = int("num_hidden_layers"),
      let heads = int("num_attention_heads")
    else {
      throw BonsaiError.unsupportedModel(
        "config.json is missing the core transformer dimensions")
    }
    let modelType = o["model_type"] as? String ?? "qwen3"

    var rope: [String: Any] = ["rope_theta": double("rope_theta") ?? 1_000_000]
    if let scaling = o["rope_scaling"] as? [String: Any] {
      if let type = scaling["rope_type"] as? String { rope["rope_type"] = type }
      if let factor = (scaling["factor"] as? NSNumber)?.doubleValue { rope["factor"] = factor }
      if let original = (scaling["original_max_position_embeddings"] as? NSNumber)?.intValue {
        rope["original_max_position_embeddings"] = original
      }
    }

    var text: [String: Any] = [
      "model_type": modelType,
      "hidden_size": hidden,
      "intermediate_size": int("intermediate_size") ?? hidden * 4,
      "num_hidden_layers": layers,
      "num_attention_heads": heads,
      "num_key_value_heads": int("num_key_value_heads") ?? heads,
      "head_dim": int("head_dim") ?? hidden / heads,
      "rms_norm_eps": double("rms_norm_eps") ?? 1e-6,
      "vocab_size": int("vocab_size") ?? 0,
      "max_position_embeddings": int("max_position_embeddings") ?? 32768,
      "tie_word_embeddings": (o["tie_word_embeddings"] as? Bool) ?? false,
      "layer_types": (o["layer_types"] as? [String])
        ?? Array(repeating: "full_attention", count: layers),
      "linear_num_value_heads": 0, "linear_num_key_heads": 0,
      "linear_value_head_dim": 0, "linear_key_head_dim": 0, "linear_conv_kernel_dim": 0,
      "rope_parameters": rope,
    ]
    if let bos = int("bos_token_id") { text["bos_token_id"] = bos }
    if let eos = int("eos_token_id") { text["eos_token_id"] = eos }

    let quantization =
      o["quantization"] as? [String: Any]
      ?? ["bits": 2, "group_size": 128, "mode": "affine"]
    let pack: [String: Any] = [
      "schema_version": 0,
      "model_type": modelType,
      "text_config": text,
      "modules": [],
      "quantization": quantization,
      "components": ["text": true, "vision": false, "mtp": false],
    ]
    return try JSONDecoder().decode(
      BonsaiConfig.self, from: try JSONSerialization.data(withJSONObject: pack))
  }

  public static let hadamardModelType = "prism_hadamard_qwen35"
  public static let legacyModelTypes: Set<String> = ["qwen3"]
  public static let affineModelTypes: Set<String> = ["qwen3_5", "qwen3_5_moe"]

  /// Which family of packing a checkpoint uses. The rotated Bonsai packs carry a sign vector
  /// and a Hadamard block per module; everything else is plain MLX affine quantization, at one
  /// width or at the mix of widths an imatrix pass chose.
  public enum Profile: Sendable, Equatable {
    case rotated
    case affine
  }

  public var profile: Profile {
    modelType == Self.hadamardModelType ? .rotated : .affine
  }

  /// Widths and group sizes MLX can actually run a quantized matmul at.
  public static let supportedBits: Set<Int> = [2, 3, 4, 5, 6, 8]
  public static let supportedGroupSizes: Set<Int> = [32, 64, 128]

  public func validate() throws {
    switch profile {
    case .rotated: try validateRotated()
    case .affine: try validateAffine()
    }
  }

  private func validateRotated() throws {
    guard quantization.bits == 2, quantization.groupSize == 128,
      quantization.mode == "affine", quantization.overrides.isEmpty
    else {
      throw BonsaiError.unsupportedModel(
        "expected uniform 2-bit affine group-128 quantization, found \(quantization.bits)-bit "
          + "\(quantization.mode) group-\(quantization.groupSize) over "
          + "\(quantization.overrides.count) override(s)")
    }
    for record in modules {
      guard record.dtype == "float16" else {
        throw BonsaiError.unsupportedModel(
          "module \(record.path) has unsupported activation dtype \(record.dtype)")
      }
      guard record.block == 0 || [512, 1024, 2048, 4096].contains(record.block) else {
        throw BonsaiError.unsupportedModel(
          "module \(record.path) has unvalidated Hadamard block \(record.block)")
      }
    }
  }

  private func validateAffine() throws {
    let known =
      [Self.hadamardModelType] + Self.legacyModelTypes.sorted()
      + Self.affineModelTypes.sorted()
    guard known.contains(modelType) else {
      throw BonsaiError.unsupportedModel(
        "unrecognised model_type '\(modelType)'; this runtime reads "
          + known.joined(separator: ", "))
    }
    guard modules.isEmpty else {
      throw BonsaiError.unsupportedModel(
        "\(modelType) packs carry no Hadamard rotation, but the config lists "
          + "\(modules.count) packed-module record(s)")
    }
    for (path, entry) in [("", quantization.default)] + quantization.overrides.map({ ($0, $1) }) {
      let label = path.isEmpty ? "the pack default" : path
      guard entry.mode == "affine" else {
        throw BonsaiError.unsupportedModel(
          "\(label) uses \(entry.mode) quantization; this runtime reads affine")
      }
      guard Self.supportedBits.contains(entry.bits) else {
        throw BonsaiError.unsupportedModel(
          "\(label) is \(entry.bits)-bit; supported widths are "
            + "\(Self.supportedBits.sorted())")
      }
      guard Self.supportedGroupSizes.contains(entry.groupSize) else {
        throw BonsaiError.unsupportedModel(
          "\(label) uses group size \(entry.groupSize); supported sizes are "
            + "\(Self.supportedGroupSizes.sorted())")
      }
    }
  }
}

public enum BonsaiError: Error, CustomStringConvertible {
  case unsupportedModel(String)
  case missingWeight(String)
  case shapeMismatch(String)
  case invalidTransform(String)
  case missingComponent(String)
  case imageProcessing(String)

  public var description: String {
    switch self {
    case .unsupportedModel(let m): "Unsupported model: \(m)"
    case .missingWeight(let m): "Missing weight: \(m)"
    case .shapeMismatch(let m): "Shape mismatch: \(m)"
    case .invalidTransform(let m): "Invalid transform: \(m)"
    case .missingComponent(let m): "Missing component: \(m)"
    case .imageProcessing(let m): "Image processing failed: \(m)"
    }
  }
}
