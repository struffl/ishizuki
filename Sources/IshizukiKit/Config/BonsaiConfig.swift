// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

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

  public struct QuantizationConfig: Codable, Sendable {
    public var bits: Int
    public var groupSize: Int
    public var mode: String

    enum CodingKeys: String, CodingKey {
      case bits
      case groupSize = "group_size"
      case mode
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

    enum CodingKeys: String, CodingKey {
      case ropeType = "rope_type"
      case ropeTheta = "rope_theta"
      case partialRotaryFactor = "partial_rotary_factor"
      case mropeSection = "mrope_section"
      case mropeInterleaved = "mrope_interleaved"
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
    return try JSONDecoder().decode(BonsaiConfig.self, from: data)
  }

  public func validate() throws {
    guard modelType == "prism_hadamard_qwen35" else {
      throw BonsaiError.unsupportedModel(
        "expected model_type 'prism_hadamard_qwen35', found '\(modelType)'")
    }
    guard quantization.bits == 2, quantization.groupSize == 128,
      quantization.mode == "affine"
    else {
      throw BonsaiError.unsupportedModel(
        "expected 2-bit affine group-128 quantization, found \(quantization.bits)-bit "
          + "\(quantization.mode) group-\(quantization.groupSize)")
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
