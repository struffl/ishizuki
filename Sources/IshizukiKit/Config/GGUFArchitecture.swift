// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// GGUF metadata and tensor names, read into the shapes the runtime already builds from.

import Foundation

/// A GGUF file's hyperparameters, in the runtime's own vocabulary.
///
/// llama.cpp names the same model differently at every level: the architecture key is `qwen35`
/// where the checkpoint says `qwen3_5`, the delta-net geometry is spelled as SSM state sizes
/// rather than head counts, and the attention output gate is implied by a doubled `attn_q`
/// rather than declared. This resolves all three against the tensor table, so a file that does
/// not actually match its own metadata is refused here instead of loading crooked.
public struct GGUFArchitecture: Sendable {
  public let textConfig: BonsaiConfig.TextConfig
  public let chatTemplate: String?
  public let hasMTP: Bool

  public static let supportedArchitectures: Set<String> = ["qwen35"]

  public init(file: GGUFFile) throws {
    let arch = file.architecture
    guard Self.supportedArchitectures.contains(arch) else {
      throw BonsaiError.unsupportedModel(
        "GGUF architecture '\(arch)'; this runtime reads "
          + Self.supportedArchitectures.sorted().joined(separator: ", "))
    }

    func int(_ suffix: String) throws -> Int {
      guard let value = file.architectureValue(suffix)?.intValue else {
        throw BonsaiError.unsupportedModel("\(arch).\(suffix) is missing from the GGUF metadata")
      }
      return value
    }
    func optionalInt(_ suffix: String) -> Int? { file.architectureValue(suffix)?.intValue }

    let layers = try int("block_count")
    let hidden = try int("embedding_length")
    let heads = try int("attention.head_count")
    let headDim = optionalInt("attention.key_length") ?? hidden / max(heads, 1)

    let valueHeads = try int("ssm.time_step_rank")
    let keyHeads = try int("ssm.group_count")
    let keyHeadDim = try int("ssm.state_size")
    let innerSize = try int("ssm.inner_size")
    guard valueHeads > 0, innerSize % valueHeads == 0 else {
      throw BonsaiError.unsupportedModel(
        "\(arch).ssm.inner_size (\(innerSize)) is not divisible by its \(valueHeads) value heads")
    }
    let valueHeadDim = innerSize / valueHeads

    let ropeDimensions = optionalInt("rope.dimension_count") ?? headDim
    let vocab =
      file["tokenizer.ggml.tokens"]?.arrayValue?.count
      ?? optionalInt("vocab_size")
      ?? 0

    var rope: [String: Any] = [
      "rope_theta": file.architectureValue("rope.freq_base")?.floatValue ?? 10_000_000
    ]
    if let sections = file.architectureValue("rope.dimension_sections")?.intArray,
      sections.contains(where: { $0 > 0 })
    {
      rope["rope_type"] = "mrope"
      rope["mrope_section"] = sections
    }
    if ropeDimensions != headDim {
      rope["partial_rotary_factor"] = Double(ropeDimensions) / Double(headDim)
    }

    let gateWidth = file[tensor: "blk.\(Self.firstFullAttentionLayer(file, layers)).attn_q.weight"]?
      .shape.first
    let outputGate = gateWidth.map { $0 == 2 * heads * headDim } ?? false

    var text: [String: Any] = [
      "model_type": "qwen3_5",
      "hidden_size": hidden,
      "intermediate_size": try int("feed_forward_length"),
      "num_hidden_layers": layers,
      "num_attention_heads": heads,
      "num_key_value_heads": optionalInt("attention.head_count_kv") ?? heads,
      "head_dim": headDim,
      "rms_norm_eps": file.architectureValue("attention.layer_norm_rms_epsilon")?.floatValue
        ?? 1e-6,
      "vocab_size": vocab,
      "max_position_embeddings": optionalInt("context_length") ?? 32768,
      "tie_word_embeddings": file[tensor: "output.weight"] == nil,
      "attn_output_gate": outputGate,
      "full_attention_interval": optionalInt("full_attention_interval") ?? 4,
      "linear_num_value_heads": valueHeads,
      "linear_num_key_heads": keyHeads,
      "linear_value_head_dim": valueHeadDim,
      "linear_key_head_dim": keyHeadDim,
      "linear_conv_kernel_dim": try int("ssm.conv_kernel"),
      "rope_parameters": rope,
    ]
    if let bos = file["tokenizer.ggml.bos_token_id"]?.intValue { text["bos_token_id"] = bos }
    if let eos = file["tokenizer.ggml.eos_token_id"]?.intValue { text["eos_token_id"] = eos }

    self.textConfig = try JSONDecoder().decode(
      BonsaiConfig.TextConfig.self,
      from: try JSONSerialization.data(withJSONObject: text))
    self.chatTemplate = file["tokenizer.chat_template"]?.stringValue
    self.hasMTP = file.tensors.contains { $0.name.hasPrefix("mtp") || $0.name.contains(".mtp") }

    try validate(against: file)
  }

  private static func firstFullAttentionLayer(_ file: GGUFFile, _ layers: Int) -> Int {
    for layer in 0..<layers where file[tensor: "blk.\(layer).attn_q.weight"] != nil {
      return layer
    }
    return 0
  }

  /// The metadata claims a geometry; the tensor table is the geometry. A mismatch here is the
  /// quiet failure mode of a hand-assembled GGUF — every tensor loads and the model talks
  /// nonsense — so the fused projections are checked against the widths they imply.
  private func validate(against file: GGUFFile) throws {
    let text = textConfig
    let keyDim = text.linearNumKeyHeads * text.linearKeyHeadDim
    let valueDim = text.linearNumValueHeads * text.linearValueHeadDim
    let convDim = 2 * keyDim + valueDim

    for layer in 0..<text.numHiddenLayers {
      if let qkv = file[tensor: "blk.\(layer).attn_qkv.weight"] {
        guard qkv.shape.first == convDim else {
          throw BonsaiError.shapeMismatch(
            "blk.\(layer).attn_qkv is \(qkv.shape.first ?? 0) wide; the SSM metadata implies "
              + "\(convDim)")
        }
        guard let gate = file[tensor: "blk.\(layer).attn_gate.weight"],
          gate.shape.first == valueDim
        else {
          throw BonsaiError.shapeMismatch(
            "blk.\(layer).attn_gate does not match the \(valueDim)-wide value path")
        }
      }
    }

    guard let embedding = file[tensor: "token_embd.weight"],
      embedding.shape.last == text.hiddenSize
    else {
      throw BonsaiError.shapeMismatch(
        "token_embd is not \(text.hiddenSize) wide")
    }
    guard embedding.shape.first == text.vocabSize else {
      throw BonsaiError.shapeMismatch(
        "token_embd holds \(embedding.shape.first ?? 0) rows against a vocabulary of "
          + "\(text.vocabSize)")
    }
  }
}

/// Translates llama.cpp's flat tensor names into the paths the runtime's modules read.
extension GGUFArchitecture {
  /// The config the rest of the runtime expects, assembled from what the file declares.
  ///
  /// `quantization` is the one field that cannot be filled honestly: a pack quantizes every
  /// module the same way and writes the scheme down, while a GGUF carries a block type per
  /// tensor and no scheme at all. It is left at a width no pack could have so that anything
  /// reading it for a GGUF fails loudly instead of believing a plausible number — which is also
  /// why `validate()`, a check on a pack's scheme, is not run against this.
  public func config(vision: BonsaiConfig.VisionConfig? = nil) -> BonsaiConfig {
    BonsaiConfig(
      schemaVersion: 0,
      modelType: textConfig.modelType,
      baseModelType: nil,
      textConfig: textConfig,
      visionConfig: vision,
      modules: [],
      quantization: BonsaiConfig.QuantizationConfig(bits: 0, groupSize: 0, mode: "ggml"),
      components: BonsaiConfig.Components(
        text: true, vision: vision != nil, mtp: hasMTP),
      tensorNamespace: nil,
      gdnActivationLayout: nil,
      requiresRuntime: nil)
  }
}

public enum GGUFTensorNaming {
  public static let prefix = "language_model.model."

  /// Nil for a tensor this runtime has no module for, so an unknown extra in a file is
  /// reported rather than silently dropped.
  public static func canonical(_ name: String) -> String? {
    switch name {
    case "token_embd.weight": return prefix + "embed_tokens.weight"
    case "output_norm.weight": return prefix + "norm.weight"
    case "output.weight": return "language_model.lm_head.weight"
    default: break
    }

    let parts = name.split(separator: ".", maxSplits: 2).map(String.init)
    guard parts.count == 3, parts[0] == "blk", let layer = Int(parts[1]) else { return nil }
    guard let suffix = blockSuffix(parts[2]) else { return nil }
    return "\(prefix)layers.\(layer).\(suffix)"
  }

  private static func blockSuffix(_ tail: String) -> String? {
    switch tail {
    case "attn_norm.weight": "input_layernorm.weight"
    case "post_attention_norm.weight": "post_attention_layernorm.weight"

    case "ffn_gate.weight": "mlp.gate_proj.weight"
    case "ffn_up.weight": "mlp.up_proj.weight"
    case "ffn_down.weight": "mlp.down_proj.weight"

    case "attn_q.weight": "self_attn.q_proj.weight"
    case "attn_k.weight": "self_attn.k_proj.weight"
    case "attn_v.weight": "self_attn.v_proj.weight"
    case "attn_output.weight": "self_attn.o_proj.weight"
    case "attn_q_norm.weight": "self_attn.q_norm.weight"
    case "attn_k_norm.weight": "self_attn.k_norm.weight"

    case "attn_qkv.weight": "linear_attn.in_proj_qkv.weight"
    case "attn_gate.weight": "linear_attn.in_proj_z.weight"
    case "ssm_alpha.weight": "linear_attn.in_proj_a.weight"
    case "ssm_beta.weight": "linear_attn.in_proj_b.weight"
    case "ssm_out.weight": "linear_attn.out_proj.weight"
    case "ssm_conv1d.weight": "linear_attn.conv1d.weight"
    case "ssm_norm.weight": "linear_attn.norm.weight"
    case "ssm_a": "linear_attn.A_log"
    case "ssm_dt.bias": "linear_attn.dt_bias"

    default: nil
    }
  }
}
