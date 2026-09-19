// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Translates a checkpoint's tensor names into the layout this runtime reads.
///
/// A HuggingFace multimodal checkpoint nests both towers under `model.` and calls the vision
/// side `visual`, while the MLX packs everything downstream expects put the language model's
/// own `model.` inside `language_model.` and call the vision side `vision_tower`. The heads
/// sit at the top level in one and under the language model in the other.
///
/// Getting this wrong does not fail loudly: the pack writes, and only refuses to load later,
/// so the mapping is explicit and tested rather than discovered by a missing weight.
public enum TensorNaming {
  public static func isHuggingFaceLayout(_ names: [String]) -> Bool {
    names.contains { $0.hasPrefix("model.language_model.") }
  }

  public static func canonical(_ name: String) -> String {
    if name.hasPrefix("model.language_model.") {
      return "language_model.model." + name.dropFirst("model.language_model.".count)
    }
    if name.hasPrefix("model.visual.") {
      return "vision_tower." + name.dropFirst("model.visual.".count)
    }
    if name.hasPrefix("visual.") {
      return "vision_tower." + name.dropFirst("visual.".count)
    }
    // Heads sit beside the towers upstream and inside the language model here.
    if name.hasPrefix("lm_head.") || name.hasPrefix("mtp.") {
      return "language_model." + name
    }
    if name.hasPrefix("model.") , !name.hasPrefix("model.vision_tower.") {
      return "language_model." + name
    }
    return name
  }

  /// Architectures whose RMSNorm weights are stored zero-centred: the checkpoint holds `w` and
  /// the norm is defined as scaling by `1 + w`, not by `w`.
  public static let zeroCentredNormModelTypes: Set<String> = ["qwen3_5", "qwen3_5_moe"]

  public static func usesZeroCentredNorms(_ config: [String: Any]) -> Bool {
    guard let modelType = config["model_type"] as? String else { return false }
    return zeroCentredNormModelTypes.contains(modelType)
  }

  /// The norm weights the `1 + w` convention covers.
  ///
  /// The delta-net's own internal RMSNorm is the one exception in the language model: it is
  /// stored already centred on one, and adding to it would scale the gated path twice. The
  /// vision tower is excluded wholesale — its norms are LayerNorms with their own bias, and a
  /// `merger.norm` would otherwise match on name alone.
  public static func isZeroCentredNorm(_ name: String) -> Bool {
    guard name.hasSuffix(".weight") else { return false }
    guard !name.hasPrefix("visual."), !name.hasPrefix("model.visual."),
      !name.hasPrefix("vision_tower.")
    else { return false }
    guard !name.hasSuffix(".linear_attn.norm.weight") else { return false }
    let module = name.dropLast(".weight".count).split(separator: ".").last.map(String.init)
    return module.map(centredNormModules.contains) ?? false
  }

  private static let centredNormModules: Set<String> = [
    "input_layernorm", "post_attention_layernorm", "q_norm", "k_norm", "norm",
    "pre_fc_norm_embedding", "pre_fc_norm_hidden",
  ]

  /// Convolution weights are laid out channels-second by PyTorch and channels-last by MLX, so
  /// they have to be permuted as well as renamed, and a zero-centred norm has its implied one
  /// folded in so the runtime can scale by the weight it reads.
  ///
  /// The delta-net's depthwise conv arrives as [out, in/groups, kernel] and is wanted as
  /// [out, kernel, in/groups]; the vision patch embedding arrives as [out, C, T, H, W] and is
  /// wanted as [out, T, H, W, C].
  ///
  /// Getting the norm fold wrong is the quiet failure this whole type exists to prevent: every
  /// weight loads, every shape checks out, and the model emits fluent-looking nonsense.
  public static func relayout(
    _ name: String, _ array: MLXArray, zeroCentredNorms: Bool = false
  ) -> MLXArray {
    if name.hasSuffix("conv1d.weight"), array.ndim == 3 {
      return array.transposed(0, 2, 1)
    }
    if name.hasSuffix("patch_embed.proj.weight"), array.ndim == 5 {
      return array.transposed(0, 2, 3, 4, 1)
    }
    if zeroCentredNorms, isZeroCentredNorm(name) {
      // In float32 rather than the source's bfloat16: adding one to a small centred value there
      // would round most of it away before the caller narrows to the pack's float16.
      return array.asType(.float32) + 1
    }
    return array
  }

  /// The mapping applied to a whole checkpoint, or the identity when it is already canonical.
  public static func map(_ names: [String]) -> [String: String] {
    guard isHuggingFaceLayout(names) else {
      return Dictionary(uniqueKeysWithValues: names.map { ($0, $0) })
    }
    return Dictionary(uniqueKeysWithValues: names.map { ($0, canonical($0)) })
  }
}
