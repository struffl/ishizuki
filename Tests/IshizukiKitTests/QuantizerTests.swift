// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("Quantizer")
struct QuantizerTests {
  @Test("a narrower width loses more of the weight than a wider one")
  func errorOrders() {
    let w = MLXRandom.normal([256, 512]).asType(.float16)
    let measured = ModuleSurvey.measure(w, path: "m", widths: [2, 3, 4, 5], groupSize: 64)
    #expect(measured.elements == 256 * 512)
    #expect(measured.error(2) > measured.error(3))
    #expect(measured.error(3) > measured.error(4))
    #expect(measured.error(4) > measured.error(5))
    #expect(measured.error(5) > 0)
  }

  @Test("the group's scale and bias are counted, not just the nominal width")
  func bpwIncludesOverhead() {
    // A "4-bit" group-64 module occupies 4.5 bits per weight, which is why oQ4e lands near 4.6.
    #expect(ModuleMeasurement.bpw(bits: 4, groupSize: 64) == 4.5)
    #expect(ModuleMeasurement.bpw(bits: 3, groupSize: 128) == 3.25)
  }

  private func measurement(
    _ path: String, elements: Int, errors: [Int: Double]
  ) -> ModuleMeasurement {
    ModuleMeasurement(path: path, elements: elements, errorAt: errors)
  }

  @Test("the budget is spent where it buys the most, and is never overspent")
  func allocatesByValue() {
    // Three equal modules: one suffers badly at the base width, one moderately, one barely.
    let modules = [
      measurement("bad", elements: 1 << 20, errors: [3: 0.30, 4: 0.10, 5: 0.05]),
      measurement("middling", elements: 1 << 20, errors: [3: 0.12, 4: 0.09, 5: 0.08]),
      measurement("fine", elements: 1 << 20, errors: [3: 0.02, 4: 0.019, 5: 0.018]),
    ]
    // Base 3-bit is 3.5 bpw with group 64, so 3.9 leaves room to lift roughly one module.
    let result = BitAllocator(profile: .quality).allocate(modules)

    #expect(result.achievedBpw <= 3.9)
    #expect(result.bits["bad"]! > result.bits["fine"]!)
    #expect(result.bits["fine"] == 3)
    #expect(result.boosted >= 1)
  }

  @Test("a budget with nothing to spend leaves every module at the base width")
  func tightBudget() {
    let modules = [
      measurement("a", elements: 1 << 20, errors: [3: 0.3, 4: 0.1, 5: 0.05]),
      measurement("b", elements: 1 << 20, errors: [3: 0.3, 4: 0.1, 5: 0.05]),
    ]
    // 3-bit at group 64 is exactly 3.5 bpw: the base costs the whole budget.
    let profile = QuantProfile(
      name: "exact", baseBits: 3, boostBits: [4, 5], targetBpw: 3.5, summary: "")
    let result = BitAllocator(profile: profile).allocate(modules)
    #expect(result.boosted == 0)
    #expect(result.achievedBpw == 3.5)
    #expect(result.histogram == [3: 2])
  }

  @Test("the steeper improvement wins, whatever the modules weigh")
  func valueIsPerByte() {
    // Same size, so only the shape of the improvement separates them.
    let modules = [
      measurement("steep", elements: 1 << 20, errors: [3: 0.30, 4: 0.05]),
      measurement("shallow", elements: 1 << 20, errors: [3: 0.30, 4: 0.28]),
    ]
    // Base 3-bit is 3.5 bpw at group 64, so 4.0 leaves room for exactly one of the two.
    let profile = QuantProfile(
      name: "one", baseBits: 3, boostBits: [4], targetBpw: 4.0, summary: "")
    let result = BitAllocator(profile: profile).allocate(modules)
    #expect(result.bits["steep"] == 4)
    #expect(result.bits["shallow"] == 3)
  }

  @Test("a lift that does not fit is skipped rather than shrinking the model past its target")
  func neverOverspends() {
    // The large module cannot be lifted inside the budget; the small one can.
    let modules = [
      measurement("large", elements: 1 << 22, errors: [3: 0.30, 4: 0.05]),
      measurement("small", elements: 1 << 16, errors: [3: 0.20, 4: 0.10]),
    ]
    let profile = QuantProfile(
      name: "tight", baseBits: 3, boostBits: [4], targetBpw: 3.6, summary: "")
    let result = BitAllocator(profile: profile).allocate(modules)
    #expect(result.bits["large"] == 3)
    #expect(result.bits["small"] == 4)
    #expect(result.achievedBpw <= 3.6)
  }

  @Test("the embedding and the head are floored even when a crowd of small wins spends the budget first")
  func pinsEmbeddingAndHead() {
    // Huge, so one lift for either would blow the whole budget if the auction ever reached them —
    // and it never does, because a swarm of tiny modules with a slightly better per-byte value
    // wins every round first.
    var modules = [
      measurement("language_model.model.embed_tokens.weight", elements: 1 << 24, errors: [3: 0.19, 4: 0.09]),
      measurement("language_model.lm_head.weight", elements: 1 << 24, errors: [3: 0.19, 4: 0.09]),
    ]
    for i in 0..<64 {
      modules.append(
        measurement("layer.\(i).proj.weight", elements: 1 << 10, errors: [3: 0.20, 4: 0.09]))
    }
    let profile = QuantProfile(
      name: "swamped", baseBits: 3, boostBits: [4, 5], targetBpw: 3.51, summary: "")
    let result = BitAllocator(profile: profile).allocate(modules)
    #expect(result.bits["language_model.model.embed_tokens.weight"] == 4)
    #expect(result.bits["language_model.lm_head.weight"] == 4)
  }
}

@Suite("Tensor naming")
struct TensorNamingTests {
  // Taken from the bf16 Qwen3.5 VLM checkpoint and the oQ4e pack built from one.
  @Test("a HuggingFace checkpoint is translated into the layout the runtime reads")
  func huggingFaceLayout() {
    let names = [
      "model.language_model.embed_tokens.weight",
      "model.language_model.layers.0.self_attn.q_proj.weight",
      "model.language_model.norm.weight",
      "model.visual.blocks.0.attn.qkv.weight",
      "lm_head.weight",
      "mtp.fc.weight",
      "mtp.layers.0.self_attn.k_proj.weight",
    ]
    #expect(TensorNaming.isHuggingFaceLayout(names))
    let mapped = TensorNaming.map(names)
    #expect(mapped["model.language_model.embed_tokens.weight"] == "language_model.model.embed_tokens.weight")
    #expect(
      mapped["model.language_model.layers.0.self_attn.q_proj.weight"]
        == "language_model.model.layers.0.self_attn.q_proj.weight")
    #expect(mapped["model.language_model.norm.weight"] == "language_model.model.norm.weight")
    #expect(mapped["model.visual.blocks.0.attn.qkv.weight"] == "vision_tower.blocks.0.attn.qkv.weight")
    #expect(mapped["lm_head.weight"] == "language_model.lm_head.weight")
    #expect(mapped["mtp.fc.weight"] == "language_model.mtp.fc.weight")
    #expect(
      mapped["mtp.layers.0.self_attn.k_proj.weight"]
        == "language_model.mtp.layers.0.self_attn.k_proj.weight")
  }

  @Test("convolution weights are permuted into the layout MLX convolves in")
  func convLayout() {
    // Shapes taken from the bf16 checkpoint and the pack MLX builds from it.
    let conv = MLXArray.zeros([10240, 1, 4], dtype: .float16)
    let moved = TensorNaming.relayout(
      "model.language_model.layers.0.linear_attn.conv1d.weight", conv)
    #expect(moved.shape == [10240, 4, 1])

    let patch = MLXArray.zeros([1152, 3, 2, 16, 16], dtype: .float16)
    let movedPatch = TensorNaming.relayout("model.visual.patch_embed.proj.weight", patch)
    #expect(movedPatch.shape == [1152, 2, 16, 16, 3])

    // Anything else keeps its shape.
    let linear = MLXArray.zeros([64, 128], dtype: .float16)
    #expect(TensorNaming.relayout("a.b.q_proj.weight", linear).shape == [64, 128])
  }

  @Test("a checkpoint already in the runtime's layout is left alone")
  func alreadyCanonical() {
    let names = [
      "language_model.model.embed_tokens.weight",
      "language_model.lm_head.weight",
      "vision_tower.patch_embed.proj.weight",
    ]
    #expect(!TensorNaming.isHuggingFaceLayout(names))
    let mapped = TensorNaming.map(names)
    for name in names { #expect(mapped[name] == name) }
  }

  /// The zero-centred convention is the quietest way to get a pack wrong: every weight loads
  /// and every shape checks out whether or not the implied one is folded in, and the only
  /// symptom of missing it is a model that emits confident nonsense.
  @Test("every language-model RMSNorm is zero-centred except the delta-net's own")
  func zeroCentredNormClassification() {
    // Classified against the oQ4e pack and the checkpoint it was built from: unanimous across
    // all 214 of that model's norm tensors.
    let folded = [
      "model.language_model.layers.0.input_layernorm.weight",
      "model.language_model.layers.0.post_attention_layernorm.weight",
      "model.language_model.layers.3.self_attn.q_norm.weight",
      "model.language_model.layers.3.self_attn.k_norm.weight",
      "model.language_model.norm.weight",
      "mtp.norm.weight",
      "mtp.layers.0.input_layernorm.weight",
      "mtp.pre_fc_norm_embedding.weight",
      "mtp.pre_fc_norm_hidden.weight",
    ]
    for name in folded {
      #expect(TensorNaming.isZeroCentredNorm(name), "\(name) should be folded")
    }

    let untouched = [
      "model.language_model.layers.0.linear_attn.norm.weight",
      "model.visual.merger.norm.weight",
      "vision_tower.blocks.0.norm1.weight",
      "vision_tower.merger.norm.bias",
      "model.language_model.layers.0.mlp.up_proj.weight",
      "model.language_model.layers.0.linear_attn.A_log",
    ]
    for name in untouched {
      #expect(!TensorNaming.isZeroCentredNorm(name), "\(name) should be left alone")
    }
  }

  @Test("folding adds the implied one, and only for architectures that store it that way")
  func zeroCentredNormFold() {
    let weight = MLXArray([-0.25, 0.0, 0.5] as [Float]).asType(.bfloat16)
    let name = "model.language_model.layers.0.input_layernorm.weight"

    let folded = TensorNaming.relayout(name, weight, zeroCentredNorms: true)
    #expect(folded.asType(.float32).asArray(Float.self) == [0.75, 1.0, 1.5])

    let asIs = TensorNaming.relayout(name, weight, zeroCentredNorms: false)
    #expect(asIs.asType(.float32).asArray(Float.self) == [-0.25, 0.0, 0.5])

    let gated = TensorNaming.relayout(
      "model.language_model.layers.0.linear_attn.norm.weight", weight, zeroCentredNorms: true)
    #expect(gated.asType(.float32).asArray(Float.self) == [-0.25, 0.0, 0.5])

    #expect(TensorNaming.usesZeroCentredNorms(["model_type": "qwen3_5"]))
    #expect(!TensorNaming.usesZeroCentredNorms(["model_type": "qwen3"]))
    #expect(!TensorNaming.usesZeroCentredNorms([:]))
  }
}
