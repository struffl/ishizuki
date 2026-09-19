// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import IshizukiKit

@Suite("GGUF")
struct GGUFTests {
  private typealias Builder = GGUFFixture.Builder

  private func temporaryURL() -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "gguf-\(UUID().uuidString).gguf")
  }

  @Test("block geometry matches llama.cpp")
  func blockGeometry() {
    let expected: [(GGMLType, Int, Int)] = [
      (.f32, 1, 4), (.f16, 1, 2), (.bf16, 1, 2),
      (.q2_K, 256, 84), (.q4_K, 256, 144), (.q6_K, 256, 210),
      (.iq1_s, 256, 50), (.iq1_m, 256, 56),
      (.iq2_xxs, 256, 66), (.iq2_xs, 256, 74), (.iq2_s, 256, 82),
      (.iq3_xxs, 256, 98), (.iq3_s, 256, 110),
      (.iq4_nl, 32, 18), (.iq4_xs, 256, 136),
    ]
    for (type, blockSize, typeSize) in expected {
      #expect(type.blockSize == blockSize, "\(type.name) block size")
      #expect(type.typeSize == typeSize, "\(type.name) type size")
    }

    // The smallest published GSQ-RCO build: 5120 x 248320 at IQ1_M is 278,118,400 bytes.
    #expect(GGMLType.iq1_m.byteCount(elements: 5120 * 248_320) == 278_118_400)
  }

  @Test("reads header, metadata and the tensor table")
  func containerRoundTrip() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url) }

    var builder = Builder()
    builder.metadata = [
      ("general.architecture", 8, Builder.string("qwen35")),
      ("qwen35.block_count", 4, Builder.u32(2)),
      ("qwen35.rope.freq_base", 6, Builder.f32(10_000_000)),
      ("qwen35.rope.dimension_sections", 9, Builder.intArray([11, 11, 10, 0])),
      ("tokenizer.ggml.tokens", 9, Builder.stringArray(["a", "b", "c"])),
    ]
    builder.tensors = [
      ("token_embd.weight", [3, 256], .iq1_m),
      ("output_norm.weight", [256], .f32),
      ("blk.0.attn_qkv.weight", [512, 256], .iq2_xs),
    ]
    try builder.write(to: url)

    let file = try GGUFFile(url: url)
    #expect(file.version == 3)
    #expect(file.architecture == "qwen35")
    #expect(file.architectureValue("block_count")?.intValue == 2)
    #expect(file.architectureValue("rope.freq_base")?.floatValue == 10_000_000)
    #expect(file.architectureValue("rope.dimension_sections")?.intArray == [11, 11, 10, 0])
    #expect(file["tokenizer.ggml.tokens"]?.stringArray == ["a", "b", "c"])

    #expect(file.tensors.count == 3)
    let qkv = try #require(file[tensor: "blk.0.attn_qkv.weight"])
    #expect(qkv.shape == [512, 256])
    #expect(qkv.type == .iq2_xs)
    #expect(qkv.byteCount == 512 * 256 / 256 * 74)

    let embedding = try #require(file[tensor: "token_embd.weight"])
    #expect(try file.data(for: embedding).count == embedding.byteCount)
    #expect(file.typeHistogram.contains { $0.type == .f32 && $0.count == 1 })
  }

  @Test("refuses a file whose element count does not fill whole blocks")
  func refusesRaggedBlocks() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url) }

    var builder = Builder()
    builder.metadata = [("general.architecture", 8, Builder.string("qwen35"))]
    builder.tensors = [("blk.0.ffn_up.weight", [4, 100], .iq2_xs)]
    try builder.write(to: url)

    #expect(throws: BonsaiError.self) { _ = try GGUFFile(url: url) }
  }

  @Test("maps llama.cpp tensor names onto the runtime's module paths")
  func tensorNames() {
    let prefix = GGUFTensorNaming.prefix
    #expect(GGUFTensorNaming.canonical("token_embd.weight") == prefix + "embed_tokens.weight")
    #expect(GGUFTensorNaming.canonical("output_norm.weight") == prefix + "norm.weight")
    #expect(GGUFTensorNaming.canonical("output.weight") == "language_model.lm_head.weight")

    #expect(
      GGUFTensorNaming.canonical("blk.7.attn_norm.weight")
        == prefix + "layers.7.input_layernorm.weight")
    #expect(
      GGUFTensorNaming.canonical("blk.3.attn_q.weight")
        == prefix + "layers.3.self_attn.q_proj.weight")
    #expect(
      GGUFTensorNaming.canonical("blk.3.attn_output.weight")
        == prefix + "layers.3.self_attn.o_proj.weight")
    #expect(
      GGUFTensorNaming.canonical("blk.0.attn_qkv.weight")
        == prefix + "layers.0.linear_attn.in_proj_qkv.weight")
    #expect(
      GGUFTensorNaming.canonical("blk.0.attn_gate.weight")
        == prefix + "layers.0.linear_attn.in_proj_z.weight")
    #expect(
      GGUFTensorNaming.canonical("blk.0.ssm_a") == prefix + "layers.0.linear_attn.A_log")
    #expect(
      GGUFTensorNaming.canonical("blk.0.ssm_dt.bias")
        == prefix + "layers.0.linear_attn.dt_bias")
    #expect(
      GGUFTensorNaming.canonical("blk.62.ffn_down.weight")
        == prefix + "layers.62.mlp.down_proj.weight")

    #expect(GGUFTensorNaming.canonical("rope_freqs.weight") == nil)
    #expect(GGUFTensorNaming.canonical("blk.x.attn_norm.weight") == nil)
  }

  @Test("reads the published Qwen3.8-27B geometry out of its metadata")
  func architecture() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url) }

    var builder = Builder()
    builder.metadata = [
      ("general.architecture", 8, Builder.string("qwen35")),
      ("qwen35.block_count", 4, Builder.u32(4)),
      ("qwen35.context_length", 4, Builder.u32(262_144)),
      ("qwen35.embedding_length", 4, Builder.u32(5120)),
      ("qwen35.feed_forward_length", 4, Builder.u32(17408)),
      ("qwen35.attention.head_count", 4, Builder.u32(24)),
      ("qwen35.attention.head_count_kv", 4, Builder.u32(4)),
      ("qwen35.attention.key_length", 4, Builder.u32(256)),
      ("qwen35.attention.layer_norm_rms_epsilon", 6, Builder.f32(1e-6)),
      ("qwen35.rope.freq_base", 6, Builder.f32(10_000_000)),
      ("qwen35.rope.dimension_count", 4, Builder.u32(64)),
      ("qwen35.rope.dimension_sections", 9, Builder.intArray([11, 11, 10, 0])),
      ("qwen35.ssm.conv_kernel", 4, Builder.u32(4)),
      ("qwen35.ssm.state_size", 4, Builder.u32(128)),
      ("qwen35.ssm.group_count", 4, Builder.u32(16)),
      ("qwen35.ssm.time_step_rank", 4, Builder.u32(48)),
      ("qwen35.ssm.inner_size", 4, Builder.u32(6144)),
      ("qwen35.full_attention_interval", 4, Builder.u32(4)),
      ("tokenizer.ggml.tokens", 9, Builder.stringArray((0..<256).map { "t\($0)" })),
      ("tokenizer.ggml.eos_token_id", 4, Builder.u32(248_046)),
      ("tokenizer.chat_template", 8, Builder.string("{{ messages }}")),
    ]
    builder.tensors = [
      ("token_embd.weight", [256, 5120], .iq1_m),
      ("output.weight", [256, 5120], .iq4_xs),
      ("output_norm.weight", [5120], .f32),
    ]
    for layer in 0..<4 {
      if (layer + 1) % 4 == 0 {
        builder.tensors += [
          ("blk.\(layer).attn_q.weight", [12288, 5120], .iq2_xs),
          ("blk.\(layer).attn_k.weight", [1024, 5120], .iq2_xs),
          ("blk.\(layer).attn_v.weight", [1024, 5120], .iq2_xs),
          ("blk.\(layer).attn_output.weight", [5120, 6144], .iq2_xs),
        ]
      } else {
        builder.tensors += [
          ("blk.\(layer).attn_qkv.weight", [10240, 5120], .iq3_s),
          ("blk.\(layer).attn_gate.weight", [6144, 5120], .iq3_s),
          ("blk.\(layer).ssm_a", [48], .f32),
        ]
      }
    }
    try builder.write(to: url)

    let architecture = try GGUFArchitecture(file: GGUFFile(url: url))
    let text = architecture.textConfig
    #expect(text.modelType == "qwen3_5")
    #expect(text.hiddenSize == 5120)
    #expect(text.intermediateSize == 17408)
    #expect(text.numAttentionHeads == 24)
    #expect(text.numKeyValueHeads == 4)
    #expect(text.headDim == 256)
    #expect(text.vocabSize == 256)
    #expect(text.maxPositionEmbeddings == 262_144)
    #expect(text.tieWordEmbeddings == false)
    #expect(text.attnOutputGate == true)

    #expect(text.linearNumValueHeads == 48)
    #expect(text.linearNumKeyHeads == 16)
    #expect(text.linearKeyHeadDim == 128)
    #expect(text.linearValueHeadDim == 128)
    #expect(text.linearConvKernelDim == 4)

    #expect(text.ropeParameters.ropeTheta == 10_000_000)
    #expect(text.ropeParameters.mropeSection == [11, 11, 10, 0])
    #expect(text.ropeDimensions == 64)
    #expect(text.isFullAttention == [false, false, false, true])
    #expect(text.eosTokenId == 248_046)
    #expect(architecture.chatTemplate == "{{ messages }}")
    #expect(architecture.hasMTP == false)
  }

  @Test("refuses a fused projection that contradicts the SSM metadata")
  func refusesGeometryMismatch() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url) }

    var builder = Builder()
    builder.metadata = [
      ("general.architecture", 8, Builder.string("qwen35")),
      ("qwen35.block_count", 4, Builder.u32(1)),
      ("qwen35.embedding_length", 4, Builder.u32(5120)),
      ("qwen35.feed_forward_length", 4, Builder.u32(17408)),
      ("qwen35.attention.head_count", 4, Builder.u32(24)),
      ("qwen35.attention.key_length", 4, Builder.u32(256)),
      ("qwen35.ssm.conv_kernel", 4, Builder.u32(4)),
      ("qwen35.ssm.state_size", 4, Builder.u32(128)),
      ("qwen35.ssm.group_count", 4, Builder.u32(16)),
      ("qwen35.ssm.time_step_rank", 4, Builder.u32(48)),
      ("qwen35.ssm.inner_size", 4, Builder.u32(6144)),
      ("tokenizer.ggml.tokens", 9, Builder.stringArray((0..<256).map { "t\($0)" })),
    ]
    builder.tensors = [
      ("token_embd.weight", [256, 5120], .iq1_m),
      ("blk.0.attn_qkv.weight", [8192, 5120], .iq3_s),
      ("blk.0.attn_gate.weight", [6144, 5120], .iq3_s),
    ]
    try builder.write(to: url)

    #expect(throws: BonsaiError.self) {
      _ = try GGUFArchitecture(file: GGUFFile(url: url))
    }
  }
  @Test("builds the tokenizer out of GGUF metadata")
  func tokenizerFromMetadata() throws {
    let url = temporaryURL()
    defer { try? FileManager.default.removeItem(at: url) }

    let tokens = ["h", "i", "\u{0120}", "t", "e", "r", "hi", "<|im_end|>"]
    var builder = Builder()
    builder.metadata = [
      ("general.architecture", 8, Builder.string("qwen35")),
      ("tokenizer.ggml.tokens", 9, Builder.stringArray(tokens)),
      ("tokenizer.ggml.token_type", 9, Builder.intArray([1, 1, 1, 1, 1, 1, 1, 3])),
      ("tokenizer.ggml.merges", 9, Builder.stringArray(["h i"])),
      ("tokenizer.ggml.eos_token_id", 4, Builder.u32(7)),
    ]
    try builder.write(to: url)

    let tokenizer = try BonsaiTokenizer(gguf: GGUFFile(url: url))
    #expect(tokenizer.eosTokenIds.contains(7))

    let ids = tokenizer.encode("hi there<|im_end|>")
    #expect(ids == [6, 2, 3, 0, 4, 5, 4, 7])
    #expect(tokenizer.decode(ids) == "hi there<|im_end|>")
    #expect(tokenizer.decode(ids, skipSpecialTokens: true) == "hi there")
  }
}
