// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

/// Loading a GGUF and running a module straight off its blocks. The split matters as much as
/// the arithmetic: a quantized tensor that quietly lands in the dense table would still give
/// right answers, at five times the memory the file was chosen for.
@Suite("GGUF weights")
struct GGUFWeightsTests {
  private struct Writer {
    var metadata: [(String, UInt32, Data)] = []
    var tensors: [(name: String, dims: [Int], type: GGMLType, payload: Data)] = []

    static func string(_ value: String) -> Data {
      var out = Data()
      let bytes = Array(value.utf8)
      withUnsafeBytes(of: UInt64(bytes.count)) { out.append(contentsOf: $0) }
      out.append(contentsOf: bytes)
      return out
    }

    static func u32(_ value: UInt32) -> Data {
      var out = Data()
      withUnsafeBytes(of: value) { out.append(contentsOf: $0) }
      return out
    }

    func write(to url: URL) throws {
      var data = Data()
      func put<T>(_ value: T) { withUnsafeBytes(of: value) { data.append(contentsOf: $0) } }

      put(GGUFFile.magic)
      put(UInt32(3))
      put(UInt64(tensors.count))
      put(UInt64(metadata.count))
      for (key, type, payload) in metadata {
        data.append(Self.string(key))
        put(type)
        data.append(payload)
      }

      var cursor = 0
      for tensor in tensors {
        data.append(Self.string(tensor.name))
        put(UInt32(tensor.dims.count))
        for dim in tensor.dims.reversed() { put(UInt64(dim)) }
        put(tensor.type.rawValue)
        cursor = (cursor + 31) / 32 * 32
        put(UInt64(cursor))
        cursor += tensor.payload.count
      }
      data.append(Data(repeating: 0, count: (32 - data.count % 32) % 32))

      var section = Data()
      for tensor in tensors {
        section.append(Data(repeating: 0, count: (32 - section.count % 32) % 32))
        section.append(tensor.payload)
      }
      data.append(section)
      try data.write(to: url)
    }
  }

  private func payload(_ type: GGMLType, elements: Int, seed: UInt64) -> Data {
    var state = seed
    let count = elements / type.blockSize * type.typeSize
    var bytes = [UInt8]()
    bytes.reserveCapacity(count)
    for _ in 0..<count {
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      bytes.append(UInt8((state >> 33) & 0xbf))
    }
    return Data(bytes)
  }

  private func floats(_ values: [Float]) -> Data {
    var out = Data()
    for value in values { withUnsafeBytes(of: value) { out.append(contentsOf: $0) } }
    return out
  }

  @Test("blocks stay blocked, small tensors come through dense, and the conv is transposed")
  func splitsAndTransposes() throws {
    let url = URL(filePath: NSTemporaryDirectory())
      .appending(path: "gguf-weights-\(UUID().uuidString).gguf")
    defer { try? FileManager.default.removeItem(at: url) }

    let hidden = 512
    let vocab = 8

    var writer = Writer()
    writer.metadata = [("general.architecture", 8, Writer.string("qwen35"))]
    writer.tensors = [
      (
        "token_embd.weight", [vocab, hidden], .iq2_xs,
        payload(.iq2_xs, elements: vocab * hidden, seed: 11)
      ),
      (
        "blk.0.ffn_down.weight", [hidden, hidden], .iq3_s,
        payload(.iq3_s, elements: hidden * hidden, seed: 22)
      ),
      ("output_norm.weight", [hidden], .f32, floats((0..<hidden).map { Float($0) * 0.01 })),
      (
        "blk.0.ssm_conv1d.weight", [hidden, 4], .f32,
        floats((0..<(4 * hidden)).map { Float($0 % 7) })
      ),
    ]
    try writer.write(to: url)

    let file = try GGUFFile(url: url)
    let store = try GGUFWeights.load(file: file)

    let prefix = GGUFTensorNaming.prefix
    #expect(store.ggml(prefix + "embed_tokens.weight")?.type == .iq2_xs)
    #expect(store.ggml(prefix + "layers.0.mlp.down_proj.weight")?.type == .iq3_s)
    #expect(store.ggml(prefix + "norm.weight") == nil)
    #expect(store.has(prefix + "norm.weight"))

    let norm = try store(prefix + "norm.weight")
    #expect(norm.shape == [hidden])
    #expect(abs(norm[3].item(Float.self) - 0.03) < 1e-6)

    // [out, kernel] on disk, [out, kernel, in/groups] in the forward pass.
    let conv = try store(prefix + "layers.0.linear_attn.conv1d.weight")
    #expect(conv.shape == [hidden, 4, 1])
  }

  /// The two conversion-time folds, pinned in opposite directions. A norm that came back near
  /// zero, or a gate decay that came back exponentiated twice, would both load and run.
  @Test("llama.cpp's norm fold is kept and its A_log fold is undone")
  func conventions() throws {
    let url = URL(filePath: NSTemporaryDirectory())
      .appending(path: "gguf-folds-\(UUID().uuidString).gguf")
    defer { try? FileManager.default.removeItem(at: url) }

    let logs: [Float] = [0.0, 0.5, 1.5, 3.0]
    var writer = Writer()
    writer.metadata = [("general.architecture", 8, Writer.string("qwen35"))]
    writer.tensors = [
      // As llama.cpp writes them: the norm already scaled by one, the decay already negated
      // and exponentiated.
      (
        "blk.0.attn_norm.weight", [4], .f32,
        floats([1.0, 1.25, 0.75, 1.5])
      ),
      (
        "blk.0.ssm_a", [4], .f32,
        floats(logs.map { -Foundation.exp($0) })
      ),
    ]
    try writer.write(to: url)

    let store = try GGUFWeights.load(file: try GGUFFile(url: url))
    let prefix = GGUFTensorNaming.prefix

    let norm = try store(prefix + "layers.0.input_layernorm.weight")
    #expect(abs(norm[1].item(Float.self) - 1.25) < 1e-6)

    let aLog = try store(prefix + "layers.0.linear_attn.A_log")
    for (i, expected) in logs.enumerated() {
      #expect(abs(aLog[i].item(Float.self) - expected) < 1e-5, "A_log[\(i)]")
    }
  }

  @Test("a projection and an embedding read straight off the blocks")
  func modulesRunOffBlocks() throws {
    let url = URL(filePath: NSTemporaryDirectory())
      .appending(path: "gguf-modules-\(UUID().uuidString).gguf")
    defer { try? FileManager.default.removeItem(at: url) }

    let hidden = 256
    let vocab = 16

    var writer = Writer()
    writer.metadata = [("general.architecture", 8, Writer.string("qwen35"))]
    writer.tensors = [
      (
        "token_embd.weight", [vocab, hidden], .iq1_m,
        payload(.iq1_m, elements: vocab * hidden, seed: 33)
      ),
      (
        "blk.0.ffn_up.weight", [hidden, hidden], .iq2_s,
        payload(.iq2_s, elements: hidden * hidden, seed: 44)
      ),
    ]
    try writer.write(to: url)

    let store = try GGUFWeights.load(file: try GGUFFile(url: url))
    let config = try GGUFArchitecture.textConfigForTest(hidden: hidden, vocab: vocab)
    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: GGUFTensorNaming.prefix)

    let projection = try factory.linear("layers.0.mlp.up_proj")
    #expect(projection.inputDim == hidden)
    #expect(projection.outputDim == hidden)

    let blocks = try #require(store.ggml(GGUFTensorNaming.prefix + "layers.0.mlp.up_proj.weight"))
    let expanded = try #require(
      GGMLKernels.dequantize(
        blocks: blocks.bytes, type: blocks.type, shape: blocks.shape, dtype: .float32))

    let x = MLXRandom.normal([1, hidden]).asType(.float32)
    let actual = projection(x)
    let expected = matmul(x, expanded.T)
    let bound = matmul(abs(x), abs(expanded).T)
    eval(actual, expected, bound)
    let error = (abs(actual - expected) / maximum(bound, MLXArray(Float(1e-6)))).max()
      .item(Float.self)
    #expect(error < 1e-5, "projection off by \(error)")

    let embedding = try factory.embedding("embed_tokens")
    let ids = MLXArray([Int32(3), Int32(11)])
    let rows = embedding(ids).asType(.float32)
    let table = try #require(
      store.ggml(GGUFTensorNaming.prefix + "embed_tokens.weight"))
    let full = try #require(
      GGMLKernels.dequantize(
        blocks: table.bytes, type: table.type, shape: table.shape, dtype: .float32))
    eval(rows, full)
    #expect(rows.shape == [2, hidden])
    let gathered = (abs(rows - full[ids])).max().item(Float.self)
    #expect(gathered < 1e-2, "gathered rows differ by \(gathered)")
  }
}

extension GGUFArchitecture {
  /// The smallest config that lets `PackedModuleFactory` hand out modules; the geometry the
  /// real loader resolves is exercised by the architecture suite.
  static func textConfigForTest(hidden: Int, vocab: Int) throws -> BonsaiConfig {
    let json: [String: Any] = [
      "schema_version": 0,
      "model_type": "qwen3_5",
      "modules": [],
      "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
      "components": ["text": true, "vision": false, "mtp": false],
      "text_config": [
        "model_type": "qwen3_5", "hidden_size": hidden, "intermediate_size": hidden,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": hidden, "rms_norm_eps": 1e-6, "vocab_size": vocab,
        "max_position_embeddings": 64, "tie_word_embeddings": false,
        "layer_types": ["full_attention"],
        "linear_num_value_heads": 0, "linear_num_key_heads": 0,
        "linear_value_head_dim": 0, "linear_key_head_dim": 0, "linear_conv_kernel_dim": 0,
        "rope_parameters": ["rope_theta": 10000.0],
      ],
    ]
    return try JSONDecoder().decode(
      BonsaiConfig.self, from: try JSONSerialization.data(withJSONObject: json))
  }
}
