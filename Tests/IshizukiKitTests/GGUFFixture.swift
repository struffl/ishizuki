// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One GGUF writer for the suites that need a file rather than bytes.

import Foundation

@testable import IshizukiKit

enum GGUFFixture {
  struct Builder {
    var data = Data()
    var tensors: [(String, [Int], GGMLType)] = []
    var metadata: [(String, UInt32, Data)] = []

    mutating func put<T>(_ value: T) {
      withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
    }

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

    static func f32(_ value: Float) -> Data { u32(value.bitPattern) }

    static func intArray(_ values: [Int]) -> Data {
      var out = u32(4)
      withUnsafeBytes(of: UInt64(values.count)) { out.append(contentsOf: $0) }
      for value in values { out.append(u32(UInt32(value))) }
      return out
    }

    static func stringArray(_ values: [String]) -> Data {
      var out = u32(8)
      withUnsafeBytes(of: UInt64(values.count)) { out.append(contentsOf: $0) }
      for value in values { out.append(string(value)) }
      return out
    }

    mutating func write(to url: URL) throws {
      data = Data()
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
      for (name, dims, type) in tensors {
        data.append(Self.string(name))
        put(UInt32(dims.count))
        for dim in dims.reversed() { put(UInt64(dim)) }
        put(type.rawValue)
        cursor = (cursor + 31) / 32 * 32
        put(UInt64(cursor))
        cursor += type.byteCount(elements: dims.reduce(1, *))
      }

      let padding = (32 - data.count % 32) % 32
      data.append(Data(repeating: 0, count: padding))
      data.append(Data(repeating: 0xab, count: cursor))
      try data.write(to: url)
    }
  }

  struct Writer {
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

  /// Every tensor the text model asks for, at a geometry a test can afford: two layers, one
  /// linear-attention and one full-attention, with the projections in blocks so the kernels are
  /// exercised rather than bypassed.
  ///
  /// The shapes are the model's, not a simplification — `in_proj_qkv` is still
  /// `2 * keyDim + valueDim` rows, `q_proj` still carries its output gate — so a wiring mistake
  /// in the loader shows up here as a shape error rather than as fluent nonsense later.
  struct TinyModel {
    var hidden = 256
    var layers = 2
    var heads = 4
    var headDim = 64
    var kvHeads = 2
    var intermediate = 512
    var vocab = 512
    var keyHeads = 2
    var keyHeadDim = 64
    var valueHeads = 4
    var kernel = 4
    var chatTemplate: String?

    var valueHeadDim: Int { innerSize / valueHeads }
    var innerSize: Int { 256 }
    var keyDim: Int { keyHeads * keyHeadDim }
    var valueDim: Int { valueHeads * valueHeadDim }
    var convDim: Int { 2 * keyDim + valueDim }
    /// Full attention every other layer, so both kinds are built.
    var interval: Int { 2 }

    func isFull(_ layer: Int) -> Bool { (layer + 1) % interval == 0 }

    /// Bytes that decode to something finite: a random fp16 scale is inf or nan often enough
    /// to poison a whole block, and the point here is the wiring, not the arithmetic.
    private func payload(_ type: GGMLType, elements: Int, seed: UInt64) -> Data {
      var state = seed &+ 1
      var bytes = [UInt8]()
      let count = elements / type.blockSize * type.typeSize
      bytes.reserveCapacity(count)
      for _ in 0..<count {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        bytes.append(UInt8((state >> 33) & 0x3f))
      }
      return Data(bytes)
    }

    private func floats(_ values: [Float]) -> Data {
      var out = Data()
      for value in values { withUnsafeBytes(of: value) { out.append(contentsOf: $0) } }
      return out
    }

    func write(to url: URL) throws {
      var writer = Writer()
      writer.metadata = [
        ("general.architecture", 8, Writer.string("qwen35")),
        ("qwen35.block_count", 4, Writer.u32(UInt32(layers))),
        ("qwen35.context_length", 4, Writer.u32(4096)),
        ("qwen35.embedding_length", 4, Writer.u32(UInt32(hidden))),
        ("qwen35.feed_forward_length", 4, Writer.u32(UInt32(intermediate))),
        ("qwen35.attention.head_count", 4, Writer.u32(UInt32(heads))),
        ("qwen35.attention.head_count_kv", 4, Writer.u32(UInt32(kvHeads))),
        ("qwen35.attention.key_length", 4, Writer.u32(UInt32(headDim))),
        ("qwen35.attention.layer_norm_rms_epsilon", 6, Builder.f32(1e-6)),
        ("qwen35.rope.freq_base", 6, Builder.f32(10_000_000)),
        ("qwen35.rope.dimension_count", 4, Writer.u32(UInt32(headDim / 2))),
        ("qwen35.rope.dimension_sections", 9, Builder.intArray([6, 5, 5, 0])),
        ("qwen35.ssm.conv_kernel", 4, Writer.u32(UInt32(kernel))),
        ("qwen35.ssm.state_size", 4, Writer.u32(UInt32(keyHeadDim))),
        ("qwen35.ssm.group_count", 4, Writer.u32(UInt32(keyHeads))),
        ("qwen35.ssm.time_step_rank", 4, Writer.u32(UInt32(valueHeads))),
        ("qwen35.ssm.inner_size", 4, Writer.u32(UInt32(innerSize))),
        ("qwen35.full_attention_interval", 4, Writer.u32(UInt32(interval))),
        ("tokenizer.ggml.tokens", 9, Builder.stringArray((0..<vocab).map { "t\($0)" })),
        ("tokenizer.ggml.eos_token_id", 4, Writer.u32(UInt32(vocab - 1))),
      ]
      if let chatTemplate {
        writer.metadata.append(("tokenizer.chat_template", 8, Writer.string(chatTemplate)))
      }

      var seed: UInt64 = 0
      var tensors: [(name: String, dims: [Int], type: GGMLType, payload: Data)] = []
      func blocks(_ name: String, _ dims: [Int]) {
        seed += 1
        tensors.append(
          (name, dims, .iq4_xs, payload(.iq4_xs, elements: dims.reduce(1, *), seed: seed)))
      }
      func dense(_ name: String, _ dims: [Int]) {
        seed += 1
        tensors.append(
          (name, dims, .f32, payload(.f32, elements: dims.reduce(1, *), seed: seed)))
      }

      blocks("token_embd.weight", [vocab, hidden])
      blocks("output.weight", [vocab, hidden])
      dense("output_norm.weight", [hidden])

      for layer in 0..<layers {
        let blk = "blk.\(layer)"
        dense("\(blk).attn_norm.weight", [hidden])
        dense("\(blk).post_attention_norm.weight", [hidden])
        blocks("\(blk).ffn_gate.weight", [intermediate, hidden])
        blocks("\(blk).ffn_up.weight", [intermediate, hidden])
        blocks("\(blk).ffn_down.weight", [hidden, intermediate])

        if isFull(layer) {
          // The query projection carries its output gate, so it is twice as tall as the heads.
          blocks("\(blk).attn_q.weight", [heads * 2 * headDim, hidden])
          blocks("\(blk).attn_k.weight", [kvHeads * headDim, hidden])
          blocks("\(blk).attn_v.weight", [kvHeads * headDim, hidden])
          blocks("\(blk).attn_output.weight", [hidden, heads * headDim])
          dense("\(blk).attn_q_norm.weight", [headDim])
          dense("\(blk).attn_k_norm.weight", [headDim])
        } else {
          blocks("\(blk).attn_qkv.weight", [convDim, hidden])
          blocks("\(blk).attn_gate.weight", [valueDim, hidden])
          blocks("\(blk).ssm_alpha.weight", [valueHeads, hidden])
          blocks("\(blk).ssm_beta.weight", [valueHeads, hidden])
          blocks("\(blk).ssm_out.weight", [hidden, valueDim])
          dense("\(blk).ssm_conv1d.weight", [convDim, kernel])
          dense("\(blk).ssm_norm.weight", [valueHeadDim])
          // llama.cpp writes -exp(A_log) here, which is always negative; random bytes would be
          // positive and the loader's log(-a) would hand the gate a NaN.
          seed += 1
          tensors.append(
            (
              "\(blk).ssm_a", [valueHeads], .f32,
              floats((0..<valueHeads).map { -1.0 - Float($0) * 0.25 })
            ))
          dense("\(blk).ssm_dt.bias", [valueHeads])
        }
      }

      writer.tensors = tensors
      try writer.write(to: url)
    }
  }

  /// The mmproj half: a tower small enough to run, with its patch embedding split into the two
  /// Conv2Ds ggml stores instead of a Conv3D. `outHidden` matches `TinyModel.hidden`, because a
  /// merger that does not is a tower bolted to a different model.
  struct TinyTower {
    var depth = 2
    var hidden = 32
    var intermediate = 64
    var heads = 2
    var patch = 16
    var merge = 2
    var outHidden = 256
    var side = 4
    var projector = "qwen3vl"
    var deepstack: [Int]?

    private func floats(_ count: Int, seed: UInt64) -> Data {
      var state = seed &+ 1
      var out = Data()
      for _ in 0..<count {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let unit = Float(Int32(bitPattern: UInt32(truncatingIfNeeded: state >> 33)))
        withUnsafeBytes(of: unit / 2_147_483_648 * 0.1) { out.append(contentsOf: $0) }
      }
      return out
    }

    func write(to url: URL) throws {
      var writer = Writer()
      writer.metadata = [
        ("clip.has_vision_encoder", 7, Data([1])),
        ("clip.vision.projector_type", 8, Writer.string(projector)),
        ("clip.vision.block_count", 4, Writer.u32(UInt32(depth))),
        ("clip.vision.embedding_length", 4, Writer.u32(UInt32(hidden))),
        ("clip.vision.feed_forward_length", 4, Writer.u32(UInt32(intermediate))),
        ("clip.vision.attention.head_count", 4, Writer.u32(UInt32(heads))),
        ("clip.vision.patch_size", 4, Writer.u32(UInt32(patch))),
        ("clip.vision.image_size", 4, Writer.u32(UInt32(side * patch))),
        ("clip.vision.spatial_merge_size", 4, Writer.u32(UInt32(merge))),
        ("clip.vision.projection_dim", 4, Writer.u32(UInt32(outHidden))),
      ]
      if let deepstack {
        writer.metadata.append(
          ("clip.vision.is_deepstack_layers", 9, Builder.intArray(deepstack)))
      }

      let merged = hidden * merge * merge
      var seed: UInt64 = 0
      var tensors: [(name: String, dims: [Int], type: GGMLType, payload: Data)] = []
      func dense(_ name: String, _ dims: [Int]) {
        seed += 1
        tensors.append((name, dims, .f32, floats(dims.reduce(1, *), seed: seed)))
      }

      dense("v.patch_embd.weight", [hidden, 3, patch, patch])
      dense("v.patch_embd.weight.1", [hidden, 3, patch, patch])
      dense("v.patch_embd.bias", [hidden])
      dense("v.position_embd.weight", [side * side, hidden])
      dense("v.post_ln.weight", [hidden])
      dense("v.post_ln.bias", [hidden])
      dense("mm.0.weight", [merged, merged])
      dense("mm.0.bias", [merged])
      dense("mm.2.weight", [outHidden, merged])
      dense("mm.2.bias", [outHidden])

      for layer in 0..<depth {
        let blk = "v.blk.\(layer)"
        dense("\(blk).ln1.weight", [hidden])
        dense("\(blk).ln1.bias", [hidden])
        dense("\(blk).ln2.weight", [hidden])
        dense("\(blk).ln2.bias", [hidden])
        dense("\(blk).attn_qkv.weight", [3 * hidden, hidden])
        dense("\(blk).attn_qkv.bias", [3 * hidden])
        dense("\(blk).attn_out.weight", [hidden, hidden])
        dense("\(blk).attn_out.bias", [hidden])
        dense("\(blk).ffn_up.weight", [intermediate, hidden])
        dense("\(blk).ffn_up.bias", [intermediate])
        dense("\(blk).ffn_down.weight", [hidden, intermediate])
        dense("\(blk).ffn_down.bias", [hidden])
      }

      writer.tensors = tensors
      try writer.write(to: url)
    }
  }

  static func temporaryURL(_ tag: String = "gguf") -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "\(tag)-\(UUID().uuidString).gguf")
  }

  /// The smallest file `GGUFArchitecture` accepts: the published Qwen3.8-27B geometry, scaled
  /// down to a vocabulary and a layer count a test can afford.
  static func qwen35(at url: URL, layers: Int = 4, vocab: Int = 256) throws {
    var builder = Builder()
    builder.metadata = [
      ("general.architecture", 8, Builder.string("qwen35")),
      ("qwen35.block_count", 4, Builder.u32(UInt32(layers))),
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
      ("tokenizer.ggml.tokens", 9, Builder.stringArray((0..<vocab).map { "t\($0)" })),
      ("tokenizer.ggml.eos_token_id", 4, Builder.u32(UInt32(vocab - 1))),
      ("tokenizer.chat_template", 8, Builder.string("{{ messages }}")),
    ]
    builder.tensors = [
      ("token_embd.weight", [vocab, 5120], .iq1_m),
      ("output.weight", [vocab, 5120], .iq4_xs),
      ("output_norm.weight", [5120], .f32),
    ]
    for layer in 0..<layers {
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
  }
}
