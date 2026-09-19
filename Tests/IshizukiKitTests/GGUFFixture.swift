// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
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
