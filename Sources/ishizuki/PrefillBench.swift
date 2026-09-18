// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit
import MLX
import MLXFast
import MLXNN
import MLXRandom

struct PrefillBench: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "prefill-bench",
    abstract: "Split prefill time between the projections and the gated delta-rule scan.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long, help: "Prefill chunk sizes to measure, in tokens.")
  var chunks: String = "512,2048"
  @Option(name: .long) var iterations: Int = 6

  enum Bucket {
    case offloadable
    case recurrent
    case scan
    case head
    case other
  }

  struct Row {
    let label: String
    let layers: Int
    let seconds: Double
    let bucket: Bucket
    var total: Double { seconds * Double(layers) }
  }

  func run() throws {
    let packURL = URL(filePath: model)
    let bonsai = try BonsaiModel(directory: packURL, loadVision: false)
    let text = bonsai.config.textConfig
    let factory = PackedModuleFactory(
      store: bonsai.store, config: bonsai.config, tensorPrefix: bonsai.tensorPrefix)

    let fullAttention = text.isFullAttention
    let attentionLayers = fullAttention.filter { $0 }.count
    let linearLayers = fullAttention.count - attentionLayers
    let linearIndex = fullAttention.firstIndex(of: false) ?? 0
    let attentionIndex = fullAttention.firstIndex(of: true) ?? 3

    print(Style.banner("prefill breakdown"))
    print("")
    print(
      Style.field(
        "layers",
        "\(text.numHiddenLayers)  (\(linearLayers) linear_attention, "
          + "\(attentionLayers) full_attention)"))
    print(Style.field("hidden", "\(text.hiddenSize)   mlp \(text.intermediateSize)"))
    print(
      Style.field(
        "gdn",
        "Hk \(text.linearNumKeyHeads)  Hv \(text.linearNumValueHeads)  "
          + "Dk \(text.linearKeyHeadDim)  Dv \(text.linearValueHeadDim)"))
    print(Style.field("vocab", "\(text.vocabSize)"))

    for field in chunks.split(separator: ",") {
      guard let chunk = Int(field.trimmingCharacters(in: .whitespaces)) else { continue }
      try measure(
        chunk: chunk, bonsai: bonsai, factory: factory,
        linearIndex: linearIndex, attentionIndex: attentionIndex,
        linearLayers: linearLayers, attentionLayers: attentionLayers)
      Memory.clearCache()
    }
  }

  private func measure(
    chunk: Int, bonsai: BonsaiModel, factory: PackedModuleFactory,
    linearIndex: Int, attentionIndex: Int, linearLayers: Int, attentionLayers: Int
  ) throws {
    let text = bonsai.config.textConfig
    let prefix = bonsai.tensorPrefix
    MLXRandom.seed(11)

    let hidden = MLXRandom.normal([1, chunk, text.hiddenSize]).asType(.float16)
    eval(hidden)

    var rows: [Row] = []

    let gdn = "model.layers.\(linearIndex).linear_attn"
    let inProjQKV = try factory.linear(gdn + ".in_proj_qkv")
    let inProjZ = try factory.linear(gdn + ".in_proj_z")
    let outProj = try factory.linear(gdn + ".out_proj")

    let gate = try factory.linear("model.layers.\(linearIndex).mlp.gate_proj")
    let up = try factory.linear("model.layers.\(linearIndex).mlp.up_proj")
    let down = try factory.linear("model.layers.\(linearIndex).mlp.down_proj")

    rows.append(
      Row(
        label: "MLP  gate_proj   \(text.hiddenSize)→\(text.intermediateSize)",
        layers: text.numHiddenLayers,
        seconds: time { eval(gate(hidden)) }, bucket: .offloadable))
    rows.append(
      Row(
        label: "MLP  up_proj     \(text.hiddenSize)→\(text.intermediateSize)",
        layers: text.numHiddenLayers,
        seconds: time { eval(up(hidden)) }, bucket: .offloadable))

    let wide = MLXRandom.normal([1, chunk, text.intermediateSize]).asType(.float16)
    eval(wide)
    rows.append(
      Row(
        label: "MLP  down_proj   \(text.intermediateSize)→\(text.hiddenSize)",
        layers: text.numHiddenLayers,
        seconds: time { eval(down(wide)) }, bucket: .other))

    rows.append(
      Row(
        label: "GDN  in_proj_qkv \(inProjQKV.inputDim)→\(inProjQKV.outputDim)",
        layers: linearLayers,
        seconds: time { eval(inProjQKV(hidden)) }, bucket: .recurrent))
    rows.append(
      Row(
        label: "GDN  in_proj_z   \(inProjZ.inputDim)→\(inProjZ.outputDim)",
        layers: linearLayers,
        seconds: time { eval(inProjZ(hidden)) }, bucket: .offloadable))

    let gated = MLXRandom.normal([1, chunk, outProj.inputDim]).asType(.float16)
    eval(gated)
    rows.append(
      Row(
        label: "GDN  out_proj    \(outProj.inputDim)→\(outProj.outputDim)",
        layers: linearLayers,
        seconds: time { eval(outProj(gated)) }, bucket: .other))

    let convWeight = try bonsai.store(prefix + gdn + ".conv1d.weight")
    let convDim = inProjQKV.outputDim
    let keep = text.linearConvKernelDim - 1
    let convInput = MLXRandom.normal([1, chunk + keep, convDim]).asType(.float16)
    eval(convInput)
    rows.append(
      Row(
        label: "GDN  conv1d + silu",
        layers: linearLayers,
        seconds: time { eval(silu(conv1d(convInput, convWeight, groups: convDim))) },
        bucket: .other))

    let hk = text.linearNumKeyHeads
    let hv = text.linearNumValueHeads
    let dk = text.linearKeyHeadDim
    let dv = text.linearValueHeadDim
    let q = MLXRandom.normal([1, chunk, hk, dk]).asType(.float16)
    let k = MLXRandom.normal([1, chunk, hk, dk]).asType(.float16)
    let v = MLXRandom.normal([1, chunk, hv, dv]).asType(.float16)
    let g = MLXRandom.uniform(low: 0.9, high: 1.0, [1, chunk, hv]).asType(.float32)
    let beta = MLXRandom.uniform(low: 0.0, high: 1.0, [1, chunk, hv]).asType(.float16)
    let state = MLXArray.zeros([1, hv, dv, dk], dtype: .float32)
    eval(q, k, v, g, beta, state)

    rows.append(
      Row(
        label: "GDN  delta rule (sequential scan)",
        layers: linearLayers,
        seconds: time {
          let (y, s) = GatedDeltaNet.deltaRule(
            q: q, k: k, v: v, g: g, beta: beta, state: state,
            headRepeat: hv / hk)
          eval(y, s)
        }, bucket: .scan))

    let attn = "model.layers.\(attentionIndex).self_attn"
    let qProj = try factory.linear(attn + ".q_proj")
    let kProj = try factory.linear(attn + ".k_proj")
    let vProj = try factory.linear(attn + ".v_proj")
    let oProj = try factory.linear(attn + ".o_proj")
    let attnOut = MLXRandom.normal([1, chunk, oProj.inputDim]).asType(.float16)
    eval(attnOut)

    rows.append(
      Row(
        label: "attn q/k/v/o projections",
        layers: attentionLayers,
        seconds: time {
          eval(qProj(hidden), kProj(hidden), vProj(hidden), oProj(attnOut))
        }, bucket: .other))

    let heads = text.numAttentionHeads
    let kvHeads = text.numKeyValueHeads
    let headDim = text.headDim
    let queries = MLXRandom.normal([1, heads, chunk, headDim]).asType(.float16)
    let keys = MLXRandom.normal([1, kvHeads, chunk, headDim]).asType(.float16)
    let values = MLXRandom.normal([1, kvHeads, chunk, headDim]).asType(.float16)
    let mask = causalMask(length: chunk, offset: 0, dtype: .float16)
    eval(queries, keys, values)

    rows.append(
      Row(
        label: "attn scaled dot-product",
        layers: attentionLayers,
        seconds: time {
          eval(
            MLXFast.scaledDotProductAttention(
              queries: queries, keys: keys, values: values,
              scale: 1.0 / Float(headDim).squareRoot(), mask: mask))
        }, bucket: .other))

    rows.append(
      Row(
        label: "lm_head  \(text.hiddenSize)→\(text.vocabSize)  (last position)",
        layers: 1,
        seconds: time { eval(bonsai.text.lastLogits(hidden)) }, bucket: .head))

    let tokens = MLXArray(Array(repeating: Int32(1), count: chunk)).reshaped([1, chunk])
    let measured = time(2) {
      let cache = bonsai.text.makeCache()
      eval(bonsai.text.hidden(inputs: tokens, cache: cache))
    }

    let allPositions = time { eval(bonsai.text.lmHead(hidden)) }

    report(chunk: chunk, rows: rows, measured: measured, allPositions: allPositions)
  }

  private func report(
    chunk: Int, rows: [Row], measured: Double, allPositions: Double
  ) {
    let modelled = rows.reduce(0) { $0 + $1.total }

    print("")
    print(Style.rule)
    print(Style.bright("chunk = \(chunk) tokens"))
    print(Style.rule)
    print(Style.muted("  " + pad("component", 38) + "  per call  layers      total   share"))

    for row in rows {
      let line =
        "  " + pad(row.label, 38)
        + String(
          format: "%8.2f ms %5d  %8.1f ms  %5.1f%%",
          row.seconds * 1000, row.layers, row.total * 1000,
          row.total / modelled * 100)
      switch row.bucket {
      case .scan: print(Style.bad(line))
      case .offloadable: print(Style.good(line))
      case .head: print(Style.warn(line))
      default: print(line)
      }
    }

    print(Style.rule)
    print("  " + pad("sum of parts", 38) + String(format: "%23.1f ms", modelled * 1000))
    print(
      "  " + pad("measured chunk forward", 38)
        + String(
          format: "%23.1f ms   (%.0f%% attributed)", measured * 1000,
          modelled / measured * 100))

    func bucketTotal(_ bucket: Bucket) -> Double {
      rows.filter { $0.bucket == bucket }.reduce(0) { $0 + $1.total }
    }

    func verdict(_ label: String, _ seconds: Double) -> String {
      pad("  " + label, 34)
        + String(
          format: "%8.1f ms   %5.1f%% of prefill", seconds * 1000,
          seconds / measured * 100)
    }

    print("")
    print(Style.bad(verdict("delta-rule scan", bucketTotal(.scan))))
    print(Style.good(verdict("offloadable (gate/up + GDN z)", bucketTotal(.offloadable))))
    print(
      Style.warn(
        pad("  lm_head over all positions", 34)
          + String(
            format: "%8.1f ms   %5.1f%% if it were not sliced", allPositions * 1000,
            allPositions / (measured + allPositions) * 100)))
    print(
      Style.faint(
        "  " + pad("throughput", 32)
          + String(format: "%8.1f tok/s", Double(chunk) / measured)))
  }

  private func pad(_ text: String, _ width: Int) -> String {
    text.count >= width
      ? text : text + String(repeating: " ", count: width - text.count)
  }

  private func time(_ count: Int? = nil, _ body: () -> Void) -> Double {
    let runs = count ?? iterations
    body()
    let start = Date()
    for _ in 0..<runs { body() }
    return -start.timeIntervalSinceNow / Double(runs)
  }
}
