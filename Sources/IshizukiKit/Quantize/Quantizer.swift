// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// Builds a mixed-width pack from a full-precision checkpoint, a layer at a time.
///
/// Two passes over the source, neither of which holds more than a layer:
///
///   survey    every quantizable module is quantized at each width it might be given and
///             compared to the original, which is what the allocation is later made of. This
///             is the slow pass — it is doing the arithmetic of several whole quantizations.
///   write     each module is quantized once more, at the width it was allocated, and the
///             shards are written as they fill.
///
/// Nothing here infers a module's importance from its name. A projection is lifted off the
/// base width because it was measured to suffer there.
public final class Quantizer: @unchecked Sendable {
  public struct Progress: Sendable {
    public enum Phase: String, Sendable {
      case scanning
      case calibrating
      case surveying
      case allocating
      case writing
      case finishing
    }

    public var phase: Phase
    public var done: Int
    public var total: Int
    public var detail: String
    /// Weights read so far, so a long pass can report throughput rather than only a fraction.
    public var elements: Int
    public var note: String

    public init(
      phase: Phase, done: Int, total: Int, detail: String, elements: Int, note: String
    ) {
      self.phase = phase
      self.done = done
      self.total = total
      self.detail = detail
      self.elements = elements
      self.note = note
    }

    public var fraction: Double {
      total > 0 ? min(max(Double(done) / Double(total), 0), 1) : 0
    }
  }

  public struct Outcome: Sendable {
    public let directory: URL
    public let byteCount: Int
    public let shards: Int
    public let achievedBpw: Double
    public let histogram: [Int: Int]
    public let seconds: Double
  }

  public let source: SourceCheckpoint
  public let profile: QuantProfile
  public let destination: URL
  public let shardLimit: Int
  /// When set, a calibration pass runs the checkpoint's real, unquantized forward over a text
  /// corpus first, and every module is quantized against the activation importance it measured
  /// (`WeightedAffineQuantizer`) rather than blind to it (plain `quantized()`). Off by default:
  /// it is a real forward pass over the whole model, not free, and the fast constant-memory
  /// path some callers want stays available without it.
  public let calibrate: Bool
  /// Pre-tokenized calibration sequences, used instead of tokenizing `CalibrationModel
  /// .defaultCorpus`. A caller with a real corpus supplies it here; tests that want to
  /// calibrate without shipping a tokenizer fixture do the same.
  public let calibrationTokens: [[Int32]]?
  /// When set, the routed experts are written beside the pack as one file per sparse layer
  /// rather than into its shards, and the runtime reads them a few at a time. It costs most of
  /// the decode rate and buys a model the machine could not otherwise hold, so it is the
  /// caller's choice rather than a default.
  public let streamExperts: Bool

  private let onProgress: @Sendable (Progress) -> Void

  public init(
    source: SourceCheckpoint, profile: QuantProfile, destination: URL,
    shardLimit: Int = 4 << 30, calibrate: Bool = false, calibrationTokens: [[Int32]]? = nil,
    streamExperts: Bool = false,
    onProgress: @escaping @Sendable (Progress) -> Void = { _ in }
  ) {
    self.source = source
    self.profile = profile
    self.destination = destination
    self.shardLimit = shardLimit
    self.calibrate = calibrate
    self.calibrationTokens = calibrationTokens
    self.streamExperts = streamExperts
    self.onProgress = onProgress
  }

  /// Tensors that are quantized. Everything else — norms, biases, the delta-net's small state
  /// tensors — is copied through at fp16, because quantizing them costs accuracy and saves
  /// almost nothing.
  public static func isQuantizable(_ name: String, shape: [Int]) -> Bool {
    guard name.hasSuffix(".weight") else { return false }
    // A bank of routed experts arrives as one stacked tensor. It quantizes along its last axis
    // like any other projection, and `gatherQuantizedMM` reads an expert's rows straight out
    // of the stack, so there is nothing to unpick first.
    guard shape.count == 2 || (shape.count == 3 && name.contains(".switch_mlp.")) else {
      return false
    }
    let quantizable = [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj",
      "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
      "lm_head", "embed_tokens", "fc",
    ]
    // A multimodal checkpoint nests the head under the language model and a text-only one
    // does not, so the bare name has to match as well as the nested one — otherwise the head
    // of a flat checkpoint is quietly carried at full width.
    return quantizable.contains {
      name.contains(".\($0).") || name.hasSuffix(".\($0).weight") || name.hasPrefix("\($0).")
    }
  }

  public func run() throws -> Outcome {
    let started = Date()
    let fm = FileManager.default
    try fm.createDirectory(at: destination, withIntermediateDirectories: true)

    report(.scanning, 0, 1, "reading the checkpoint", 0, "")
    // The n-gram table and the buffers that address it are lifted out beside the pack, not
    // written into its shards: four gigabytes of rows that a step reads eight of.
    let engrams = try EngramRepack.run(source: source, destination: destination) { note in
      self.report(.writing, 0, 1, note, 0, "")
    }
    let names = source.tensorNames.sorted().filter {
      !$0.hasPrefix("model.ngram_embedding.") && !$0.hasPrefix("model.ple_embedding.")
    }
    // Upstream checkpoints nest their towers differently from the packs this runtime reads.
    let canonical = TensorNaming.map(names)
    // A checkpoint MLX has already been through carries its scales and has had its
    // convolutions and norms relaid out on the way. One straight from upstream has not, and
    // that is true whether or not it nests a vision tower — which is what this used to ask.
    let upstream = TensorNaming.isUpstream(names)
    let zeroCentredNorms = TensorNaming.usesZeroCentredNorms(source.config)
    var quantizable: [String] = []
    var passthrough: [String] = []
    for name in names {
      let shape = try source.tensor(name).shape
      if Self.isQuantizable(canonical[name] ?? name, shape: shape) {
        quantizable.append(name)
      } else {
        passthrough.append(name)
      }
    }

    // Pass zero: calibration, if asked for. Importance is keyed by the original (pre-canonical)
    // name, same as everything else in this pass, so the lookups below never have to reconcile
    // two naming schemes.
    var importance: [String: MLXArray] = [:]
    if calibrate {
      importance = try runCalibration(quantizable: quantizable, canonical: canonical)
    }

    // Pass one.
    let widths = ([profile.baseBits] + profile.boostBits).sorted()
    var measurements: [ModuleMeasurement] = []
    measurements.reserveCapacity(quantizable.count)
    var surveyed = 0
    for name in quantizable {
      let weight = try source.tensor(name)
      // A module whose input width does not divide the group cannot be quantized at this
      // group size; it is carried at fp16 rather than silently reshaped.
      guard weight.dim(weight.ndim - 1) % profile.groupSize == 0 else {
        passthrough.append(name)
        continue
      }
      measurements.append(
        ModuleSurvey.measure(
          weight, path: name, widths: widths, groupSize: profile.groupSize,
          importance: importance[name]))
      surveyed += weight.size
      report(
        .surveying, measurements.count, quantizable.count, short(name), surveyed,
        "quantizing each module at \(widths.map(String.init).joined(separator: "/")) bits to see what it loses"
      )
    }
    quantizable = measurements.map(\.path)

    report(.allocating, 0, 1, "spending the budget", surveyed, "")
    let allocation = BitAllocator(profile: profile).allocate(measurements)
    let canonicalBits = Dictionary(
      uniqueKeysWithValues: allocation.bits.map { (canonical[$0.key] ?? $0.key, $0.value) })

    // Pass two.
    var writer = PackWriter(directory: destination, shardLimit: shardLimit)
    var written = 0
    let totalToWrite = quantizable.count + passthrough.count

    // A streamed pack's experts never reach the shards: they are quantized a layer at a time
    // and cut straight into the blobs the runtime reads slots out of.
    var streamed: Set<String> = []
    var expertBytes = 0
    if streamExperts {
      streamed = Set(
        (quantizable + passthrough).filter { ExpertRepack.isExpert(canonical[$0] ?? $0) })
      if !streamed.isEmpty {
        expertBytes = try writeExperts(
          names: streamed, canonical: canonical, allocation: allocation,
          importance: importance, written: &written, total: totalToWrite, surveyed: surveyed)
      }
    }

    for name in quantizable where !streamed.contains(name) {
      let bits = allocation.bits[name] ?? profile.baseBits
      let weight = try source.tensor(name)
      let wq: MLXArray
      let scales: MLXArray
      let biases: MLXArray?
      if let moduleImportance = importance[name] {
        (wq, scales, biases) = WeightedAffineQuantizer.quantize(
          weight, groupSize: profile.groupSize, bits: bits, importance: moduleImportance)
      } else {
        (wq, scales, biases) = quantized(
          weight, groupSize: profile.groupSize, bits: bits, mode: .affine)
      }
      let target = canonical[name] ?? name
      let base = String(target.dropLast(".weight".count))
      try writer.add(target, wq)
      try writer.add(base + ".scales", scales.asType(.float16))
      try writer.add(base + ".biases", (biases ?? MLXArray.zeros(like: scales)).asType(.float16))
      written += 1
      report(
        .writing, written, totalToWrite, "\(short(name)) at \(bits)-bit", surveyed, "")
    }

    for name in passthrough where !streamed.contains(name) {
      var tensor = try source.tensor(name)
      if upstream {
        tensor = TensorNaming.relayout(name, tensor, zeroCentredNorms: zeroCentredNorms)
      }
      // fp16 throughout: it is what ishizuki runs in, and prefill is faster for it.
      try writer.add(canonical[name] ?? name, tensor.asType(.float16))
      written += 1
      report(.writing, written, totalToWrite, short(name), surveyed, "")
    }

    report(.finishing, 0, 1, "writing the index and config", surveyed, "")
    let summary = try writer.finish()
    try writeConfig(allocation: allocation, bits: canonicalBits)

    return Outcome(
      directory: destination,
      byteCount: summary.byteCount + (engrams?.byteCount ?? 0) + expertBytes,
      shards: summary.shards,
      achievedBpw: allocation.achievedBpw,
      histogram: allocation.histogram,
      seconds: -started.timeIntervalSinceNow)
  }

  // MARK: - Streamed experts

  /// Quantizes one sparse layer's experts at a time and cuts each layer straight into its
  /// blob, so the bank never has to sit in memory whole and never reaches the shards.
  ///
  /// The widths are the allocated ones, the same the shards would have carried: streaming
  /// moves where a weight lives, not what it is worth.
  private func writeExperts(
    names: Set<String>, canonical: [String: String], allocation: BitAllocator.Result,
    importance: [String: MLXArray], written: inout Int, total: Int, surveyed: Int
  ) throws -> Int {
    var byLayer: [Int: [String: String]] = [:]
    for name in names {
      guard let address = ExpertRepack.address(canonical[name] ?? name) else {
        throw BonsaiError.unsupportedModel("\(name) is an expert of no layer this can name")
      }
      byLayer[address.layer, default: [:]][address.part] = name
    }

    var layout: ExpertLayout?
    var bytes = 0
    for layer in byLayer.keys.sorted() {
      var parts: [String: MLXArray] = [:]
      for (part, name) in byLayer[layer]! {
        let tensor = try source.tensor(name)
        // The runtime reads a streamed expert through `gatherQuantizedMM`, which wants a
        // weight and its scales and biases. A projection that cannot be quantized has no
        // streamed form, so this refuses rather than writing a pack that will not load.
        guard part.hasSuffix(".weight"),
          tensor.dim(tensor.ndim - 1) % profile.groupSize == 0
        else {
          throw BonsaiError.unsupportedModel(
            "\(name) cannot be quantized at group \(profile.groupSize), so it cannot stream")
        }
        let bits = allocation.bits[name] ?? profile.baseBits
        let (wq, scales, biases) = quantize(
          tensor, bits: bits, importance: importance[name])
        let base = String(part.dropLast(".weight".count))
        parts[part] = wq
        parts[base + ".scales"] = scales.asType(.float16)
        parts[base + ".biases"] = (biases ?? MLXArray.zeros(like: scales)).asType(.float16)
        written += 1
        report(.writing, written, total, "\(short(name)) at \(bits)-bit, streamed", surveyed, "")
      }

      let expertCount = parts.values.first?.dim(0) ?? 0
      let planned = layout ?? ExpertRepack.layout(of: parts, expertCount: expertCount)
      layout = planned
      bytes += try ExpertRepack.blob(
        parts: parts, layout: planned,
        to: destination.appending(path: ExpertRepack.layerFile(layer)))
    }

    guard let layout else { return 0 }
    try ExpertRepack.writeLayout(layout, to: destination)
    return bytes
  }

  private func quantize(
    _ weight: MLXArray, bits: Int, importance: MLXArray?
  ) -> (MLXArray, MLXArray, MLXArray?) {
    if let importance {
      return WeightedAffineQuantizer.quantize(
        weight, groupSize: profile.groupSize, bits: bits, importance: importance)
    }
    return quantized(weight, groupSize: profile.groupSize, bits: bits, mode: .affine)
  }

  // MARK: - Calibration

  /// Runs the checkpoint's exact forward over the calibration corpus and returns each
  /// quantizable module's importance, keyed by its original (pre-canonical) name so callers
  /// never have to reconcile naming schemes. A module calibration never reached — the
  /// embedding and the head, which oMLX excludes for the same reason (a row is gathered by
  /// token id, not consumed as an input channel), or one an unusually short corpus missed —
  /// is simply absent, and falls back to plain, unweighted quantization.
  private func runCalibration(
    quantizable: [String], canonical: [String: String]
  ) throws -> [String: MLXArray] {
    let model = try CalibrationModel(source: source)

    if let calibrationTokens {
      report(.calibrating, 0, calibrationTokens.count, "loading the checkpoint", 0, "")
      for (index, tokens) in calibrationTokens.enumerated() {
        model.calibrate(tokens)
        report(
          .calibrating, index + 1, calibrationTokens.count,
          "measuring what the real network actually uses", 0, "")
      }
    } else {
      report(.calibrating, 0, CalibrationModel.defaultCorpus.count, "loading the checkpoint", 0, "")
      let tokenizer = try BonsaiTokenizer(directory: source.directory)
      for (index, text) in CalibrationModel.defaultCorpus.enumerated() {
        model.calibrate(text: text, tokenizer: tokenizer)
        report(
          .calibrating, index + 1, CalibrationModel.defaultCorpus.count,
          "measuring what the real network actually uses", 0, "")
      }
    }

    var importance: [String: MLXArray] = [:]
    for name in quantizable {
      let key = model.importanceKey(forCanonicalName: canonical[name] ?? name)
      importance[name] = model.collector.importance(for: key)
    }
    return importance
  }

  // MARK: - Config

  /// Writes the source's config back out with a quantization block in the shape MLX uses: the
  /// base width at the top, and an entry for every module that was lifted off it.
  private func writeConfig(allocation: BitAllocator.Result, bits: [String: Int]) throws {
    var config = source.config
    var quantization: [String: Any] = [
      "bits": profile.baseBits,
      "group_size": profile.groupSize,
      "mode": "affine",
    ]
    for (path, width) in bits where width != profile.baseBits {
      let module = String(path.dropLast(".weight".count))
      quantization[module] = [
        "bits": width, "group_size": profile.groupSize, "mode": "affine",
      ]
    }
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    config["ishizuki_profile"] = [
      "name": profile.name,
      "target_bpw": profile.targetBpw,
      "achieved_bpw": allocation.achievedBpw,
      "boosted_modules": allocation.boosted,
      "quantized_modules": allocation.total,
    ]

    let data = try JSONSerialization.data(
      withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: destination.appending(path: "config.json"))

    // The tokenizer and its template travel with the pack, or it cannot be served.
    let fm = FileManager.default
    for name in [
      "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
      "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
    ] {
      let from = source.directory.appending(path: name)
      guard fm.fileExists(atPath: from.path) else { continue }
      let to = destination.appending(path: name)
      if fm.fileExists(atPath: to.path) { try fm.removeItem(at: to) }
      try fm.copyItem(at: from.resolvingSymlinksInPath(), to: to)
    }
  }

  private func short(_ name: String) -> String {
    name.replacingOccurrences(of: "language_model.model.layers.", with: "L")
      .replacingOccurrences(of: ".weight", with: "")
      .replacingOccurrences(of: "language_model.", with: "")
  }

  private func report(
    _ phase: Progress.Phase, _ done: Int, _ total: Int, _ detail: String,
    _ elements: Int, _ note: String
  ) {
    onProgress(
      Progress(
        phase: phase, done: done, total: total, detail: detail, elements: elements,
        note: note))
  }
}
