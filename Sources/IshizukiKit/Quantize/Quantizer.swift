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

  private let onProgress: @Sendable (Progress) -> Void

  public init(
    source: SourceCheckpoint, profile: QuantProfile, destination: URL,
    shardLimit: Int = 4 << 30, calibrate: Bool = false, calibrationTokens: [[Int32]]? = nil,
    onProgress: @escaping @Sendable (Progress) -> Void = { _ in }
  ) {
    self.source = source
    self.profile = profile
    self.destination = destination
    self.shardLimit = shardLimit
    self.calibrate = calibrate
    self.calibrationTokens = calibrationTokens
    self.onProgress = onProgress
  }

  /// Tensors that are quantized. Everything else — norms, biases, the delta-net's small state
  /// tensors — is copied through at fp16, because quantizing them costs accuracy and saves
  /// almost nothing.
  public static func isQuantizable(_ name: String, shape: [Int]) -> Bool {
    guard name.hasSuffix(".weight"), shape.count == 2 else { return false }
    let quantizable = [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj",
      "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
      "lm_head", "embed_tokens", "fc",
    ]
    return quantizable.contains { name.contains(".\($0).") || name.hasSuffix(".\($0).weight") }
  }

  public func run() throws -> Outcome {
    let started = Date()
    let fm = FileManager.default
    try fm.createDirectory(at: destination, withIntermediateDirectories: true)

    report(.scanning, 0, 1, "reading the checkpoint", 0, "")
    let names = source.tensorNames.sorted()
    // Upstream checkpoints nest their towers differently from the packs this runtime reads.
    let canonical = TensorNaming.map(names)
    let upstream = TensorNaming.isHuggingFaceLayout(names)
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
      guard weight.dim(1) % profile.groupSize == 0 else {
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

    for name in quantizable {
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

    for name in passthrough {
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
      byteCount: summary.byteCount,
      shards: summary.shards,
      achievedBpw: allocation.achievedBpw,
      histogram: allocation.histogram,
      seconds: -started.timeIntervalSinceNow)
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
