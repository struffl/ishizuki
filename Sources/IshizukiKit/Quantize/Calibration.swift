// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Per-input-channel activation energy, accumulated across a calibration corpus.
///
/// This is oMLX's imatrix signal (`OQImatrixEntry.in_sum2`/`counts`): the mean square of what
/// actually flows into a module's input channels, which `WeightedAffineQuantizer` uses to
/// decide which channels a group's quantization range should be clipped in favor of. A channel
/// nothing ever activates is one whose exact weight barely matters; a channel that regularly
/// carries a lot of energy is one worth spending precision on, even if that means clipping an
/// outlier weight the model rarely if ever exercises.
public final class ActivationCollector: @unchecked Sendable {
  private struct Entry {
    var sumSquares: MLXArray
    var count: Int
  }

  private var entries: [String: Entry] = [:]
  private let lock = NSLock()

  public init() {}

  /// Records one forward call's input to the module at `path`. `x`'s last axis is the input
  /// channel; every other axis (batch, sequence, ...) is a sample to accumulate over.
  ///
  /// Evaluates immediately rather than leaving this lazy: a calibration run is many small
  /// forward passes across many modules, and letting each one's contribution linger unevaluated
  /// would build one arbitrarily large graph for the whole run instead of many small ones.
  public func record(path: String, x: MLXArray) {
    let flat = x.reshaped([-1, x.dim(-1)]).asType(.float32)
    let sumSquares = (flat * flat).sum(axis: 0)
    eval(sumSquares)
    let count = flat.dim(0)

    lock.lock()
    defer { lock.unlock() }
    if let existing = entries[path] {
      entries[path] = Entry(sumSquares: existing.sumSquares + sumSquares, count: existing.count + count)
    } else {
      entries[path] = Entry(sumSquares: sumSquares, count: count)
    }
  }

  /// Mean squared activation per input channel, or `nil` for a path calibration never saw —
  /// a module the calibration corpus never reached, or one (like an embedding) this collector
  /// was never wired to in the first place.
  public func importance(for path: String) -> MLXArray? {
    lock.lock()
    defer { lock.unlock() }
    guard let entry = entries[path], entry.count > 0 else { return nil }
    return entry.sumSquares / Float(entry.count)
  }

  public var paths: [String] {
    lock.lock()
    defer { lock.unlock() }
    return Array(entries.keys)
  }
}

/// Runs a source checkpoint's exact, unquantized forward pass over a calibration corpus, to
/// measure the real per-channel activation energy `WeightedAffineQuantizer` needs.
///
/// This reuses `TextModel`/`Attention`/`GatedDeltaNet`/`MLP` exactly as the real inference path
/// does, rather than re-deriving the architecture's math a second time: `PackedModuleFactory`'s
/// `dense` mode hands every module back as a `PackedLinear` over the real fp16 weight (see
/// `PackedLinear.init(dense:)`), so the classes that do the actual computation cannot tell this
/// from a real, about-to-be-served pack. That is deliberate — a calibration forward that quietly
/// diverged from the production one would measure the wrong function.
public final class CalibrationModel: @unchecked Sendable {
  public let text: TextModel
  public let collector: ActivationCollector
  public let tensorPrefix: String

  public init(source: SourceCheckpoint) throws {
    let config = try BonsaiConfig.standard(source.config)
    let store = try Self.denseStore(from: source)
    let collector = ActivationCollector()
    self.collector = collector
    self.tensorPrefix = store.has("language_model.model.norm.weight") ? "language_model." : ""
    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: tensorPrefix,
      dense: true, collector: collector)
    self.text = try TextModel(config: config, factory: factory, store: store)
  }

  /// Runs one calibration sequence through the trunk. The output itself is discarded — only
  /// the collector's running per-channel statistics matter, and those accumulate as a side
  /// effect of `PackedLinear`'s dense forward path.
  public func calibrate(_ tokens: [Int32]) {
    guard !tokens.isEmpty else { return }
    let ids = MLXArray(tokens).reshaped([1, tokens.count])
    let hidden = text.trunk(inputs: ids, cache: nil, positions: nil)
    eval(hidden)
  }

  public func calibrate(text: String, tokenizer: BonsaiTokenizer, maxTokens: Int = 512) {
    let ids = tokenizer.encode(text).prefix(maxTokens).map { Int32($0) }
    calibrate(Array(ids))
  }

  /// The key a quantized tensor's activation importance is filed under: this model's prefix
  /// and the `.weight` suffix stripped from its canonical name, matching the bare path
  /// `Attention`/`GatedDeltaNet`/`MLP` hand to `PackedModuleFactory.linear` when they build.
  public func importanceKey(forCanonicalName name: String) -> String {
    var key = name
    if key.hasPrefix(tensorPrefix) { key.removeFirst(tensorPrefix.count) }
    if key.hasSuffix(".weight") { key.removeLast(".weight".count) }
    return key
  }

  /// A small, fixed, offline corpus: not curated against any particular model, just enough
  /// topical and structural variety (narrative, technical, dialogue, lists, numbers) to
  /// exercise most of a model's channels without a downloaded dataset. Good enough to start
  /// calibrating from — a real corpus can replace this later without changing anything else
  /// about how calibration runs.
  public static let defaultCorpus: [String] = [
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    "In 1969, Apollo 11 carried three astronauts to the surface of the Moon.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "\"Could you pass the salt?\" she asked, glancing up from her plate.",
    "The stock market fell 2.3% on Tuesday amid concerns over rising interest rates.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
    "1. Preheat the oven to 350°F.\n2. Mix the flour, sugar, and baking soda.\n3. Bake for 25 minutes.",
    "The old lighthouse keeper had not seen another soul in three long, storm-battered months.",
    "SELECT name, age FROM users WHERE age > 21 ORDER BY name ASC;",
    "Quantum entanglement allows two particles to remain correlated regardless of the distance between them.",
    "Dear Sir or Madam, I am writing to formally request a refund for my recent purchase.",
    "The committee voted 7 to 2 in favor of the new zoning ordinance after a lengthy debate.",
    "She laced up her boots, shouldered her pack, and set off into the misty mountains alone.",
    "The GDP grew by 3.1% in the third quarter, exceeding most economists' expectations.",
    "import numpy as np\narr = np.array([1, 2, 3])\nprint(arr.sum())",
    "Once upon a time, in a village surrounded by dense forest, there lived a curious young girl.",
    "The patient presented with a fever of 101.4°F, mild fatigue, and a persistent dry cough.",
    "Turn left at the second traffic light, then continue straight for about half a mile.",
    "The Treaty of Westphalia in 1648 is often cited as the origin of the modern state system.",
    "He whispered, \"We need to leave now, before anyone notices we're gone.\"",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The novel explores themes of memory, loss, and the unreliable nature of storytelling.",
    "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
    "Thank you for your email. I will respond in more detail by end of day tomorrow.",
  ]

  /// The real weight for every tensor the text trunk reads, renamed into this runtime's
  /// canonical layout exactly as `Quantizer` does for its passthrough tensors — calibration
  /// needs the same tensors under the same names, just never quantized.
  private static func denseStore(from source: SourceCheckpoint) throws -> WeightStore {
    let names = source.tensorNames.sorted()
    let canonical = TensorNaming.map(names)
    let upstream = TensorNaming.isHuggingFaceLayout(names)
    let zeroCentredNorms = TensorNaming.usesZeroCentredNorms(source.config)

    var arrays: [String: MLXArray] = [:]
    arrays.reserveCapacity(names.count)
    for name in names {
      var tensor = try source.tensor(name)
      if upstream {
        tensor = TensorNaming.relayout(name, tensor, zeroCentredNorms: zeroCentredNorms)
      }
      arrays[canonical[name] ?? name] = tensor.asType(.float16)
    }
    return WeightStore(arrays: arrays)
  }
}
