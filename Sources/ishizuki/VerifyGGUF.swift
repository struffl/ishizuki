// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

/// The same checkpoint, read both ways.
///
/// Everything else about the GGUF path is checked against llama.cpp's source: that its converter
/// folds `1 + w` into the norms, that `ssm_a` holds `-exp(A_log)`, that the value heads are
/// retiled. Source can be read wrongly. If any of those is off, a pack and a GGUF built from one
/// checkpoint will disagree here, and nowhere else — both load, both generate, and only one of
/// them means anything.
struct VerifyGGUF: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "verify-gguf",
    abstract: "Compare this runtime's logits against another runtime's, or against a pack.")

  @Option(name: .long, help: "This runtime's pack.") var pack: String?
  @Option(name: .long, help: "The model to check: a pack directory or a .gguf.")
  var model: String
  @Option(
    name: .long,
    help: "Another runtime's logits for this model, from Scripts/gen-llama-logits.c or gen-mlx-logits.py.")
  var reference: String?
  @Option(name: .long, help: "Tokens to run through both.") var prompt =
    "The quick brown fox jumps over the lazy dog."

  func run() throws {
    let file = try BonsaiModel(path: URL(filePath: model))
    if let reference { try compareToReference(file, URL(filePath: reference)) }
    guard let pack else { return }
    try compareToPack(pack, file)
  }

  /// The strongest form of this check: the same weights, read by this runtime and by the one
  /// they were published for. Nothing differs but the code, so a convention read wrongly — a
  /// fold applied twice, a router normalized at the wrong point — has nowhere to hide.
  private func compareToReference(_ model: BonsaiModel, _ url: URL) throws {
    let data = try Data(contentsOf: url)
    guard data.count > 16, data.prefix(8) == Data("LLAMALG1".utf8) else {
      throw BonsaiError.missingComponent("\(url.lastPathComponent) is not a reference dump")
    }
    let counts = data.withUnsafeBytes { raw -> (Int, Int) in
      (
        Int(raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)),
        Int(raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self))
      )
    }
    let (tokenCount, vocab) = counts
    let tokens = (0..<tokenCount).map { i in
      data.withUnsafeBytes {
        Int($0.loadUnaligned(fromByteOffset: 16 + 4 * i, as: Int32.self))
      }
    }
    let base = 16 + 4 * tokenCount
    let wanted = (0..<vocab).map { i in
      data.withUnsafeBytes {
        $0.loadUnaligned(fromByteOffset: base + 4 * i, as: Float.self)
      }
    }

    print("")
    print("against the reference on the same weights:")
    let ours = model.tokenizer.encode(
      prompt, addSpecialTokens: false)
    print("  tokens: \(tokens.count) theirs, \(ours.count) ours\(ours == tokens ? "" : " — DIFFERENT")")

    let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
    let got = model.text(input, cache: nil).asType(.float32)[0, tokens.count - 1]
    let want = MLXArray(wanted)
    eval(got, want)
    report(got, want)
  }

  private func compareToPack(_ pack: String, _ file: BonsaiModel) throws {
    let packed = try BonsaiModel(path: URL(filePath: pack))
    print("")
    print("against a pack of the same checkpoint:")
    print("  pack: \(packed.config.textConfig.numHiddenLayers) layers")

    guard packed.config.textConfig.vocabSize == file.config.textConfig.vocabSize else {
      throw BonsaiError.shapeMismatch(
        "the two carry different vocabularies: \(packed.config.textConfig.vocabSize) "
          + "and \(file.config.textConfig.vocabSize)")
    }

    // Tokenized once, by the pack, so a tokenizer difference shows up as its own line rather
    // than as a logit difference.
    let ids = packed.tokenizer.encode(prompt)
    let alternative = file.tokenizer.encode(prompt)
    print("tokens: \(ids.count) from the pack, \(alternative.count) from the GGUF")
    if ids != alternative {
      print("  the two tokenizers disagree, which every number below inherits")
    }

    let input = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
    let a = packed.text(input, cache: nil).asType(.float32)
    let b = file.text(input, cache: nil).asType(.float32)
    eval(a, b)

    let last = ids.count - 1
    report(a[0, last], b[0, last])
  }

  private func report(_ one: MLXArray, _ two: MLXArray) {
    let spread = maximum(abs(one).max(), abs(two).max()).item(Float.self)
    let delta = abs(one - two).max().item(Float.self)

    let meanOne = one.mean()
    let meanTwo = two.mean()
    let covariance = ((one - meanOne) * (two - meanTwo)).mean().item(Float.self)
    let deviation = (one.variance().sqrt() * two.variance().sqrt()).item(Float.self)
    let correlation = covariance / max(deviation, 1e-9)

    print(String(format: "  max|Δ| : %.4f over a range of %.1f", delta, spread))
    print(String(format: "  correlation : %.6f", correlation))
    print("  argmax ours=\(one.argMax().item(Int.self)) theirs=\(two.argMax().item(Int.self))")

    // A fold applied twice does not shift a logit slightly; it changes what the model is.
    // Quantization noise between two different block formats is the only expected difference.
    if correlation > 0.999 {
      print("  the same model")
    } else if correlation > 0.9 {
      print("  close: a different checkpoint or quantization, not a different convention")
    } else {
      print("  not the same model — check the folds in GGUFWeights")
    }
  }
}
