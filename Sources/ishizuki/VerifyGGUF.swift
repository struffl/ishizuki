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
    abstract: "Compare the logits of a pack and a GGUF of the same checkpoint.")

  @Option(name: .long, help: "This runtime's pack.") var pack: String
  @Option(name: .long, help: "llama.cpp's file for the same checkpoint.") var gguf: String
  @Option(name: .long, help: "Tokens to run through both.") var prompt =
    "The quick brown fox jumps over the lazy dog."

  func run() throws {
    let packed = try BonsaiModel(path: URL(filePath: pack))
    print("pack: \(packed.config.textConfig.numHiddenLayers) layers, loaded")
    let file = try BonsaiModel(path: URL(filePath: gguf))
    print("gguf: \(file.config.textConfig.numHiddenLayers) layers, loaded")

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
    let one = a[0, last]
    let two = b[0, last]
    let spread = maximum(abs(one).max(), abs(two).max()).item(Float.self)
    let delta = abs(one - two).max().item(Float.self)

    let meanOne = one.mean()
    let meanTwo = two.mean()
    let covariance = ((one - meanOne) * (two - meanTwo)).mean().item(Float.self)
    let deviation = (one.variance().sqrt() * two.variance().sqrt()).item(Float.self)
    let correlation = covariance / max(deviation, 1e-9)

    print("")
    print(String(format: "max|Δ| at the last position : %.4f over a range of %.1f", delta, spread))
    print(String(format: "correlation                 : %.6f", correlation))
    print("argmax pack=\(one.argMax().item(Int.self)) gguf=\(two.argMax().item(Int.self))")

    // A fold applied twice does not shift a logit slightly; it changes what the model is.
    // Quantization noise between two different block formats is the only expected difference.
    if correlation > 0.999 {
      print("\nthese are the same model")
    } else if correlation > 0.9 {
      print("\nclose, but not the same: suspect one layer rather than a convention")
    } else {
      print("\nthese are not the same model — check the folds in GGUFWeights")
    }
  }
}
