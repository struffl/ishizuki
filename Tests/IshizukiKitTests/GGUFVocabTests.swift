// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// The tokenizer against llama.cpp's own vectors, over llama.cpp's own vocabulary file.

import Foundation
import Testing

@testable import IshizukiKit

/// Everything else in this suite is checked against fixtures this runtime wrote, which proves
/// only that it agrees with itself. `ggml-vocab-qwen35.gguf` and the `.inp`/`.out` beside it are
/// llama.cpp's, produced by its own tokenizer, so a disagreement here is ours.
///
/// The vocabulary is six megabytes and belongs to another project, so it is not vendored. The
/// test looks for a llama.cpp checkout and says nothing when there is not one.
@Suite("GGUF vocabulary")
struct GGUFVocabTests {
  private static let separator = "\n__ggml_vocab_test__\n"

  /// Swift compares strings by canonical equivalence, so a decomposed string is `==` to its
  /// composed form and cannot be told apart that way. The bytes can.
  private static func isComposed(_ text: String) -> Bool {
    Array(text.utf8) == Array(text.precomposedStringWithCanonicalMapping.utf8)
  }

  private static var vocabulary: URL? {
    let roots = [
      ProcessInfo.processInfo.environment["LLAMA_CPP"],
      NSHomeDirectory() + "/genAI/llama.cpp",
    ].compactMap { $0 }
    for root in roots {
      let url = URL(filePath: root).appending(path: "models/ggml-vocab-qwen35.gguf")
      if FileManager.default.fileExists(atPath: url.path) { return url }
    }
    return nil
  }

  /// The `.inp` is the test strings joined by a marker; the `.out` is one line of ids per
  /// string, in the same order, blank where a string tokenizes to nothing.
  private func vectors(_ vocabulary: URL) throws -> [(text: String, ids: [Int])] {
    let inputs = try String(contentsOf: vocabulary.appendingPathExtension("inp"), encoding: .utf8)
      .components(separatedBy: Self.separator)
    var outputs = try String(
      contentsOf: vocabulary.appendingPathExtension("out"), encoding: .utf8
    ).components(separatedBy: "\n")

    // Both files end with the separator's trailing newline rather than another case.
    var texts = inputs
    if texts.last?.isEmpty == true { texts.removeLast() }
    if outputs.count > texts.count { outputs.removeLast(outputs.count - texts.count) }

    return zip(texts, outputs).map { text, line in
      (text, line.split(separator: " ").compactMap { Int($0) })
    }
  }

  @Test("tokenizes llama.cpp's vectors exactly as llama.cpp does")
  func matchesReferenceVectors() throws {
    guard let vocabulary = Self.vocabulary else { return }

    let tokenizer = try BonsaiTokenizer(gguf: try GGUFFile(url: vocabulary))
    let cases = try vectors(vocabulary)
    #expect(cases.count > 20, "only \(cases.count) vectors read")

    var failures: [String] = []
    for (text, expected) in cases where Self.isComposed(text) {
      // llama.cpp's harness runs these with parse_special off, so a literal `<|im_end|>` in the
      // text is BPE'd like any other run of bytes rather than matched whole.
      let actual = tokenizer.encode(text, addSpecialTokens: false)
      if actual != expected {
        failures.append("\(text.debugDescription): \(actual) != \(expected)")
      }
    }
    let report = failures.joined(separator: "\n")
    #expect(failures.isEmpty, "\(failures.count) of \(cases.count) differ:\n\(report)")
  }

  /// The one place the two disagree, pinned so it cannot drift into a surprise: Qwen's
  /// `tokenizer.json` declares an NFC normalizer and transformers applies it, while llama.cpp
  /// implements no normalizer for BPE and tokenizes the bytes it was handed. Decomposed text is
  /// the only input that can tell them apart, and this runtime follows the checkpoint.
  @Test("decomposed text follows the checkpoint's normalizer, not llama.cpp")
  func normalizesLikeTransformers() throws {
    guard let vocabulary = Self.vocabulary else { return }

    let tokenizer = try BonsaiTokenizer(gguf: try GGUFFile(url: vocabulary))
    let decomposed = try vectors(vocabulary)
      .filter { !Self.isComposed($0.text) }
    #expect(!decomposed.isEmpty, "the vectors carry no decomposed text any more")

    for (text, reference) in decomposed {
      let ids = tokenizer.encode(text, addSpecialTokens: false)
      #expect(ids != reference, "\(text.debugDescription) no longer differs")
      // Composed first, it is the same string to both, and they agree again.
      let composed = text.precomposedStringWithCanonicalMapping
      #expect(ids == tokenizer.encode(composed, addSpecialTokens: false))
    }
  }

  @Test("round-trips every one of those strings back")
  func decodesBack() throws {
    guard let vocabulary = Self.vocabulary else { return }

    let tokenizer = try BonsaiTokenizer(gguf: try GGUFFile(url: vocabulary))
    for (text, _) in try vectors(vocabulary) {
      let decoded = tokenizer.decode(tokenizer.encode(text, addSpecialTokens: false))
      let source = text.debugDescription
      #expect(decoded == text.precomposedStringWithCanonicalMapping, "\(source)")
    }
  }
}
