// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DeepSeek-V4.1's tokenizer and engram vocabulary, against Hugging Face's own.

import CryptoKit
import Foundation
import Testing

@testable import IshizukiKit

/// Where a real V4.1 checkpoint is: `ISHIZUKI_DEEPSEEK`, or the release on the external disk.
enum DeepSeekRelease {
  static var directory: URL? {
    let candidates = [
      ProcessInfo.processInfo.environment["ISHIZUKI_DEEPSEEK"],
      "/Volumes/tank/models/DeepSeek-V4.1-Flash",
    ].compactMap { $0 }
    return candidates.map { URL(fileURLWithPath: $0) }.first {
      FileManager.default.fileExists(atPath: $0.appending(path: "tokenizer.json").path)
    }
  }
}

/// These read the release's `tokenizer.json`, which is too large to keep as a fixture, so they
/// run only where a checkpoint is.
@Suite("DeepSeek-V4.1 tokenizer", .enabled(if: DeepSeekRelease.directory != nil))
struct DeepSeekTokenizerTests {
  private func tokenizer() throws -> BonsaiTokenizer {
    try BonsaiTokenizer(directory: try #require(DeepSeekRelease.directory))
  }

  @Test("splits and merges the way Hugging Face does")
  func encodesLikeHuggingFace() throws {
    let tokenizer = try tokenizer()
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .appending(path: "Fixtures/deepseek-v41/tokenizer-cases.json")
    let cases = try #require(
      try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
    for entry in cases {
      let text = try #require(entry["text"] as? String)
      let want = try #require(entry["ids"] as? [Int])
      #expect(tokenizer.encode(text) == want, "\(text.debugDescription) tokenized differently")
    }
    #expect(tokenizer.eosTokenIds.contains(1))
  }

  @Test("folds the vocabulary into the 99,092 ids the engram was hashed over")
  func foldsTheEngramVocabulary() throws {
    let map = DeepSeekTokenMap.build(tokenizer: try tokenizer(), count: 129_280)
    #expect(Int(map.max()! + 1) == 99_092)
    let json = "[" + map.map { String($0) }.joined(separator: ", ") + "]"
    let digest = SHA256.hash(data: Data(json.utf8)).map { String(format: "%02x", $0) }.joined()
    #expect(digest == "de40bcdd71f24430ba82752ef385093be962adbf845a1c97db39161597d5dc28")
  }
}
