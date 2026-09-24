// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The smaller vocabulary V4.1's engram hashes over, derived from the tokenizer.

import Foundation

/// Folds every token id onto a compressed id shared by every token that normalises alike, so
/// " The", "the" and "THE" hash to the same n-gram rows.
///
/// DeepSeek derives this at load from the tokenizer rather than shipping it: each token is
/// decoded on its own, pushed through NFKC, NFD, accent stripping, lowercasing and whitespace
/// folding, and ids are handed out in order of first appearance. A token that decodes to a
/// broken UTF-8 fragment is keyed by its raw byte-level spelling instead. Every hash multiplier
/// is derived from how many compressed ids come out, so the count is checked against the one
/// the checkpoint was hashed over before anything is looked up. Keys are compared by their
/// bytes: Swift calls canonically equivalent strings equal, and Python does not.
public enum DeepSeekTokenMap {
  public static func build(tokenizer: BonsaiTokenizer, count: Int) -> [Int32] {
    var assigned: [[UInt8]: Int32] = [:]
    var map = [Int32](repeating: 0, count: count)
    for id in 0..<count {
      let text = tokenizer.standaloneText(id)
      let key: String
      if text.unicodeScalars.contains("\u{FFFD}") {
        key = tokenizer.tokenString(id) ?? text
      } else {
        let folded = normalize(text)
        key = folded.isEmpty ? text : folded
      }
      let bytes = Array(key.utf8)
      if let existing = assigned[bytes] {
        map[id] = existing
      } else {
        let next = Int32(assigned.count)
        assigned[bytes] = next
        map[id] = next
      }
    }
    return map
  }

  /// A map written beside the weights, `engram_token_map.json`, when there is one of the right
  /// length. It is only a saving: the hasher still checks it folds into the checkpoint's count.
  public static func saved(in directory: URL, count: Int) -> [Int32]? {
    guard let data = try? Data(contentsOf: directory.appending(path: "engram_token_map.json")),
      let ids = try? JSONSerialization.jsonObject(with: data) as? [NSNumber], ids.count == count
    else { return nil }
    return ids.map { Int32(truncating: $0) }
  }

  static func normalize(_ text: String) -> String {
    let decomposed = text.precomposedStringWithCompatibilityMapping
      .decomposedStringWithCanonicalMapping
    var scalars = String.UnicodeScalarView()
    for scalar in decomposed.unicodeScalars {
      switch scalar.properties.generalCategory {
      case .nonspacingMark, .spacingMark, .enclosingMark: continue
      default: break
      }
      scalars.append(contentsOf: scalar.properties.lowercaseMapping.unicodeScalars)
    }
    var folded = String.UnicodeScalarView()
    var inRun = false
    for scalar in scalars {
      if [" ", "\t", "\r", "\n"].contains(scalar) {
        if !inRun { folded.append(" ") }
        inRun = true
      } else {
        folded.append(scalar)
        inRun = false
      }
    }
    let collapsed = String(folded)
    guard collapsed != " " else { return collapsed }
    let kept = collapsed.unicodeScalars
    guard let first = kept.firstIndex(where: { !$0.properties.isWhitespace }),
      let last = kept.lastIndex(where: { !$0.properties.isWhitespace })
    else { return "" }
    return String(String.UnicodeScalarView(kept[first...last]))
  }
}
