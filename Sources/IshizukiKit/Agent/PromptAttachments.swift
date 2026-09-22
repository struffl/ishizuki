// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// How a prompt says which pictures came with it. The transcript carries text and nothing else,
// so the pictures travel as a marker on the front of the prompt and are resolved from disk.

import Foundation

/// A marker the window writes and the bridge reads back. It lives inside the prompt because
/// that is the one part of a turn a session stores verbatim and hands back unchanged, which is
/// what lets a conversation reopened tomorrow still know what it was shown.
public enum PromptAttachments {
  static let open = "\u{2062}ishizuki:images\n"
  static let close = "\n\u{2062}\n"

  /// The marker for a set of pictures, or nothing at all when there are none.
  public static func marker(for images: [URL]) -> String {
    guard !images.isEmpty else { return "" }
    return open + images.map(\.path).joined(separator: "\n") + close
  }

  /// A prompt split back into the pictures it named and the words that followed them.
  public static func split(_ text: String) -> (images: [String], body: String) {
    guard text.hasPrefix(open), let end = text.range(of: close) else { return ([], text) }
    let paths = text[text.index(text.startIndex, offsetBy: open.count)..<end.lowerBound]
      .split(separator: "\n")
      .map(String.init)
      .filter { !$0.isEmpty }
    return (paths, String(text[end.upperBound...]))
  }

  /// Whether a prompt carries one at all, which is the cheap check a transcript fold wants.
  public static func carriesImages(_ text: String) -> Bool {
    text.hasPrefix(open)
  }
}
