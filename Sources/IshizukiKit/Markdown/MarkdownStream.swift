// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Markdown read a token at a time. A fence that has opened and not closed is a code block
// already, so a block can be drawn and highlighted while it is still being written.

import Foundation

public enum MarkdownBlock: Equatable, Sendable {
  case prose(String)
  /// `closed` is false while the model is still inside the fence, which is most of the time a
  /// block is on screen.
  case code(language: String?, body: String, closed: Bool)
}

public enum MarkdownStream {
  /// Splits what has arrived so far into blocks. Safe to call on every fragment: it reads the
  /// whole string each time and holds no state, so there is nothing to get out of step.
  public static func blocks(in text: String) -> [MarkdownBlock] {
    guard !text.isEmpty else { return [] }

    var lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    // A trailing newline produces an empty final element that is not a line of its own.
    let endsWithNewline = text.hasSuffix("\n")
    if endsWithNewline { lines.removeLast() }

    var blocks: [MarkdownBlock] = []
    var prose: [String] = []
    var body: [String] = []
    var open: Fence?

    func flushProse() {
      // Newlines only: a run of blank lines is noise, but leading spaces can be the content.
      let joined = prose.joined(separator: "\n").trimmingCharacters(in: .newlines)
      if !joined.isEmpty { blocks.append(.prose(joined)) }
      prose.removeAll()
    }

    for (index, line) in lines.enumerated() {
      let isLast = index == lines.count - 1
      // The last line is only half-written unless a newline has landed after it, and half a
      // fence is not a fence.
      let complete = !isLast || endsWithNewline

      if let fence = open {
        // A closing fence needs no newline after it to be recognised: it carries nothing but
        // its own characters, so there is nothing still to learn about it. If more arrives and
        // makes it something else, the next parse of the whole string says so.
        if let closing = Fence(line: line), closing.closes(fence) {
          blocks.append(
            .code(language: fence.language, body: body.joined(separator: "\n"), closed: true))
          body.removeAll()
          open = nil
        } else {
          body.append(line)
        }
        continue
      }

      if complete, let fence = Fence(line: line) {
        flushProse()
        open = fence
        continue
      }
      prose.append(line)
    }

    if let fence = open {
      blocks.append(
        .code(language: fence.language, body: body.joined(separator: "\n"), closed: false))
    } else {
      flushProse()
    }
    return blocks
  }

  private struct Fence {
    var marker: Character
    var length: Int
    var language: String?

    init?(line: String) {
      // CommonMark allows a fence to be indented by up to three spaces.
      let stripped = line.drop(while: { $0 == " " })
      guard line.count - stripped.count <= 3, let marker = stripped.first,
        marker == "`" || marker == "~"
      else { return nil }
      let length = stripped.prefix(while: { $0 == marker }).count
      guard length >= 3 else { return nil }

      let info = stripped.dropFirst(length).trimmingCharacters(in: .whitespaces)
      // A backtick fence cannot carry a backtick in its info string.
      if marker == "`", info.contains("`") { return nil }

      self.marker = marker
      self.length = length
      let name = info.split(separator: " ").first.map(String.init)
      self.language = (name?.isEmpty ?? true) ? nil : name?.lowercased()
    }

    /// A closing fence is the same character, at least as long, and carries nothing else.
    func closes(_ opening: Fence) -> Bool {
      marker == opening.marker && length >= opening.length && language == nil
    }
  }
}
