// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Markdown as it arrives: prose with its inline marks, and fenced code highlighted from the
// moment the fence opens rather than once it closes.

import Highlightr
import IshizukiKit
import SwiftUI

/// One highlighter for the app, with the last few results kept. Highlighting runs through
/// JavaScriptCore, so a streaming block asking for the same work every frame is worth avoiding.
@MainActor
final class SyntaxHighlighter {
  static let shared = SyntaxHighlighter()

  private let highlightr = Highlightr()
  private var cache: [Key: AttributedString] = [:]
  private var order: [Key] = []
  private let limit = 96
  private var dark: Bool?

  private struct Key: Hashable {
    var code: String
    var language: String?
    var dark: Bool
  }

  var languages: [String] { highlightr?.supportedLanguages() ?? [] }

  func supports(_ language: String?) -> Bool {
    guard let language else { return false }
    return highlightr?.supportedLanguages().contains(language) ?? false
  }

  func highlight(_ code: String, language: String?, dark: Bool, font: NSFont) -> AttributedString? {
    guard let highlightr, supports(language) else { return nil }

    let key = Key(code: code, language: language, dark: dark)
    if let hit = cache[key] { return hit }

    if self.dark != dark {
      _ = highlightr.setTheme(to: dark ? "atom-one-dark" : "atom-one-light")
      self.dark = dark
    }
    highlightr.theme.codeFont = font

    guard let highlighted = highlightr.highlight(code, as: language, fastRender: true) else {
      return nil
    }
    let result = AttributedString(highlighted)

    cache[key] = result
    order.append(key)
    if order.count > limit {
      cache[order.removeFirst()] = nil
    }
    return result
  }
}

@available(macOS 27.0, *)
struct MarkdownText: View {
  let text: String
  let mono: Font
  let size: Double

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      ForEach(Array(MarkdownStream.blocks(in: text).enumerated()), id: \.offset) { _, block in
        switch block {
        case .prose(let prose):
          ProseText(text: prose, size: size)
        case .code(let language, let body, let closed):
          CodeBlock(language: language, code: body, closed: closed, mono: mono, size: size)
        }
      }
    }
  }
}

/// Headings and bullets are drawn here rather than handed to the markdown parser, which would
/// collapse the whitespace a streamed answer depends on.
@available(macOS 27.0, *)
struct ProseText: View {
  let text: String
  let size: Double

  var body: some View {
    VStack(alignment: .leading, spacing: 3) {
      ForEach(Array(text.components(separatedBy: "\n").enumerated()), id: \.offset) { _, raw in
        rendered(raw)
      }
    }
  }

  @ViewBuilder private func rendered(_ raw: String) -> some View {
    let trimmed = raw.trimmingCharacters(in: .whitespaces)
    if trimmed.isEmpty {
      Spacer().frame(height: 4)
    } else if let heading = heading(trimmed) {
      Text(inline(heading.text))
        .font(.system(size: size + (heading.level == 1 ? 5 : 3), weight: .semibold))
        .padding(.top, 2)
    } else if let bullet = bullet(trimmed) {
      HStack(alignment: .firstTextBaseline, spacing: 6) {
        Text(bullet.marker)
          .font(.system(size: size, design: .monospaced))
          .foregroundStyle(.tertiary)
        Text(inline(bullet.text))
          .font(.system(size: size))
      }
      .padding(.leading, CGFloat(indent(raw)) * 12)
    } else {
      Text(inline(raw))
        .font(.system(size: size))
    }
  }

  private func inline(_ source: String) -> AttributedString {
    (try? AttributedString(
      markdown: source,
      options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
      ?? AttributedString(source)
  }

  private func heading(_ line: String) -> (level: Int, text: String)? {
    let hashes = line.prefix(while: { $0 == "#" }).count
    guard hashes > 0, hashes <= 6, line.dropFirst(hashes).hasPrefix(" ") else { return nil }
    return (hashes, String(line.dropFirst(hashes + 1)))
  }

  private func bullet(_ line: String) -> (marker: String, text: String)? {
    for marker in ["- ", "* ", "+ "] where line.hasPrefix(marker) {
      return ("•", String(line.dropFirst(marker.count)))
    }
    // An ordered item keeps its own number, since the model chose it.
    let digits = line.prefix(while: \.isNumber)
    if !digits.isEmpty, line.dropFirst(digits.count).hasPrefix(". ") {
      return ("\(digits).", String(line.dropFirst(digits.count + 2)))
    }
    return nil
  }

  private func indent(_ raw: String) -> Int {
    (raw.prefix(while: { $0 == " " }).count) / 2
  }
}

@available(macOS 27.0, *)
struct CodeBlock: View {
  let language: String?
  let code: String
  let closed: Bool
  let mono: Font
  let size: Double

  @Environment(\.colorScheme) private var scheme

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      HStack(spacing: 6) {
        Text(language ?? "text")
          .font(.system(size: 9, weight: .medium, design: .monospaced))
          .foregroundStyle(.secondary)
        if !closed {
          // The fence is still open, so the block says so rather than looking finished.
          AnimatedDots(size: 3, tint: .secondary)
        }
        Spacer()
        Button {
          NSPasteboard.general.clearContents()
          NSPasteboard.general.setString(code, forType: .string)
        } label: {
          Image(systemName: "doc.on.doc")
            .font(.system(size: 9))
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .help("Copy")
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 5)
      .background(.white.opacity(0.05))

      ScrollView(.horizontal, showsIndicators: false) {
        text
          .textSelection(.enabled)
          .padding(.horizontal, 10)
          .padding(.vertical, 8)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
    }
    .background(.black.opacity(scheme == .dark ? 0.28 : 0.05))
    .clipShape(.rect(cornerRadius: 8))
    .overlay {
      RoundedRectangle(cornerRadius: 8)
        .strokeBorder(.white.opacity(0.08), lineWidth: 0.5)
    }
  }

  @ViewBuilder private var text: some View {
    if let highlighted = SyntaxHighlighter.shared.highlight(
      code, language: language, dark: scheme == .dark,
      font: NSFont.monospacedSystemFont(ofSize: size, weight: .regular))
    {
      Text(highlighted)
    } else {
      Text(code)
        .font(mono)
        .foregroundStyle(.primary)
    }
  }
}
