// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Markdown as it arrives: prose with its inline marks and a fading edge where it is still
// being written, and fenced code shown from the moment the fence opens, highlighted once it
// closes.

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

  func highlight(_ code: String, language: String?, dark: Bool, font: PlatformFont)
    -> AttributedString?
  {
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

@available(macOS 27.0, iOS 27.0, *)
struct MarkdownText: View {
  let text: String
  let mono: Font
  let size: Double
  /// How many characters at the very end are still arriving, drawn as a fading edge so prose
  /// grows into the bubble rather than snapping into it. Only the last block can have one.
  var fadeTail = 0

  var body: some View {
    let blocks = MarkdownStream.blocks(in: text)
    VStack(alignment: .leading, spacing: 9) {
      ForEach(Array(blocks.enumerated()), id: \.offset) { index, block in
        let tail = index == blocks.count - 1 ? fadeTail : 0
        switch block {
        case .prose(let prose):
          ProseText(text: prose, size: size, fadeTail: tail)
        case .code(let language, let body, let closed):
          CodeBlock(language: language, code: body, closed: closed, mono: mono, size: size)
            .transition(.opacity)
        }
      }
    }
  }
}

/// The ramp that makes a streamed edge soft: the newest characters come in from nothing rather
/// than appearing whole.
@MainActor
enum StreamFade {
  static let window = 18

  static func ramped(_ text: String, tail: Int) -> AttributedString {
    let characters = Array(text)
    let faded = min(tail, characters.count)
    guard faded > 0 else { return AttributedString(text) }

    var out = AttributedString(String(characters.prefix(characters.count - faded)))
    for (step, character) in characters.suffix(faded).enumerated() {
      var piece = AttributedString(String(character))
      // Newest last, so the ramp runs from almost solid down to almost nothing.
      let through = Double(step + 1) / Double(faded)
      piece.foregroundColor = Color.primary.opacity(max(0.06, 1 - through * 0.94))
      out += piece
    }
    return out
  }
}

/// Headings and bullets are drawn here rather than handed to the markdown parser, which would
/// collapse the whitespace a streamed answer depends on.
extension EnvironmentValues {
  /// The face prose is set in.
  @Entry var proseDesign: Font.Design = .serif
}

@available(macOS 27.0, iOS 27.0, *)
struct ProseText: View {
  let text: String
  let size: Double
  var fadeTail = 0
  @Environment(\.proseDesign) private var face

  var body: some View {
    VStack(alignment: .leading, spacing: face == .serif ? 6 : 3) {
      ForEach(Array(fades.enumerated()), id: \.offset) { _, line in
        rendered(line.text, fade: line.fade)
      }
    }
  }

  /// Each line with the number of its own trailing characters that fall inside the fading
  /// edge, counted back from the end of the block.
  private var fades: [(text: String, fade: Int)] {
    let lines = text.components(separatedBy: "\n")
    guard fadeTail > 0 else { return lines.map { ($0, 0) } }
    var remaining = fadeTail
    var out: [(String, Int)] = []
    for line in lines.reversed() {
      let share = min(remaining, line.count)
      out.append((line, share))
      remaining = max(0, remaining - line.count - 1)
    }
    return out.reversed().map { (text: $0.0, fade: $0.1) }
  }

  @ViewBuilder private func rendered(_ raw: String, fade: Int) -> some View {
    let trimmed = raw.trimmingCharacters(in: .whitespaces)
    if trimmed.isEmpty {
      // Not a Spacer: a flexible spacer in a VStack absorbs whatever slack an ancestor hands
      // down, so a blank line between paragraphs could stretch into a void of its own.
      Color.clear.frame(height: 4)
    } else if let heading = heading(trimmed) {
      Text(styled(heading.text, fade: fade))
        .font(.system(size: size + (heading.level == 1 ? 5 : 3), weight: .semibold, design: face))
        .padding(.top, 2)
    } else if let bullet = bullet(trimmed) {
      HStack(alignment: .firstTextBaseline, spacing: 7) {
        Text(bullet.marker)
          .font(.system(size: size, design: .monospaced))
          .fontDesign(.monospaced)
          .foregroundStyle(.tertiary)
        Text(styled(bullet.text, fade: fade))
          .font(.system(size: size, design: face))
          .lineSpacing(face == .serif ? 3 : 0)
      }
      .padding(.leading, CGFloat(indent(raw)) * 12)
    } else {
      Text(styled(raw, fade: fade))
        .font(.system(size: size, design: face))
        .lineSpacing(face == .serif ? 3 : 0)
    }
  }

  /// Marks are read from the settled part of a line; the fading edge is left plain, since half
  /// an emphasis pair is not emphasis yet and would pop as the other half landed.
  private func styled(_ source: String, fade: Int) -> AttributedString {
    let faded = min(fade, source.count)
    guard faded > 0 else { return inline(source) }
    var out = inline(String(source.dropLast(faded)))
    out += StreamFade.ramped(String(source.suffix(faded)), tail: faded)
    return out
  }

  private func inline(_ source: String) -> AttributedString {
    var attributed =
      (try? AttributedString(
        markdown: source,
        options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
      ?? AttributedString(source)

    // A backtick span is given the chip it reads as everywhere else: the parser marks it, but
    // draws it in the body face, which leaves a file name indistinguishable from prose.
    let coded = attributed.runs.compactMap { run in
      run.inlinePresentationIntent?.contains(.code) == true ? run.range : nil
    }
    for range in coded {
      attributed[range].font = .system(size: size * 0.94, design: .monospaced)
      attributed[range].backgroundColor = Color.ink.opacity(0.08)
    }
    return attributed
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

@available(macOS 27.0, iOS 27.0, *)
struct CodeBlock: View {
  let language: String?
  let code: String
  let closed: Bool
  let mono: Font
  let size: Double

  @Environment(\.colorScheme) private var scheme

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      HStack(spacing: 7) {
        Text(language ?? "text")
          .font(.system(.subheadline, design: .monospaced, weight: .medium))
          .fontDesign(.monospaced)
          .foregroundStyle(.secondary)
        if !closed {
          // The fence is still open, so the block says so rather than looking finished.
          AnimatedDots(size: 3, tint: .secondary)
        }
        Spacer()
        Button {
          Clipboard.copy(code)
        } label: {
          Image(systemName: "doc.on.doc")
            .font(.subheadline)
            .hitTarget()
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .accessibilityLabel("Copy this code")
        .help("Copy")
      }
      .padding(.horizontal, 11)
      .padding(.vertical, 6)
      .background(.quaternary)

      ScrollView(.horizontal, showsIndicators: false) {
        text
          .textSelection(.enabled)
          .padding(.horizontal, 11)
          .padding(.vertical, 9)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
    }
    .background(.quinary)
    .clipShape(.rect(cornerRadius: 9))
    .overlay {
      RoundedRectangle(cornerRadius: 9)
        .strokeBorder(Color.hairline, lineWidth: 0.5)
    }
  }

  @ViewBuilder private var text: some View {
    // An open fence is being written into, and highlighting it means a run through
    // JavaScriptCore for every token that lands, on the thread trying to draw them. It is
    // highlighted the moment the fence closes instead.
    if closed,
      let highlighted = SyntaxHighlighter.shared.highlight(
        code, language: language, dark: scheme == .dark,
        font: .mono(size))
    {
      Text(highlighted)
    } else {
      Text(code)
        .font(mono)
        .fontDesign(.monospaced)
        .foregroundStyle(.primary)
    }
  }
}
