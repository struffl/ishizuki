// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import IshizukiKit

struct MarkdownStream {
  private enum Kind {
    case prose
    case list(String)
    case quote
    case held
  }

  private let styled = Style.depth != .none
  private var line: [Character] = []
  private var emitted = 0
  private var kind: Kind?
  private var language: String?
  private var comment = false

  mutating func push(_ fragment: String) -> String {
    guard styled else { return fragment }
    var out = ""
    for character in fragment {
      if character.isNewline {
        out += flush() + "\n"
        continue
      }
      line.append(character)
      out += stream()
    }
    return out
  }

  mutating func finish() -> String {
    guard styled, !line.isEmpty || language != nil else { return "" }
    var out = line.isEmpty ? "" : flush()
    if language != nil {
      language = nil
      out += Style.faint("  ╰") + "\n"
    }
    return out
  }

  private mutating func stream() -> String {
    guard language == nil else { return "" }
    if let kind {
      switch kind {
      case .prose, .list, .quote: return inline(force: false)
      case .held: return ""
      }
    }
    guard line.count >= 4 else { return "" }
    let prefix = classify()
    return prefix + stream()
  }

  private mutating func classify() -> String {
    let text = String(line)

    if text.hasPrefix("```") || text.hasPrefix("#") || text.hasPrefix("---")
      || text.hasPrefix("***") || text.hasPrefix("===")
    {
      kind = .held
      return ""
    }

    if let marker = listMarker(text) {
      kind = .list(marker)
      emitted = marker.count
      return Style.accent(marker.hasSuffix(". ") ? marker : "• ")
    }

    if text.hasPrefix("> ") || text == ">" {
      kind = .quote
      emitted = min(2, line.count)
      return Style.faint("▌ ")
    }

    kind = .prose
    return ""
  }

  private func listMarker(_ text: String) -> String? {
    for bullet in ["- ", "* ", "+ "] where text.hasPrefix(bullet) {
      return bullet
    }
    let digits = text.prefix { $0.isNumber }
    if !digits.isEmpty, text.dropFirst(digits.count).hasPrefix(". ") {
      return String(text.prefix(digits.count + 2))
    }
    return nil
  }

  private mutating func flush() -> String {
    defer {
      line = []
      emitted = 0
      kind = nil
    }
    let text = String(line)

    if let language {
      if text.hasPrefix("```") {
        self.language = nil
        comment = false
        return Style.faint("  ╰")
      }
      return Style.faint("  │ ") + Syntax.highlight(text, language: language, comment: &comment)
    }

    if text.hasPrefix("```") {
      let named = text.dropFirst(3).trimmingCharacters(in: .whitespaces)
      language = named.isEmpty ? "text" : named.lowercased()
      comment = false
      return Style.faint("  ╭ ") + Style.muted(language ?? "text")
    }

    if text.hasPrefix("#") {
      let hashes = text.prefix { $0 == "#" }.count
      let body = text.dropFirst(hashes).trimmingCharacters(in: .whitespaces)
      return Style.paint(body, hashes <= 2 ? .accentBright : .text, bold: true)
    }

    let bare = text.trimmingCharacters(in: .whitespaces)
    if bare.count >= 3, bare.allSatisfy({ $0 == "-" || $0 == "*" || $0 == "=" }) {
      return Style.rule
    }

    let prefix = kind == nil ? classify() : ""
    return prefix + inline(force: true)
  }

  private mutating func inline(force: Bool) -> String {
    var out = ""
    var plain = ""
    var index = emitted

    func drain() {
      guard !plain.isEmpty else { return }
      out += plain
      plain = ""
    }

    while index < line.count {
      let character = line[index]

      if character == "`", let close = find("`", from: index + 1) {
        drain()
        out += Style.good(String(line[(index + 1)..<close]))
        index = close + 1
        continue
      }

      if character == "*", index + 1 < line.count, line[index + 1] == "*",
        let close = find("**", from: index + 2)
      {
        drain()
        out += Style.paint(String(line[(index + 2)..<close]), .text, bold: true)
        index = close + 2
        continue
      }

      if character == "*" || character == "_", let close = find(String(character), from: index + 1),
        close > index + 1
      {
        drain()
        out += Style.paint(String(line[(index + 1)..<close]), .text, italic: true)
        index = close + 1
        continue
      }

      if character == "[", let label = find("](", from: index + 1),
        let close = find(")", from: label + 2)
      {
        drain()
        out += Style.accent(String(line[(index + 1)..<label]))
        out += Style.faint(" (" + String(line[(label + 2)..<close]) + ")")
        index = close + 1
        continue
      }

      if "`*_[".contains(character), !force {
        break
      }

      plain.append(character)
      index += 1
    }

    drain()
    emitted = index
    return out
  }

  private func find(_ needle: String, from start: Int) -> Int? {
    let pattern = Array(needle)
    guard start >= 0, pattern.count > 0, line.count >= pattern.count else { return nil }
    var index = start
    while index + pattern.count <= line.count {
      if Array(line[index..<(index + pattern.count)]) == pattern { return index }
      index += 1
    }
    return nil
  }
}

enum Syntax {
  static func highlight(_ line: String, language: String, comment: inout Bool) -> String {
    let characters = Array(line)
    let words = keywords(for: language)
    let lineComment = lineCommentMarker(for: language)
    var out = ""
    var plain = ""
    var index = 0

    func drain() {
      guard !plain.isEmpty else { return }
      out += Style.paint(plain, .text)
      plain = ""
    }

    func matches(_ marker: String, at position: Int) -> Bool {
      let pattern = Array(marker)
      guard position + pattern.count <= characters.count else { return false }
      return Array(characters[position..<(position + pattern.count)]) == pattern
    }

    while index < characters.count {
      if comment {
        guard let close = locate("*/", in: characters, from: index) else {
          out += Style.faint(String(characters[index...]))
          return out
        }
        out += Style.faint(String(characters[index..<(close + 2)]))
        index = close + 2
        comment = false
        continue
      }

      let character = characters[index]

      if matches("/*", at: index), blocksComments(language) {
        drain()
        comment = true
        continue
      }

      if let lineComment, matches(lineComment, at: index) {
        drain()
        out += Style.faint(String(characters[index...]))
        return out
      }

      if character == "\"" || character == "'" || character == "`" {
        drain()
        var cursor = index + 1
        while cursor < characters.count {
          if characters[cursor] == "\\" {
            cursor += 2
            continue
          }
          if characters[cursor] == character { break }
          cursor += 1
        }
        let end = min(cursor, characters.count - 1)
        out += Style.good(String(characters[index...end]))
        index = end + 1
        continue
      }

      if character.isNumber, index == 0 || !isWord(characters[index - 1]) {
        drain()
        var cursor = index
        while cursor < characters.count,
          characters[cursor].isHexDigit || characters[cursor] == "." || characters[cursor] == "x"
            || characters[cursor] == "_"
        {
          cursor += 1
        }
        out += Style.warn(String(characters[index..<cursor]))
        index = cursor
        continue
      }

      if isWord(character), !character.isNumber {
        var cursor = index
        while cursor < characters.count, isWord(characters[cursor]) { cursor += 1 }
        let word = String(characters[index..<cursor])
        if words.contains(word) {
          drain()
          out += Style.accent(word)
        } else if let first = word.first, first.isUppercase {
          drain()
          out += Style.paint(word, .accentBright)
        } else {
          plain += word
        }
        index = cursor
        continue
      }

      plain.append(character)
      index += 1
    }

    drain()
    return out
  }

  private static func isWord(_ character: Character) -> Bool {
    character.isLetter || character.isNumber || character == "_"
  }

  private static func locate(_ needle: String, in characters: [Character], from start: Int) -> Int?
  {
    let pattern = Array(needle)
    var index = start
    while index + pattern.count <= characters.count {
      if Array(characters[index..<(index + pattern.count)]) == pattern { return index }
      index += 1
    }
    return nil
  }

  private static func blocksComments(_ language: String) -> Bool {
    !["python", "py", "bash", "sh", "zsh", "shell", "ruby", "yaml", "toml", "text"].contains(
      language)
  }

  private static func lineCommentMarker(for language: String) -> String? {
    switch language {
    case "python", "py", "bash", "sh", "zsh", "shell", "ruby", "yaml", "toml", "just", "make":
      return "#"
    case "sql", "lua", "haskell":
      return "--"
    case "json", "text":
      return nil
    default:
      return "//"
    }
  }

  private static let common: Set<String> = [
    "if", "else", "for", "while", "return", "break", "continue", "true", "false", "null", "class",
    "import", "new", "try", "catch", "throw", "switch", "case", "default", "in", "not", "and", "or",
  ]

  private static func keywords(for language: String) -> Set<String> {
    switch language {
    case "swift":
      return common.union([
        "func", "let", "var", "struct", "enum", "protocol", "extension", "guard", "defer", "self",
        "init", "mutating", "static", "public", "private", "internal", "throws", "async", "await",
        "some", "any", "where", "nil", "inout", "typealias", "actor", "final",
      ])
    case "python", "py":
      return common.union([
        "def", "lambda", "elif", "None", "True", "False", "with", "as", "yield", "async", "await",
        "pass", "raise", "from", "global", "is", "del", "assert",
      ])
    case "rust":
      return common.union([
        "fn", "let", "mut", "impl", "trait", "struct", "enum", "pub", "use", "mod", "match", "move",
        "ref", "unsafe", "dyn", "crate", "where", "Some", "None", "Ok", "Err",
      ])
    case "go":
      return common.union([
        "func", "var", "const", "type", "struct", "interface", "package", "defer", "go", "chan",
        "map", "range", "select", "nil", "fallthrough",
      ])
    case "javascript", "js", "typescript", "ts", "tsx", "jsx":
      return common.union([
        "function", "const", "let", "var", "=>", "async", "await", "export", "interface", "type",
        "extends", "implements", "undefined", "this", "of", "yield", "static",
      ])
    case "bash", "sh", "zsh", "shell":
      return common.union([
        "fi", "then", "elif", "do", "done", "esac", "function", "local", "export", "source", "echo",
        "set", "unset", "trap",
      ])
    case "json":
      return ["true", "false", "null"]
    default:
      return common.union(["func", "fn", "def", "let", "const", "var", "struct", "enum", "pub"])
    }
  }
}
