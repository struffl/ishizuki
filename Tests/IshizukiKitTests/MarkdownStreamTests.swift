// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

@Suite("Markdown stream")
struct MarkdownStreamTests {
  @Test("a fence that has only just opened is already a code block")
  func opensEarly() {
    let blocks = MarkdownStream.blocks(in: "Here:\n```swift\n")
    #expect(blocks == [.prose("Here:"), .code(language: "swift", body: "", closed: false)])
  }

  @Test("an open block grows as its lines land")
  func growsWhileOpen() {
    let blocks = MarkdownStream.blocks(in: "```swift\nlet a = 1\nlet b = 2")
    #expect(
      blocks == [.code(language: "swift", body: "let a = 1\nlet b = 2", closed: false)])
  }

  @Test("closing the fence closes the block and prose resumes after it")
  func closes() {
    let blocks = MarkdownStream.blocks(in: "```py\nx = 1\n```\nDone.")
    #expect(
      blocks == [
        .code(language: "py", body: "x = 1", closed: true),
        .prose("Done."),
      ])
  }

  /// Rendering half a fence as a code block would flicker, so it stays prose for the one
  /// fragment it takes for the newline to arrive.
  @Test("half an opening fence is not yet a fence")
  func partialFenceWaits() {
    #expect(MarkdownStream.blocks(in: "text\n``") == [.prose("text\n``")])
    #expect(MarkdownStream.blocks(in: "text\n```swi") == [.prose("text\n```swi")])
  }

  @Test("a fence with no language is still a block")
  func noLanguage() {
    let blocks = MarkdownStream.blocks(in: "```\nplain\n```")
    #expect(blocks == [.code(language: nil, body: "plain", closed: true)])
  }

  @Test("tildes fence too, and backticks inside them do not close them")
  func tildeFence() {
    let blocks = MarkdownStream.blocks(in: "~~~ruby\nputs `ls`\n~~~")
    #expect(blocks == [.code(language: "ruby", body: "puts `ls`", closed: true)])
  }

  @Test("a longer fence is needed to close a longer one")
  func fenceLength() {
    let blocks = MarkdownStream.blocks(in: "````md\n```\ninner\n```\n````")
    #expect(blocks == [.code(language: "md", body: "```\ninner\n```", closed: true)])
  }

  @Test("blank lines inside a block are kept, and around prose are not")
  func whitespace() {
    let blocks = MarkdownStream.blocks(in: "\n\nhi\n\n```\na\n\nb\n```\n\n")
    #expect(
      blocks == [
        .prose("hi"),
        .code(language: nil, body: "a\n\nb", closed: true),
      ])
  }

  @Test("an indented fence opens, but a deeply indented one is code by indentation")
  func indentation() {
    #expect(
      MarkdownStream.blocks(in: "   ```swift\nx\n   ```")
        == [.code(language: "swift", body: "x", closed: true)])
    #expect(MarkdownStream.blocks(in: "    ```swift\nx") == [.prose("    ```swift\nx")])
  }

  @Test("prose with no fence at all is one block")
  func plainProse() {
    #expect(MarkdownStream.blocks(in: "just words") == [.prose("just words")])
    #expect(MarkdownStream.blocks(in: "") == [])
  }

  /// The parser is called again on every fragment, so the same text must give the same blocks
  /// whether it arrived at once or a character at a time.
  @Test("reading a document one character at a time ends where reading it whole does")
  func incrementalMatchesWhole() {
    let document = "Intro\n\n```swift\nfunc f() {}\n```\n\nOutro\n\n```\ntail"
    var partial = ""
    for character in document {
      partial.append(character)
      _ = MarkdownStream.blocks(in: partial)
    }
    #expect(MarkdownStream.blocks(in: partial) == MarkdownStream.blocks(in: document))
    #expect(
      MarkdownStream.blocks(in: document) == [
        .prose("Intro"),
        .code(language: "swift", body: "func f() {}", closed: true),
        .prose("Outro"),
        .code(language: nil, body: "tail", closed: false),
      ])
  }
}
