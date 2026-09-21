// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

@Suite("Tool call parser")
struct ToolCallParserTests {
  private let call =
    "<tool_call><function=read><parameter=path>\"a.swift\"</parameter></function></tool_call>"

  @Test("a closed thought is split from the answer")
  func closedThought() {
    let parsed = ToolCallParser.parse("<think>weighing it up</think>Here is the answer.")
    #expect(parsed.reasoning == "weighing it up")
    #expect(parsed.content == "Here is the answer.")
  }

  /// The case that put a whole chain of reasoning in the reply: generation stopped before the
  /// closing tag, so nothing split the two and the answer became fifteen kilobytes of thinking.
  @Test("a thought that was never closed is still a thought, not the answer")
  func unterminatedThought() {
    let parsed = ToolCallParser.parse("<think>still working it out, and then the tokens ran out")
    #expect(parsed.reasoning == "still working it out, and then the tokens ran out")
    #expect(parsed.content.isEmpty)
    #expect(!parsed.content.contains("<think>"))
  }

  @Test("content before an unclosed thought is kept as the answer")
  func contentBeforeUnterminatedThought() {
    let parsed = ToolCallParser.parse("Short answer.\n<think>then it kept going")
    #expect(parsed.content == "Short answer.")
    #expect(parsed.reasoning == "then it kept going")
  }

  @Test("a tool call after an unclosed thought is still made")
  func callAfterUnterminatedThought() {
    let parsed = ToolCallParser.parse("<think>never closed" + call)
    #expect(parsed.toolCalls.count == 1)
    #expect(parsed.toolCalls.first?.name == "read")
    #expect(parsed.reasoning == "never closed")
    #expect(parsed.content.isEmpty)
  }

  @Test("a tool call after a closed thought is unaffected")
  func callAfterClosedThought() {
    let parsed = ToolCallParser.parse("<think>ok</think>doing it" + call)
    #expect(parsed.toolCalls.first?.name == "read")
    #expect(parsed.content == "doing it")
  }

  @Test("no thought at all leaves the answer alone")
  func noThought() {
    let parsed = ToolCallParser.parse("Just the answer.")
    #expect(parsed.reasoning == nil)
    #expect(parsed.content == "Just the answer.")
  }
}
