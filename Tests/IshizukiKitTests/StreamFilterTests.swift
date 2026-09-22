// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

@Suite("Stream filter")
struct StreamFilterTests {
  private struct Drain {
    var reasoning = ""
    var content = ""
    var tool = ""

    mutating func take(_ output: StreamFilter.Output) {
      if let thought = output.reasoning { reasoning += thought }
      if let visible = output.content { content += visible }
      if let command = output.toolText { tool += command }
    }
  }

  /// Fragment by fragment, the way the detokenizer hands them over.
  private func run(thinking: Bool, _ fragments: [String]) -> Drain {
    var filter = StreamFilter(thinking: thinking)
    var drain = Drain()
    for fragment in fragments { drain.take(filter.push(fragment)) }
    drain.take(filter.flush())
    return drain
  }

  @Test("thinking arrives before the answer, and the tag itself never does")
  func splitsTheTwo() {
    let drain = run(
      thinking: true,
      ["I should ", "look at the file first.", "</think>", "Reading it now.", " Done."])
    #expect(drain.reasoning == "I should look at the file first.")
    #expect(drain.content == "Reading it now. Done.")
    #expect(!drain.reasoning.contains("think"))
  }

  @Test("thinking is handed over as it arrives rather than held to the end")
  func reasoningStreams() {
    var filter = StreamFilter(thinking: true)
    var seen: [String] = []
    // Long enough to clear the guard window, which is what holds a split tag back.
    for fragment in ["Weighing the options here, ", "and the tradeoffs involved. "] {
      if let thought = filter.push(fragment).reasoning { seen.append(thought) }
    }
    #expect(!seen.isEmpty)
    #expect(seen.joined().hasPrefix("Weighing the options"))
  }

  @Test("a closing tag split across fragments is not mistaken for thinking")
  func neverLeaksASplitTag() {
    let drain = run(
      thinking: true,
      ["Thinking about this carefully now.", "</thi", "nk>", "The answer follows here."])
    #expect(drain.reasoning == "Thinking about this carefully now.")
    #expect(drain.content == "The answer follows here.")
  }

  @Test("a tool call ends the visible text and nothing after it is shown")
  func stopsAtAToolCall() {
    let drain = run(
      thinking: false,
      ["Let me read that file.", "<tool_call><function=read>", "trailing noise"])
    #expect(drain.content == "Let me read that file.")
    #expect(drain.reasoning.isEmpty)
  }

  @Test("thinking that never closes is still thinking, not an answer")
  func unterminatedStaysReasoning() {
    let drain = run(thinking: true, ["Still working it out and never finishing"])
    #expect(drain.content.isEmpty)
    #expect(drain.reasoning == "Still working it out and never finishing")
  }

  /// The gap between a thought ending and a call landing used to be blank, because the call's
  /// own text was thrown away rather than handed over.
  @Test("a tool call's text arrives as it is written")
  func toolTextStreams() {
    let drain = run(
      thinking: true,
      [
        "Checking the directory first.", "</think>", "<tool_call>",
        "<function=shell><parameter=command>ls -la", " Documents</parameter>",
        "</function></tool_call>",
      ])
    #expect(drain.reasoning == "Checking the directory first.")
    #expect(drain.tool.contains("ls -la Documents"))
    #expect(drain.content.isEmpty)
  }

  @Test("with thinking off, every fragment is the answer")
  func plainContent() {
    let drain = run(thinking: false, ["Hello ", "there, this is the answer."])
    #expect(drain.content == "Hello there, this is the answer.")
    #expect(drain.reasoning.isEmpty)
  }

  @Test("a thought the model opens itself is still split off, even across fragments")
  func modelOpensItsOwnThought() {
    let drain = run(
      thinking: false,
      ["\n<th", "ink>\nWeighing it up first.", "</think>", "Here is the answer."])
    #expect(
      drain.reasoning.trimmingCharacters(in: .whitespacesAndNewlines) == "Weighing it up first.")
    #expect(drain.content == "Here is the answer.")
  }

  @Test("a short answer that is not a thought is not held back")
  func shortAnswerSurvives() {
    let drain = run(thinking: false, ["<", "b>Hi</b>"])
    #expect(drain.content == "<b>Hi</b>")
    #expect(drain.reasoning.isEmpty)
  }
}
