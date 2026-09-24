// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DeepSeek-V4.1's prompts, against the golden outputs its release ships with its encoder.

import Foundation
import Jinja
import Testing

@testable import IshizukiKit

/// The cases are `encoding/tests/test_input_*.json` and their outputs from the release. Two of
/// its five are left out: one hangs tools on a system message part way through, and one uses
/// the internal classification tasks, neither of which a request to this server can say.
@Suite("DeepSeek-V4.1 chat format")
struct DeepSeekChatFormatTests {
  private var folder: URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .appending(path: "Fixtures/deepseek-v41/encoding")
  }

  private func messages(_ raw: [[String: Any]]) -> [ChatMessage] {
    raw.map { entry in
      let role = entry["role"] as? String ?? "user"
      let calls = (entry["tool_calls"] as? [[String: Any]] ?? []).map { call -> ToolCall in
        let function = call["function"] as? [String: Any] ?? [:]
        return ToolCall(
          name: function["name"] as? String ?? "",
          argumentsJSON: function["arguments"] as? String ?? "{}")
      }
      let content: ChatMessage.Content
      if let blocks = entry["content"] as? [[String: Any]] {
        content = .parts(
          blocks.map { block in
            block["type"] as? String == "text" ? .text(block["text"] as? String ?? "") : .image
          })
      } else {
        content = .text(entry["content"] as? String ?? "")
      }
      return ChatMessage(
        role: role, content: content, toolCalls: calls,
        reasoning: entry["reasoning_content"] as? String)
    }
  }

  private func check(_ number: Int) throws {
    let input = try Data(contentsOf: folder.appending(path: "test_input_\(number).json"))
    let want = try String(
      contentsOf: folder.appending(path: "test_output_\(number).txt"), encoding: .utf8)
    let object = try JSONSerialization.jsonObject(with: input)
    let raw = (object as? [String: Any])?["messages"] as? [[String: Any]]
      ?? (object as? [[String: Any]]) ?? []
    let settings = object as? [String: Any] ?? [:]
    let thinking = settings["thinking_mode"] as? String == "thinking"
    let effort: ReasoningEffort? = settings["reasoning_effort"] as? String == "max" ? .xhigh : nil

    var ordered: [Value]?
    if case .object(let root) = OrderedJSON.object(String(decoding: input, as: UTF8.self)),
      case .array(let tools)? = root["tools"]
    {
      ordered = tools
    }
    let got = DeepSeekChatFormat.render(
      messages: messages(raw), addGenerationPrompt: true, thinking: thinking, effort: effort,
      orderedTools: ordered)
    #expect(got == want, "case \(number) rendered differently:\n\(got)")
  }

  @Test("renders tools, calls and results in thinking mode") func toolCase() throws { try check(1) }
  @Test("renders a chat that drops past reasoning") func chatCase() throws { try check(2) }
  @Test("renders pictures and the reasoning budget") func visionCase() throws { try check(5) }

  @Test("writes an agent tool's schema in the order its text has it")
  func keepsAgentSchemaOrder() {
    let tool = ToolSchema(
      name: "search", description: "Finds things.",
      parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"},"#
        + #""limit":{"type":"integer"}},"required":["query"]}"#)
    let prompt = DeepSeekChatFormat.render(
      messages: [.system("Be brief."), .user("hi")], thinking: false,
      orderedTools: [tool.orderedValue])
    let schema =
      #"{"name": "search", "description": "Finds things.", "parameters": {"type": "object", "#
      + #""properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "#
      + #""required": ["query"]}}"#
    #expect(prompt.contains(schema), "the schema lost its order:\n\(prompt)")
  }

  @Test("reads a DSML call back into the arguments it encodes")
  func roundTrips() throws {
    let call = ToolCall(name: "lookup", argumentsJSON: #"{"query":"value","limit":2}"#)
    let block = DeepSeekChatFormat.callsBlock([call])
    let completion = "  reason  </think>summary\n\n" + block + "<｜end▁of▁sentence｜>"
    let parsed = ToolCallParser.parse("<think>" + completion)
    #expect(parsed.reasoning == "reason")
    #expect(parsed.content == "summary")
    #expect(parsed.toolCalls.count == 1)
    #expect(parsed.toolCalls.first?.name == "lookup")
    #expect(parsed.toolCalls.first?.argumentsJSON == #"{"query": "value", "limit": 2}"#)
  }

  @Test("stops short of the assistant header when no answer is asked for")
  func leavesTheTurnOpen() throws {
    let prompt = DeepSeekChatFormat.render(
      messages: [.user("hi")], addGenerationPrompt: false, thinking: false)
    #expect(prompt == "<｜begin▁of▁sentence｜><｜User｜>hi")
    let asked = DeepSeekChatFormat.render(messages: [.user("hi")], thinking: false)
    #expect(asked == "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>")
  }
}
