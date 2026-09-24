// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DeepSeek-V4.1's prompt format, which the release ships as Python rather than a template.

import Foundation
import Jinja
import OrderedCollections

/// Renders a conversation the way `encoding/encoding.py` in the release does.
///
/// Tool results have no role of their own: they fold into the user turn that carries them.
/// In thinking mode the reasoning of every turn before the latest user message is dropped,
/// unless tools are in play, and the assistant header ends in `<think>` for the turn being
/// answered and `</think>` everywhere else. Tool calls are DSML blocks with each argument
/// marked as a string or as JSON.
public enum DeepSeekChatFormat {
  static let bos = "<｜begin▁of▁sentence｜>"
  static let eos = "<｜end▁of▁sentence｜>"
  static let system = "<｜System｜>"
  static let user = "<｜User｜>"
  static let assistant = "<｜Assistant｜>"
  static let reminder = "<｜latest_reminder｜>"
  static let image = "<｜deepseek_image｜>"
  static let open = "<think>"
  static let close = "</think>"
  static let dsml = "｜DSML｜"
  static let callsOpen = "<\(dsml) calls>"
  static let callsClose = "</\(dsml) calls>"

  /// The reasoning budget a thinking prompt opens with, on DeepSeek's scale of 1 to 100.
  public static func budget(_ effort: ReasoningEffort?) -> Int {
    switch effort {
    case .some(.low): 50
    case .some(.medium): 60
    case .some(.xhigh): 100
    case .some(.none), .some(.high), nil: 75
    }
  }

  private struct Turn {
    var role: String
    var text: String
    var blocks: [String]? = nil
    var reasoning: String? = nil
    var calls: [ToolCall] = []
    var tools: [String]? = nil
  }

  /// `orderedTools` is the request's own tool list, keys in the order the client wrote them;
  /// without it the schemas are written in a fixed order of their own.
  public static func render(
    messages: [ChatMessage], addGenerationPrompt: Bool = true, thinking: Bool,
    effort: ReasoningEffort? = nil, tools: [[String: Any]]? = nil, orderedTools: [Value]? = nil
  ) -> String {
    var turns: [Turn] = []
    for message in messages {
      switch message.role {
      case "tool":
        let block = "<tool_result>\(message.plainText)</tool_result>"
        if turns.last?.role == "user" {
          turns[turns.count - 1].blocks!.append(block)
        } else {
          turns.append(Turn(role: "user", text: "", blocks: [block]))
        }
      case "user":
        let blocks = userBlocks(message)
        if turns.last?.role == "user" {
          turns[turns.count - 1].blocks!.append(contentsOf: blocks)
        } else {
          turns.append(Turn(role: "user", text: "", blocks: blocks))
        }
      case "assistant":
        turns.append(
          Turn(
            role: "assistant", text: message.plainText, reasoning: message.reasoning,
            calls: message.toolCalls))
      default:
        turns.append(Turn(role: message.role, text: message.plainText))
      }
    }
    let schemas =
      orderedTools.map { $0.map(schema) } ?? (tools ?? []).map { tool in
        PythonJSON.encode((tool["function"] as? [String: Any]) ?? tool)
      }
    if !schemas.isEmpty {
      if turns.first?.role == "system" {
        turns[0].tools = schemas
      } else {
        turns.insert(Turn(role: "system", text: "", tools: schemas), at: 0)
      }
    }

    let dropThinking = schemas.isEmpty
    let lastUserIndex =
      turns.indices.last { index in
        turns[index].role == "user" || (turns[index].role == "system" && index > 0)
      } ?? -1

    var prompt = bos
    for index in turns.indices {
      let turn = turns[index]
      var piece = ""
      let effortLine =
        index == 0 && thinking
        ? "Reasoning Effort: \(budget(effort)) "
          + "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        : ""
      if index == 0, !effortLine.isEmpty || turn.role == "system" { piece += system }
      piece += effortLine

      switch turn.role {
      case "system":
        if index > 0 { piece += system }
        piece += turn.text
        if let tools = turn.tools { piece += "\n\n" + toolsBlock(tools) }
      case "user":
        piece += user + (turn.blocks ?? [turn.text]).joined(separator: "\n\n")
      case "latest_reminder":
        piece += reminder + turn.text
      case "assistant":
        var reasoning = ""
        if thinking, !dropThinking || index > lastUserIndex {
          reasoning = (turn.reasoning ?? "") + close
        }
        let calls = turn.calls.isEmpty ? "" : "\n\n" + callsBlock(turn.calls)
        piece += reasoning + turn.text + calls + eos
      default:
        piece += user + turn.text
      }

      let next = index + 1 < turns.count ? turns[index + 1].role : nil
      if let next, next != "assistant", next != "latest_reminder" {
        prompt += piece
        continue
      }
      let opensAnswer = turn.role == "user" || (turn.role == "system" && index > 0)
      if opensAnswer, next != nil || addGenerationPrompt {
        piece += assistant
        if thinking, !dropThinking || index >= lastUserIndex {
          piece += open
        } else {
          piece += close
        }
      }
      prompt += piece
    }
    return prompt
  }

  private static func userBlocks(_ message: ChatMessage) -> [String] {
    switch message.content {
    case .text(let text): return [text]
    case .parts(let parts):
      return parts.map { part in
        switch part {
        case .text(let text): text
        case .image: image
        case .video: "[Unsupported video]"
        }
      }
    }
  }

  private static func schema(_ tool: Value) -> String {
    if case .object(let fields) = tool, let function = fields["function"] {
      return PythonJSON.encode(function)
    }
    return PythonJSON.encode(tool)
  }

  static func toolsBlock(_ schemas: [String]) -> String {
    """
      ## Tools

      You have access to a set of tools to help answer the user's question. You can invoke \
      tools by writing a "\(callsOpen)" block like the following:

      \(callsOpen)
      <\(dsml) invoke name="$TOOL_NAME">
      <\(dsml) parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(dsml) \
      parameter>
      ...
      </\(dsml) invoke>
      <\(dsml) invoke name="$TOOL_NAME2">
      ...
      </\(dsml) invoke>
      \(callsClose)

      String parameters should be specified as is and set `string="true"`. For all other \
      types (numbers, booleans, arrays, objects), pass the value in JSON format and set \
      `string="false"`.

      If thinking_mode is enabled (triggered by \(open)), you MUST output your complete \
      reasoning inside \(open)...\(close) BEFORE any tool calls or final response.

      Otherwise, output directly after \(close) with tool calls or final response.

      ### Available Tool Schemas

      \(schemas.joined(separator: "\n"))

      You MUST strictly follow the above defined tool name and parameter schemas to invoke \
      tool calls.

      """
  }

  static func callsBlock(_ calls: [ToolCall]) -> String {
    let invocations = calls.map { call -> String in
      var arguments: [(String, Value)] = []
      if case .object(let fields) = OrderedJSON.object(call.argumentsJSON) {
        arguments = fields.map { ($0.key.text, $0.value) }
      }
      let parameters = arguments.map { key, value -> String in
        let (text, isString): (String, Bool) =
          if case .string(let raw) = value { (raw, true) } else { (PythonJSON.encode(value), false) }
        return "<\(dsml) parameter name=\"\(key)\" string=\"\(isString)\">\(text)</\(dsml) parameter>"
      }
      return "<\(dsml) invoke name=\"\(call.name)\">\n\(parameters.joined(separator: "\n"))\n</\(dsml) invoke>"
    }
    return "\(callsOpen)\n\(invocations.joined(separator: "\n"))\n\(callsClose)"
  }

  /// The tool calls a completion made, read back out of its DSML block.
  public static func parseCalls(_ text: String) -> (content: String, calls: [ToolCall])? {
    guard let block = text.range(of: callsOpen) else { return nil }
    let before = String(text[..<block.lowerBound])
    let end = text.range(of: callsClose, range: block.upperBound..<text.endIndex)
    let body = String(text[block.upperBound..<(end?.lowerBound ?? text.endIndex)])
    let after = (end.map { String(text[$0.upperBound...]) } ?? "")
      .replacingOccurrences(of: eos, with: "")

    var calls: [ToolCall] = []
    var cursor = body.startIndex
    let invokeOpen = "<\(dsml) invoke name=\""
    let invokeClose = "</\(dsml) invoke>"
    let parameterOpen = "<\(dsml) parameter name=\""
    let parameterClose = "</\(dsml) parameter>"
    while let invoke = body.range(of: invokeOpen, range: cursor..<body.endIndex),
      let nameEnd = body.range(of: "\">", range: invoke.upperBound..<body.endIndex)
    {
      let name = String(body[invoke.upperBound..<nameEnd.lowerBound])
      let stop = body.range(of: invokeClose, range: nameEnd.upperBound..<body.endIndex)
      let inner = body[nameEnd.upperBound..<(stop?.lowerBound ?? body.endIndex)]
      var fields: [String] = []
      var at = inner.startIndex
      while let parameter = inner.range(of: parameterOpen, range: at..<inner.endIndex),
        let keyEnd = inner.range(of: "\" string=\"", range: parameter.upperBound..<inner.endIndex),
        let flagEnd = inner.range(of: "\">", range: keyEnd.upperBound..<inner.endIndex),
        let valueEnd = inner.range(of: parameterClose, range: flagEnd.upperBound..<inner.endIndex)
      {
        let key = String(inner[parameter.upperBound..<keyEnd.lowerBound])
        let isString = inner[keyEnd.upperBound..<flagEnd.lowerBound] == "true"
        let value = String(inner[flagEnd.upperBound..<valueEnd.lowerBound])
        let encoded = isString ? PythonJSON.string(value) : value
        fields.append(PythonJSON.string(key) + ": " + encoded)
        at = valueEnd.upperBound
      }
      calls.append(ToolCall(name: name, argumentsJSON: "{" + fields.joined(separator: ", ") + "}"))
      cursor = stop?.upperBound ?? body.endIndex
    }
    return (before + after, calls)
  }
}

/// `json.dumps(value, ensure_ascii=False)`: Python's separators, its escapes, key order kept
/// where there is an order to keep.
enum PythonJSON {
  /// Keys a schema object is written in when the order it arrived in is gone: the ones every
  /// tool definition leads with, then the rest alphabetically.
  static let leading = [
    "name", "description", "type", "parameters", "properties", "items", "required", "enum",
  ]

  static func encode(_ value: Any) -> String {
    switch value {
    case let value as Value: return encode(value)
    case let string as String: return self.string(string)
    case let number as NSNumber:
      if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
      if CFNumberIsFloatType(number) { return double(number.doubleValue) }
      return number.stringValue
    case let array as [Any]:
      return "[" + array.map { encode($0) }.joined(separator: ", ") + "]"
    case let object as [String: Any]:
      let keys = object.keys.sorted { a, b in
        let ia = leading.firstIndex(of: a) ?? Int.max
        let ib = leading.firstIndex(of: b) ?? Int.max
        return ia != ib ? ia < ib : a < b
      }
      return "{" + keys.map { string($0) + ": " + encode(object[$0]!) }.joined(separator: ", ") + "}"
    case is NSNull: return "null"
    default: return "null"
    }
  }

  static func encode(_ value: Value) -> String {
    switch value {
    case .string(let s): return string(s)
    case .int(let i): return String(i)
    case .double(let d): return double(d)
    case .boolean(let b): return b ? "true" : "false"
    case .null, .undefined: return "null"
    case .array(let items): return "[" + items.map { encode($0) }.joined(separator: ", ") + "]"
    case .object(let fields):
      return "{" + fields.map { string($0.key.text) + ": " + encode($0.value) }.joined(separator: ", ") + "}"
    default: return "null"
    }
  }

  static func double(_ d: Double) -> String {
    guard d.isFinite else { return d.isNaN ? "NaN" : (d > 0 ? "Infinity" : "-Infinity") }
    if d == d.rounded(), abs(d) < 1e16 { return String(format: "%.1f", d) }
    return "\(d)"
  }

  static func string(_ s: String) -> String {
    var out = "\""
    for scalar in s.unicodeScalars {
      switch scalar {
      case "\"": out += "\\\""
      case "\\": out += "\\\\"
      case "\n": out += "\\n"
      case "\r": out += "\\r"
      case "\t": out += "\\t"
      case "\u{08}": out += "\\b"
      case "\u{0C}": out += "\\f"
      default:
        if scalar.value < 0x20 {
          out += String(format: "\\u%04x", scalar.value)
        } else {
          out.unicodeScalars.append(scalar)
        }
      }
    }
    return out + "\""
  }
}

extension ObjectKey {
  var text: String {
    switch self {
    case .string(let s): s
    case .int(let i): String(i)
    }
  }
}
