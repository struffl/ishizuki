// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

public struct ToolCall: Sendable, Equatable {
  public var id: String
  public var name: String
  public var argumentsJSON: String

  public init(
    id: String = "call_" + UUID().uuidString.prefix(8), name: String, argumentsJSON: String
  ) {
    self.id = id
    self.name = name
    self.argumentsJSON = argumentsJSON
  }
}

public struct ParsedCompletion: Sendable {
  public var reasoning: String?
  public var content: String
  public var toolCalls: [ToolCall]
}

public enum ToolCallParser {
  public static func parse(
    _ raw: String, types: [String: [String: String]] = [:]
  ) -> ParsedCompletion {
    var text = raw
    var reasoning: String?

    if let end = text.range(of: "</think>") {
      let thought = String(text[text.startIndex..<end.lowerBound])
        .replacingOccurrences(of: "<think>", with: "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
      if !thought.isEmpty { reasoning = thought }
      text = String(text[end.upperBound...])
    } else if let open = text.range(of: "<think>") {
      // A thought that was never closed is still a thought. Generation can stop before the
      // closing tag — a token budget runs out, or the model simply omits it — and treating the
      // rest as the answer put the whole chain of reasoning in the reply, which is where a
      // fifteen-kilobyte "answer" came from.
      //
      // The thought runs to the first tool call rather than to the end of the text: a call that
      // followed an unclosed tag would otherwise be read as prose and never made. What sits
      // before the tag is the answer proper and is kept.
      let after = open.upperBound
      let stop = [text.range(of: "<tool_call>", range: after..<text.endIndex)?.lowerBound,
        text.range(of: DeepSeekChatFormat.callsOpen, range: after..<text.endIndex)?.lowerBound]
        .compactMap { $0 }.min()
      let thought = String(text[after..<(stop ?? text.endIndex)])
        .trimmingCharacters(in: .whitespacesAndNewlines)
      if !thought.isEmpty { reasoning = thought }
      text =
        String(text[text.startIndex..<open.lowerBound])
        + (stop.map { String(text[$0...]) } ?? "")
    }

    if let (content, calls) = DeepSeekChatFormat.parseCalls(text) {
      return ParsedCompletion(
        reasoning: reasoning,
        content: content.trimmingCharacters(in: .whitespacesAndNewlines), toolCalls: calls)
    }

    var calls: [ToolCall] = []
    var content = ""
    var cursor = text.startIndex

    while let open = text.range(of: "<tool_call>", range: cursor..<text.endIndex) {
      content += text[cursor..<open.lowerBound]
      let close = text.range(of: "</tool_call>", range: open.upperBound..<text.endIndex)
      let body = String(text[open.upperBound..<(close?.lowerBound ?? text.endIndex)])
      if let call = parseCall(body, types: types) { calls.append(call) }
      cursor = close?.upperBound ?? text.endIndex
    }
    content += text[cursor...]

    return ParsedCompletion(
      reasoning: reasoning,
      content: content.trimmingCharacters(in: .whitespacesAndNewlines),
      toolCalls: calls)
  }

  /// Each tool's parameters and their declared JSON types, so text is never read as JSON.
  public static func parameterTypes(_ tools: [[String: Any]]?) -> [String: [String: String]] {
    var out: [String: [String: String]] = [:]
    for tool in tools ?? [] {
      let function = (tool["function"] as? [String: Any]) ?? tool
      guard let name = function["name"] as? String,
        let parameters = function["parameters"] as? [String: Any],
        let properties = parameters["properties"] as? [String: Any]
      else { continue }
      var types: [String: String] = [:]
      for (key, property) in properties {
        guard let property = property as? [String: Any] else { continue }
        if let type = property["type"] as? String {
          types[key] = type
        } else if let spelled = property["type"] as? [String] {
          types[key] = spelled.first { $0 != "null" }
        } else if let options = (property["anyOf"] ?? property["oneOf"]) as? [[String: Any]] {
          types[key] = options.compactMap { $0["type"] as? String }.first { $0 != "null" }
        }
      }
      out[name] = types
    }
    return out
  }

  private static func parseCall(_ body: String, types: [String: [String: String]]) -> ToolCall? {
    guard let nameStart = body.range(of: "<function="),
      let nameEnd = body.range(of: ">", range: nameStart.upperBound..<body.endIndex)
    else { return nil }
    let name = String(body[nameStart.upperBound..<nameEnd.lowerBound])
      .trimmingCharacters(in: .whitespaces)
    guard !name.isEmpty else { return nil }

    var fields: [(key: String, value: Any)] = []
    var cursor = nameEnd.upperBound
    while let open = body.range(of: "<parameter=", range: cursor..<body.endIndex),
      let openEnd = body.range(of: ">", range: open.upperBound..<body.endIndex),
      let close = body.range(of: "</parameter>", range: openEnd.upperBound..<body.endIndex)
    {
      let key = String(body[open.upperBound..<openEnd.lowerBound])
        .trimmingCharacters(in: .whitespaces)
      let value = unframed(String(body[openEnd.upperBound..<close.lowerBound]))
      let decoded = decodeValue(value, as: types[name]?[key])
      if let index = fields.firstIndex(where: { $0.key == key }) {
        fields[index].value = decoded
      } else {
        fields.append((key, decoded))
      }
      cursor = close.upperBound
    }

    let json =
      "{"
      + fields.map { encodeFragment($0.key) + ":" + encodeFragment($0.value) }
        .joined(separator: ",")
      + "}"
    return ToolCall(name: name, argumentsJSON: json)
  }

  /// A value without the newline the format frames it with; one-line values are trimmed.
  static func unframed(_ raw: String) -> String {
    var value = Substring(raw)
    if let first = value.first, first == "\n" || first == "\r\n" { value = value.dropFirst() }
    if let last = value.last, last == "\n" || last == "\r\n" { value = value.dropLast() }
    guard value.contains(where: \.isNewline) else {
      return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    return String(value)
  }

  private static func encodeFragment(_ value: Any) -> String {
    (try? JSONSerialization.data(
      withJSONObject: value, options: [.fragmentsAllowed, .withoutEscapingSlashes]))
      .flatMap { String(data: $0, encoding: .utf8) } ?? "null"
  }

  private static func decodeValue(_ value: String, as type: String?) -> Any {
    if type == "string" { return value }
    guard let data = "[\(value)]".data(using: .utf8),
      let array = try? JSONSerialization.jsonObject(with: data) as? [Any],
      let first = array.first, !(first is NSNull)
    else { return value }
    return first
  }
}
