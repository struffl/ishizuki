// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

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
  public static func parse(_ raw: String) -> ParsedCompletion {
    var text = raw
    var reasoning: String?

    if let end = text.range(of: "</think>") {
      let thought = String(text[text.startIndex..<end.lowerBound])
        .replacingOccurrences(of: "<think>", with: "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
      if !thought.isEmpty { reasoning = thought }
      text = String(text[end.upperBound...])
    }

    var calls: [ToolCall] = []
    var content = ""
    var cursor = text.startIndex

    while let open = text.range(of: "<tool_call>", range: cursor..<text.endIndex) {
      content += text[cursor..<open.lowerBound]
      let close = text.range(of: "</tool_call>", range: open.upperBound..<text.endIndex)
      let body = String(text[open.upperBound..<(close?.lowerBound ?? text.endIndex)])
      if let call = parseCall(body) { calls.append(call) }
      cursor = close?.upperBound ?? text.endIndex
    }
    content += text[cursor...]

    return ParsedCompletion(
      reasoning: reasoning,
      content: content.trimmingCharacters(in: .whitespacesAndNewlines),
      toolCalls: calls)
  }

  private static func parseCall(_ body: String) -> ToolCall? {
    guard let nameStart = body.range(of: "<function="),
      let nameEnd = body.range(of: ">", range: nameStart.upperBound..<body.endIndex)
    else { return nil }
    let name = String(body[nameStart.upperBound..<nameEnd.lowerBound])
      .trimmingCharacters(in: .whitespaces)
    guard !name.isEmpty else { return nil }

    var arguments: [String: Any] = [:]
    var cursor = nameEnd.upperBound
    while let open = body.range(of: "<parameter=", range: cursor..<body.endIndex),
      let openEnd = body.range(of: ">", range: open.upperBound..<body.endIndex),
      let close = body.range(of: "</parameter>", range: openEnd.upperBound..<body.endIndex)
    {
      let key = String(body[open.upperBound..<openEnd.lowerBound])
        .trimmingCharacters(in: .whitespaces)
      let value = String(body[openEnd.upperBound..<close.lowerBound])
        .trimmingCharacters(in: .whitespacesAndNewlines)
      arguments[key] = decodeValue(value)
      cursor = close.upperBound
    }

    let json =
      (try? JSONSerialization.data(withJSONObject: arguments))
      .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    return ToolCall(name: name, argumentsJSON: json)
  }

  private static func decodeValue(_ value: String) -> Any {
    guard let data = "[\(value)]".data(using: .utf8),
      let array = try? JSONSerialization.jsonObject(with: data) as? [Any],
      let first = array.first
    else { return value }
    return first
  }
}
