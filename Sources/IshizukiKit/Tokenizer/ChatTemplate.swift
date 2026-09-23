// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Jinja
import OrderedCollections

public struct ChatMessage: Sendable {
  public enum Content: Sendable {
    case text(String)
    case parts([Part])
  }

  public enum Part: Sendable {
    case text(String)
    case image
    case video
  }

  public var role: String
  public var content: Content
  public var toolCalls: [ToolCall]
  /// Files on disk this message's pictures come from, in the order its image parts appear.
  /// Carried so a turn can be rendered before the pictures have been read, and so a picture
  /// that has since been deleted can be dropped from both lists at once.
  public var imagePaths: [String]
  /// An assistant turn's thinking, rendered back so the prompt matches what was generated.
  public var reasoning: String?

  public init(
    role: String, content: Content, toolCalls: [ToolCall] = [], imagePaths: [String] = [],
    reasoning: String? = nil
  ) {
    self.role = role
    self.content = content
    self.toolCalls = toolCalls
    self.imagePaths = imagePaths
    self.reasoning = reasoning
  }

  public static func user(_ text: String) -> ChatMessage {
    ChatMessage(role: "user", content: .text(text))
  }
  public static func system(_ text: String) -> ChatMessage {
    ChatMessage(role: "system", content: .text(text))
  }
  public static func assistant(_ text: String) -> ChatMessage {
    ChatMessage(role: "assistant", content: .text(text))
  }
  public static func assistant(_ text: String, toolCalls: [ToolCall]) -> ChatMessage {
    ChatMessage(role: "assistant", content: .text(text), toolCalls: toolCalls)
  }
  public static func assistant(
    _ text: String, reasoning: String?, toolCalls: [ToolCall] = []
  ) -> ChatMessage {
    ChatMessage(
      role: "assistant", content: .text(text), toolCalls: toolCalls, reasoning: reasoning)
  }

  /// Whatever words this message carries, whichever shape it is in.
  public var plainText: String {
    switch content {
    case .text(let text):
      return text
    case .parts(let parts):
      return parts.compactMap { if case .text(let text) = $0 { text } else { nil } }
        .joined(separator: "\n")
    }
  }

  public static func toolResult(_ text: String) -> ChatMessage {
    ChatMessage(role: "tool", content: .text(text))
  }

  public static func user(text: String, imageCount: Int) -> ChatMessage {
    var parts: [Part] = Array(repeating: .image, count: imageCount)
    if !text.isEmpty { parts.append(.text(text)) }
    return ChatMessage(role: "user", content: .parts(parts))
  }

  /// A turn that came with pictures, named rather than counted: the engine reads them when it
  /// runs, and settles the count against however many it could actually open.
  public static func user(text: String, imagePaths: [String]) -> ChatMessage {
    guard !imagePaths.isEmpty else { return .user(text) }
    var message = user(text: text, imageCount: imagePaths.count)
    message.imagePaths = imagePaths
    return message
  }
}

public enum ReasoningEffort: String, Sendable, CaseIterable, Codable {
  case none, low, medium, high, xhigh
}

public final class ChatTemplate: @unchecked Sendable {
  private let template: Template
  public let source: String

  public init(directory: URL) throws {
    let url = directory.appending(path: "chat_template.jinja")
    guard let source = try? String(contentsOf: url, encoding: .utf8) else {
      throw BonsaiError.missingComponent("no chat_template.jinja in \(directory.path)")
    }
    self.source = source
    self.template = try Template(source)
  }

  public init(source: String) throws {
    self.source = source
    self.template = try Template(source)
  }

  /// A pack keeps its template in a file beside the weights; a GGUF keeps it in the metadata.
  /// Callers hold one path either way, so the shape of what it points at is decided here.
  public convenience init(path: URL) throws {
    guard path.pathExtension.lowercased() == "gguf" else {
      try self.init(directory: path)
      return
    }
    guard let source = try GGUFFile(url: path)["tokenizer.chat_template"]?.stringValue else {
      throw BonsaiError.missingComponent(
        "no tokenizer.chat_template in \(path.lastPathComponent)")
    }
    try self.init(source: source)
  }

  public func render(
    messages: [ChatMessage],
    addGenerationPrompt: Bool = true,
    enableThinking: Bool = true,
    reasoningEffort: ReasoningEffort? = nil,
    tools: [[String: Any]]? = nil,
    extraContext: [String: Value] = [:]
  ) throws -> String {
    var context: [String: Value] = [
      "messages": try Value(any: messages.map(Self.encode)),
      "add_generation_prompt": .boolean(addGenerationPrompt),
      "enable_thinking": .boolean(enableThinking),
    ]
    if let reasoningEffort {
      context["reasoning_effort"] = .string(reasoningEffort.rawValue)
    }
    if let tools {
      context["tools"] = try Value(any: tools)
    }
    if messages.contains(where: { $0.reasoning != nil }) {
      context["preserve_thinking"] = .boolean(true)
    }
    for (key, value) in extraContext { context[key] = value }

    return try template.render(context)
  }

  /// Arguments keep the order they were written in; a dictionary would render them sorted.
  private static func encode(_ message: ChatMessage) -> [String: Any] {
    var encoded = encodeContent(message)
    if let reasoning = message.reasoning, !reasoning.isEmpty {
      encoded["reasoning_content"] = reasoning
    }
    if !message.toolCalls.isEmpty {
      encoded["tool_calls"] = message.toolCalls.map { call -> [String: Any] in
        [
          "id": call.id, "type": "function",
          "function": ["name": call.name, "arguments": OrderedJSON.object(call.argumentsJSON)],
        ]
      }
    }
    return encoded
  }

  private static func encodeContent(_ message: ChatMessage) -> [String: Any] {
    switch message.content {
    case .text(let text):
      return ["role": message.role, "content": text]
    case .parts(let parts):
      let encoded: [[String: Any]] = parts.map { part in
        switch part {
        case .text(let text): ["type": "text", "text": text]
        case .image: ["type": "image", "image": ""]
        case .video: ["type": "video", "video": ""]
        }
      }
      return ["role": message.role, "content": encoded]
    }
  }
}

/// Order-preserving JSON reader for tool call arguments.
enum OrderedJSON {
  static func object(_ text: String) -> Value {
    var reader = Reader(Array(text.utf8))
    guard let value = try? reader.value(), case .object = value else {
      return .object(OrderedDictionary<String, Value>())
    }
    return value
  }

  struct Malformed: Error {}

  struct Reader {
    let bytes: [UInt8]
    var at = 0

    init(_ bytes: [UInt8]) { self.bytes = bytes }

    mutating func skip() {
      while at < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[at]) { at += 1 }
    }

    mutating func value() throws -> Value {
      skip()
      guard at < bytes.count else { throw Malformed() }
      switch bytes[at] {
      case UInt8(ascii: "{"): return try object()
      case UInt8(ascii: "["): return try array()
      case UInt8(ascii: "\""): return .string(try string())
      case UInt8(ascii: "t"): return try literal("true", .boolean(true))
      case UInt8(ascii: "f"): return try literal("false", .boolean(false))
      case UInt8(ascii: "n"): return try literal("null", .null)
      default: return try number()
      }
    }

    mutating func literal(_ word: String, _ value: Value) throws -> Value {
      let spelled = Array(word.utf8)
      guard at + spelled.count <= bytes.count,
        Array(bytes[at..<(at + spelled.count)]) == spelled
      else { throw Malformed() }
      at += spelled.count
      return value
    }

    mutating func object() throws -> Value {
      at += 1
      var fields = OrderedDictionary<String, Value>()
      skip()
      if at < bytes.count, bytes[at] == UInt8(ascii: "}") {
        at += 1
        return .object(fields)
      }
      while true {
        skip()
        guard at < bytes.count, bytes[at] == UInt8(ascii: "\"") else { throw Malformed() }
        let key = try string()
        skip()
        guard at < bytes.count, bytes[at] == UInt8(ascii: ":") else { throw Malformed() }
        at += 1
        fields[key] = try value()
        skip()
        guard at < bytes.count else { throw Malformed() }
        if bytes[at] == UInt8(ascii: ",") {
          at += 1
          continue
        }
        guard bytes[at] == UInt8(ascii: "}") else { throw Malformed() }
        at += 1
        return .object(fields)
      }
    }

    mutating func array() throws -> Value {
      at += 1
      var items: [Value] = []
      skip()
      if at < bytes.count, bytes[at] == UInt8(ascii: "]") {
        at += 1
        return .array(items)
      }
      while true {
        items.append(try value())
        skip()
        guard at < bytes.count else { throw Malformed() }
        if bytes[at] == UInt8(ascii: ",") {
          at += 1
          continue
        }
        guard bytes[at] == UInt8(ascii: "]") else { throw Malformed() }
        at += 1
        return .array(items)
      }
    }

    mutating func string() throws -> String {
      at += 1
      var out: [UInt8] = []
      while at < bytes.count {
        let byte = bytes[at]
        at += 1
        switch byte {
        case UInt8(ascii: "\""):
          return String(decoding: out, as: UTF8.self)
        case UInt8(ascii: "\\"):
          guard at < bytes.count else { throw Malformed() }
          let escaped = bytes[at]
          at += 1
          switch escaped {
          case UInt8(ascii: "n"): out.append(0x0A)
          case UInt8(ascii: "t"): out.append(0x09)
          case UInt8(ascii: "r"): out.append(0x0D)
          case UInt8(ascii: "b"): out.append(0x08)
          case UInt8(ascii: "f"): out.append(0x0C)
          case UInt8(ascii: "u"):
            var scalar = try hex()
            if (0xD800..<0xDC00).contains(scalar), at + 1 < bytes.count,
              bytes[at] == UInt8(ascii: "\\"), bytes[at + 1] == UInt8(ascii: "u")
            {
              at += 2
              let low = try hex()
              scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
            }
            let character = Unicode.Scalar(scalar).map(Character.init) ?? "\u{FFFD}"
            out.append(contentsOf: Array(String(character).utf8))
          default: out.append(escaped)
          }
        default:
          out.append(byte)
        }
      }
      throw Malformed()
    }

    mutating func hex() throws -> UInt32 {
      guard at + 4 <= bytes.count,
        let value = UInt32(String(decoding: bytes[at..<(at + 4)], as: UTF8.self), radix: 16)
      else { throw Malformed() }
      at += 4
      return value
    }

    mutating func number() throws -> Value {
      let start = at
      while at < bytes.count, "+-0123456789.eE".utf8.contains(bytes[at]) { at += 1 }
      let text = String(decoding: bytes[start..<at], as: UTF8.self)
      if let int = Int(text) { return .int(int) }
      if let double = Double(text) { return .double(double) }
      throw Malformed()
    }
  }
}
