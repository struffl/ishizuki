// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Jinja

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

  public init(role: String, content: Content, toolCalls: [ToolCall] = []) {
    self.role = role
    self.content = content
    self.toolCalls = toolCalls
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

  public static func toolResult(_ text: String) -> ChatMessage {
    ChatMessage(role: "tool", content: .text(text))
  }

  public static func user(text: String, imageCount: Int) -> ChatMessage {
    var parts: [Part] = Array(repeating: .image, count: imageCount)
    if !text.isEmpty { parts.append(.text(text)) }
    return ChatMessage(role: "user", content: .parts(parts))
  }
}

public enum ReasoningEffort: String, Sendable, CaseIterable {
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
    for (key, value) in extraContext { context[key] = value }

    return try template.render(context)
  }

  private static func encode(_ message: ChatMessage) -> [String: Any] {
    var encoded = encodeContent(message)
    if !message.toolCalls.isEmpty {
      encoded["tool_calls"] = message.toolCalls.map { call -> [String: Any] in
        let arguments =
          (call.argumentsJSON.data(using: .utf8)
            .flatMap { try? JSONSerialization.jsonObject(with: $0) }
            as? [String: Any]) ?? [:]
        return ["function": ["name": call.name, "arguments": arguments]]
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
