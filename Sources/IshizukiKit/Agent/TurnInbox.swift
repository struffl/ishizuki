// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The person's side of a running turn: what they say while it works, and the question it is
// waiting on them to answer.

import Foundation

public final class TurnInbox: @unchecked Sendable {
  public struct Steer: Sendable, Equatable, Identifiable {
    public let id: String
    public let text: String

    public init(id: String = UUID().uuidString, text: String) {
      self.id = id
      self.text = text
    }
  }

  public struct Question: Sendable, Equatable, Identifiable {
    public let id: String
    public let text: String
    public let options: [String]
  }

  private let lock = NSLock()
  private var steers: [Steer] = []
  private var question: Question?
  private var reply: CheckedContinuation<String, Error>?

  public init() {}

  public func steer(_ steer: Steer) {
    lock.withLock { steers.append(steer) }
  }

  public var waiting: [Steer] { lock.withLock { steers } }

  @discardableResult
  public func remove(_ id: String) -> Bool {
    lock.withLock {
      let before = steers.count
      steers.removeAll { $0.id == id }
      return steers.count != before
    }
  }

  public func take() -> [Steer] {
    lock.withLock {
      defer { steers.removeAll() }
      return steers
    }
  }

  public var pending: Question? { lock.withLock { question } }

  /// Waits for the person. Only one question is open at a time; a stopped turn withdraws it.
  public func ask(_ text: String, options: [String]) async throws -> String {
    let id = UUID().uuidString
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        let refusal: Error? = lock.withLock {
          if Task.isCancelled { return CancellationError() }
          if reply != nil {
            return LedgerRefusal(
              message: "another question is already waiting; ask again once it is answered")
          }
          question = Question(id: id, text: text, options: options)
          reply = continuation
          return nil
        }
        if let refusal { continuation.resume(throwing: refusal) }
      }
    } onCancel: {
      withdraw(id, with: CancellationError())
    }
  }

  @discardableResult
  public func answer(_ text: String) -> Bool {
    let waiting: CheckedContinuation<String, Error>? = lock.withLock {
      defer {
        reply = nil
        question = nil
      }
      return reply
    }
    waiting?.resume(returning: text)
    return waiting != nil
  }

  /// Stops waiting on the person, for a turn that is being stopped.
  public func cancelQuestion() {
    guard let id = lock.withLock({ question?.id }) else { return }
    withdraw(id, with: CancellationError())
  }

  private func withdraw(_ id: String, with error: Error) {
    let waiting: CheckedContinuation<String, Error>? = lock.withLock {
      guard question?.id == id else { return nil }
      defer {
        reply = nil
        question = nil
      }
      return reply
    }
    waiting?.resume(throwing: error)
  }

  /// A tool's output with whatever the person said since the last one appended, so it reaches
  /// the model mid-turn rather than after it.
  public func deliver(into output: String) -> String {
    let said = take()
    guard !said.isEmpty else { return output }
    return output + said.map { SteerBlock.open + $0.text + SteerBlock.close }.joined()
  }
}

public enum SteerBlock {
  static let open = "\n\n<steer>\n"
  static let close = "\n</steer>"

  /// A tool output split back into what the tool said and what the person added to it.
  public static func split(_ text: String) -> (output: String, steers: [String]) {
    guard let first = text.range(of: open) else { return (text, []) }
    var steers: [String] = []
    var rest = text[first.lowerBound...]
    while let start = rest.range(of: open) {
      let body = rest[start.upperBound...]
      guard let end = body.range(of: close) else { break }
      steers.append(String(body[..<end.lowerBound]))
      rest = body[end.upperBound...]
    }
    return (String(text[..<first.lowerBound]), steers)
  }
}

/// Where and when a conversation is happening, written once at the head of its first prompt.
/// Kept out of the instructions so they stay the same for every conversation and every day,
/// which is what lets one cached copy of them serve all of them.
public enum PromptEnvironment {
  static let open = "<environment>\n"
  static let close = "\n</environment>\n\n"

  public static func block(folder: URL?, branch: String?, date: Date = Date()) -> String {
    var lines: [String] = []
    if let folder { lines.append("folder: \(folder.path)") }
    if let branch, !branch.isEmpty { lines.append("git branch: \(branch)") }
    lines.append("date: \(date.formatted(.iso8601.year().month().day()))")
    let system = ProcessInfo.processInfo.operatingSystemVersion
    lines.append("os: macOS \(system.majorVersion).\(system.minorVersion)")
    return open + lines.joined(separator: "\n") + close
  }

  public static func split(_ text: String) -> (environment: String?, body: String) {
    guard text.hasPrefix(open), let end = text.range(of: close) else { return (nil, text) }
    let inside = text[text.index(text.startIndex, offsetBy: open.count)..<end.lowerBound]
    return (String(inside), String(text[end.upperBound...]))
  }
}
