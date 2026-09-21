// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What a session does with a streamed reply. The window was showing the first few characters
// of an answer and nothing after it, and the only way to know whether that is the session's
// doing or ours is to send it a known stream and read back what it kept.

import Foundation
import FoundationModels
import Testing

@testable import IshizukiKit

@available(macOS 27.0, *)
struct Chunks: Sendable {
  var texts: [String]
  var tokenCount: Int
}

@available(macOS 27.0, *)
struct StubModel: LanguageModel {
  typealias Executor = StubExecutor

  let executorConfiguration: StubExecutor.Configuration

  init(chunks: Chunks) {
    self.executorConfiguration = .init(chunks: chunks)
  }

  var capabilities: LanguageModelCapabilities { LanguageModelCapabilities([]) }
}

@available(macOS 27.0, *)
struct StubExecutor: LanguageModelExecutor {
  typealias Model = StubModel

  struct Configuration: Hashable, Sendable {
    var chunks: Chunks

    static func == (a: Configuration, b: Configuration) -> Bool {
      a.chunks.texts == b.chunks.texts && a.chunks.tokenCount == b.chunks.tokenCount
    }

    func hash(into hasher: inout Hasher) {
      hasher.combine(chunks.texts)
      hasher.combine(chunks.tokenCount)
    }
  }

  let configuration: Configuration

  init(configuration: Configuration) throws {
    self.configuration = configuration
  }

  func respond(
    to request: LanguageModelExecutorGenerationRequest,
    model: Model,
    streamingInto channel: LanguageModelExecutorGenerationChannel
  ) async throws {
    for text in configuration.chunks.texts {
      await channel.send(
        .response(action: .appendText(text, tokenCount: configuration.chunks.tokenCount)))
    }
  }
}

@Suite("Transcript append")
struct TranscriptAppendTests {
  /// Three chunks sent the way the executor sends them, then read back.
  private func reply(_ texts: [String], tokenCount: Int) async throws -> String? {
    guard #available(macOS 27.0, *) else { return nil }
    let session = LanguageModelSession(
      model: StubModel(chunks: Chunks(texts: texts, tokenCount: tokenCount)))
    _ = try await session.respond(to: "go")

    var seen: [String] = []
    for entry in session.transcript {
      guard case .response(let response) = entry else { continue }
      for segment in response.segments {
        if case .text(let text) = segment { seen.append(text.content) }
      }
    }
    return seen.joined()
  }

  @Test("every chunk of a streamed reply is kept")
  func keepsEveryChunk() async throws {
    let joined = try await reply(["The us", "er is asking ", "where we are."], tokenCount: 2)
    try #require(joined != nil)
    #expect(joined == "The user is asking where we are.")
  }

  /// The suspicion: a zero token count reads as nothing having arrived, so everything after
  /// the first append is discarded.
  @Test("a zero token count does not lose the rest of the reply")
  func zeroTokenCount() async throws {
    let joined = try await reply(["The us", "er is asking ", "where we are."], tokenCount: 0)
    try #require(joined != nil)
    #expect(joined == "The user is asking where we are.")
  }
}
