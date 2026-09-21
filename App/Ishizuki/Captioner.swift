// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Short labels for long things, written by the system's own model rather than the pack. A
// thought or a command says what it is at a glance, and the 27B stays free for the work.

import Foundation
import FoundationModels
import Observation

@available(macOS 27.0, *)
@MainActor
@Observable
final class Captioner {
  enum Subject {
    case thought
    case command(tool: String)
    case conversation

    var instruction: String {
      switch self {
      case .thought:
        "Say what this reasoning is doing, as a phrase of at most six words. "
          + "Write it like a heading: 'Checking the workspace layout'. No trailing full stop."
      case .command(let tool):
        "Say what this \(tool) call does, as a phrase of at most six words. "
          + "Write it like a heading: 'Listing the project files'. No trailing full stop."
      case .conversation:
        "Give this conversation a title of at most five words. "
          + "Name the task, not the pleasantries. No trailing full stop."
      }
    }

    var limit: Int {
      switch self {
      case .conversation: 16
      default: 20
      }
    }
  }

  /// Content tagging is the lighter of the system model's paths and the one meant for this;
  /// which variant answers is the system's to choose, not ours.
  private let model = SystemLanguageModel(useCase: .contentTagging)
  private var captions: [String: String] = [:]
  private var inFlight: Set<String> = []

  var isAvailable: Bool { model.isAvailable }

  var unavailableReason: String? {
    switch model.availability {
    case .available: nil
    case .unavailable(.appleIntelligenceNotEnabled): "Apple Intelligence is off"
    case .unavailable(.deviceNotEligible): "this Mac cannot run it"
    case .unavailable(.modelNotReady): "the system model is still downloading"
    case .unavailable: "the system model is unavailable"
    }
  }

  func caption(for key: String) -> String? { captions[key] }

  /// Best effort and once per thing: a caption that fails to arrive is a caption not shown,
  /// never a turn that fails. `onCaption` is for a caller that needs to know the moment it
  /// lands, rather than poll `caption(for:)`, since the request always returns before the
  /// model has answered.
  func request(_ key: String, text: String, as subject: Subject, onCaption: ((String) -> Void)? = nil) {
    guard model.isAvailable, captions[key] == nil, !inFlight.contains(key) else { return }
    let source = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard source.count > 40 else { return }
    inFlight.insert(key)

    Task { [model] in
      defer { inFlight.remove(key) }
      let session = LanguageModelSession(model: model, instructions: subject.instruction)
      // Long input is wasted on a caption, and the system model has its own context to mind.
      let excerpt = String(source.prefix(1600))
      guard
        let response = try? await session.respond(
          to: excerpt,
          options: GenerationOptions(maximumResponseTokens: subject.limit))
      else { return }

      let caption =
        response.content
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .trimmingCharacters(in: CharacterSet(charactersIn: "\"'.“”"))
        .replacing(/\s+/, with: " ")
      guard !caption.isEmpty, caption.count < 80 else { return }
      captions[key] = caption
      onCaption?(caption)
    }
  }
}
