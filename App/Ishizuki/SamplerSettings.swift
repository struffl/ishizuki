// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The sampler knobs, kept per pack rather than global — a setting dialed in for one model
// should not leak into the next one activated.

import Foundation
import IshizukiKit
import Observation

struct SamplerSettings: Codable, Equatable {
  var temperature: Float = 0.7
  var minP: Float = 0.05

  var topKEnabled = false
  var topK: Int = 40

  var topPEnabled = false
  var topP: Float = 0.9

  var repetitionPenaltyEnabled = false
  var repetitionPenalty: Float = 1.1

  var presencePenaltyEnabled = false
  var presencePenalty: Float = 0.3

  /// Meaningless for a pack; this is what Private Cloud Compute is asked per turn. Optional so
  /// a settings blob saved before this existed still decodes.
  var appleReasoningLevel: AppleReasoningLevel?
  /// Meaningless for a pack; the on-device model's guardrail posture.
  var appleGuardrails: AppleGuardrails?

  var resolvedAppleReasoningLevel: AppleReasoningLevel { appleReasoningLevel ?? .deep }
  var resolvedAppleGuardrails: AppleGuardrails { appleGuardrails ?? .standard }

  static let `default` = SamplerSettings()

  var samplingOptions: SamplingOptions {
    var options = SamplingOptions(temperature: temperature, minP: minP)
    if topKEnabled { options.topK = topK }
    if topPEnabled { options.topP = topP }
    if repetitionPenaltyEnabled { options.repetitionPenalty = repetitionPenalty }
    if presencePenaltyEnabled { options.presencePenalty = presencePenalty }
    return options
  }
}

@MainActor
@Observable
final class SamplerSettingsStore {
  private var table: [String: SamplerSettings]
  private let defaults = UserDefaults.standard
  private static let key = "samplerSettingsByModel"

  init() {
    if let data = UserDefaults.standard.data(forKey: Self.key),
      let decoded = try? JSONDecoder().decode([String: SamplerSettings].self, from: data)
    {
      table = decoded
    } else {
      table = [:]
    }
  }

  func settings(for modelID: String) -> SamplerSettings {
    table[modelID] ?? .default
  }

  func set(_ settings: SamplerSettings, for modelID: String) {
    guard table[modelID] != settings else { return }
    table[modelID] = settings
    guard let data = try? JSONEncoder().encode(table) else { return }
    defaults.set(data, forKey: Self.key)
  }

  func samplingOptions(for modelID: String) -> SamplingOptions {
    settings(for: modelID).samplingOptions
  }
}
