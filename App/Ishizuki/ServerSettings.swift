// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The knobs `ishizuki serve` takes on the command line, kept between launches.

import Foundation
import IshizukiKit
import Observation

@MainActor
@Observable
final class ServerSettings {
  var port: Int { didSet { write(port, "port") } }
  var kvBits: Double { didSet { write(kvBits, "kvBits") } }
  var kvWindow: Int { didSet { write(kvWindow, "kvWindow") } }
  var idleTimeout: Double { didSet { write(idleTimeout, "idleTimeout") } }
  var evictTimeout: Double { didSet { write(evictTimeout, "evictTimeout") } }
  var politeness: String { didSet { write(politeness, "politeness") } }
  var contextScale: Double { didSet { write(contextScale, "contextScale") } }
  var prefixCacheGB: Double { didSet { write(prefixCacheGB, "prefixCacheGB") } }
  var wireGB: Double { didSet { write(wireGB, "wireGB") } }
  var preload: Bool { didSet { write(preload, "preload") } }
  var hot: Bool { didSet { write(hot, "hot") } }
  var activeModelID: String { didSet { write(activeModelID, "activeModelID") } }
  var startOnLaunch: Bool { didSet { write(startOnLaunch, "startOnLaunch") } }
  var neuralEngine: Bool { didSet { write(neuralEngine, "neuralEngine") } }
  var expertSlots: Int { didSet { write(expertSlots, "expertSlots") } }

  /// Empty when the resident pack is answering; an `AppleFoundationModel` raw value when one of
  /// Apple's own models has been chosen instead. Reasoning level and guardrails for that choice
  /// live in `SamplerSettingsStore` beside every pack's own knobs, keyed the same way.
  var appleModelID: String { didSet { write(appleModelID, "appleModelID") } }

  private let defaults = UserDefaults.standard

  init() {
    let defaults = UserDefaults.standard
    port = defaults.object(forKey: "port") as? Int ?? 8128
    kvBits = defaults.object(forKey: "kvBits") as? Double ?? 3.5
    kvWindow = defaults.object(forKey: "kvWindow") as? Int ?? 128
    idleTimeout = defaults.object(forKey: "idleTimeout") as? Double ?? 120
    evictTimeout = defaults.object(forKey: "evictTimeout") as? Double ?? 0
    politeness = defaults.object(forKey: "politeness") as? String ?? "normal"
    contextScale = defaults.object(forKey: "contextScale") as? Double ?? 1
    prefixCacheGB = defaults.object(forKey: "prefixCacheGB") as? Double ?? 8
    wireGB = defaults.object(forKey: "wireGB") as? Double ?? 0
    preload = defaults.object(forKey: "preload") as? Bool ?? true
    hot = defaults.object(forKey: "hot") as? Bool ?? false
    activeModelID = defaults.object(forKey: "activeModelID") as? String ?? ""
    startOnLaunch = defaults.object(forKey: "startOnLaunch") as? Bool ?? false
    neuralEngine = defaults.object(forKey: "neuralEngine") as? Bool ?? false
    appleModelID = defaults.object(forKey: "appleModelID") as? String ?? ""
    expertSlots = defaults.object(forKey: "expertSlots") as? Int ?? BonsaiRuntime.expertSlots
  }

  private func write(_ value: Any, _ key: String) {
    defaults.set(value, forKey: key)
  }

  var kvConfig: KVCacheConfig {
    KVCacheConfig(bits: Float(kvBits), residualWindow: kvWindow)
  }

  var politenessLevel: Politeness.Level {
    Politeness.Level(rawValue: politeness) ?? .adaptive
  }

  var maxContextTokens: Int {
    Int(262_144 * max(contextScale, 1))
  }

  /// nil when the resident pack is answering.
  ///
  /// A model this build no longer offers reads as nil, so a choice saved before it was
  /// withdrawn cannot start a turn that would take the app down.
  var appleModel: AppleFoundationModel? {
    get { AppleFoundationModel(rawValue: appleModelID).flatMap { $0.isOffered ? $0 : nil } }
    set { appleModelID = newValue?.rawValue ?? "" }
  }
}
