// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Apple's own models, reached straight through the Foundation Models framework instead of
// through the resident pack. Picking one bypasses APIServer entirely: nothing to load, nothing
// this runtime quantized, nothing for the readout to meter.

import Foundation

public enum AppleFoundationModel: String, CaseIterable, Sendable, Identifiable, Codable {
  /// The on-device model behind Apple Intelligence: free, private, and the only one of the two
  /// that answers with no network at all.
  case onDevice
  /// The same family, run on Apple's own silicon under Private Cloud Compute for a turn the
  /// phone or the Mac cannot carry alone. New in OS 27.
  case privateCloudCompute

  public var id: String { rawValue }

  public var displayName: String {
    switch self {
    case .onDevice: return "Apple Intelligence"
    case .privateCloudCompute: return "Apple Intelligence (Private Cloud Compute)"
    }
  }

  public var subtitle: String {
    switch self {
    case .onDevice: return "On this device, no network"
    case .privateCloudCompute: return "Apple's own servers, for a heavier turn"
    }
  }
}

/// How hard Private Cloud Compute is asked to think, chosen per turn. The on-device model has
/// only the one gear and ignores this.
public enum AppleReasoningLevel: String, CaseIterable, Sendable, Codable {
  case light
  case deep

  public var displayName: String { self == .light ? "Light" : "Deep" }
}

/// The on-device model's guardrail posture. Permissive is for an app that has to work with
/// mature or sensitive material a chat assistant would ordinarily refuse outright.
public enum AppleGuardrails: String, CaseIterable, Sendable, Codable {
  case standard
  case permissive

  public var displayName: String { self == .standard ? "Standard" : "Permissive" }
}
