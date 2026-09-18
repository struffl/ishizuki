// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Darwin
import Foundation
import MLX

public enum Politeness {
  public enum Level: String, Sendable, CaseIterable {
    case normal
    case polite
    case adaptive
    case background
  }

  @discardableResult
  public static func apply(_ level: Level) -> Bool {
    switch level {
    case .background:
      return setpriority(PRIO_DARWIN_PROCESS, 0, PRIO_DARWIN_BG) == 0
    case .normal, .polite, .adaptive:
      return setpriority(PRIO_DARWIN_PROCESS, 0, 0) == 0
    }
  }

  public static func qos(for level: Level) -> DispatchQoS {
    switch level {
    case .normal: .userInitiated
    case .polite, .adaptive, .background: .utility
    }
  }

  public static func prefillChunk(for level: Level, default defaultChunk: Int = 512) -> Int {
    switch level {
    case .normal: defaultChunk
    case .polite, .adaptive, .background: max(64, defaultChunk / 4)
    }
  }

  public static var thermalState: ProcessInfo.ThermalState {
    ProcessInfo.processInfo.thermalState
  }

  public static var thermalDescription: String {
    switch thermalState {
    case .nominal: "nominal"
    case .fair: "fair"
    case .serious: "serious"
    case .critical: "critical"
    @unknown default: "unknown"
    }
  }

  public static var isLowPowerMode: Bool {
    ProcessInfo.processInfo.isLowPowerModeEnabled
  }

  public static func throttleDelay(for level: Level) -> TimeInterval {
    guard level == .adaptive else { return 0 }
    if isLowPowerMode { return 0.010 }
    switch thermalState {
    case .nominal: return 0
    case .fair: return 0.002
    case .serious: return 0.010
    case .critical: return 0.050
    @unknown default: return 0
    }
  }

  public static func describe(_ level: Level) -> String {
    var parts = ["priority \(level.rawValue)"]
    parts.append("prefill chunk \(prefillChunk(for: level))")
    if level == .adaptive {
      parts.append("thermal \(thermalDescription)")
      if isLowPowerMode { parts.append("low-power mode") }
    }
    return parts.joined(separator: ", ")
  }
}
