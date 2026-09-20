// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One gathered picture of a running server, rendered as text by the CLI and as views by the app.

import Foundation
import MLX

public struct ServeReadout: Sendable {
  public struct Load: Sendable {
    public var held: Int
    public var ceiling: Int
    public var weights: Int
    public var peak: Int
    public var gpu: Double?

    public var fraction: Double { ceiling > 0 ? Double(held) / Double(ceiling) : 0 }
  }

  public struct Context: Sendable {
    public var peakTokens: Int
    public var reservedTokens: Int
    public var ceilingTokens: Int
    public var kvHeldBytes: Int
  }

  public struct Prefix: Sendable {
    public var hits: Int
    public var misses: Int
    public var branches: Int
    public var diskHits: Int
    public var evictions: Int
    public var slots: Int
    public var ramBytes: Int
    public var ramLimit: Int
    public var diskBytes: Int?
    public var diskLimit: Int?

    public var lookups: Int { hits + misses }
    public var hitRate: Double { lookups > 0 ? Double(hits) / Double(lookups) : 0 }
  }

  public struct State: Sendable {
    public var politeness: Politeness.Level
    public var thermal: String
    public var lowPower: Bool
    public var idleSeconds: Double
    public var evictSeconds: Double
    public var uptime: Double
  }

  public var modelName: String
  public var isLoaded: Bool
  public var inFlight: [ServeStats.Request]
  public var totals: ServeStats.Totals
  public var load: Load
  public var budgetSummary: String
  public var headroom: Int
  public var context: Context
  public var prefix: Prefix?
  public var state: State

  public var running: Int { inFlight.filter { $0.phase != .queued }.count }
  public var queued: Int { inFlight.filter { $0.phase == .queued }.count }
}

/// How the readout's numbers are spelled, so a window and a terminal word them the same.
public enum ReadoutFormat {
  public static func gigabytes(_ bytes: Int) -> String {
    String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
  }

  public static func compact(_ bytes: Int) -> String {
    bytes < 1_073_741_824
      ? String(format: "%.0f MB", Double(bytes) / 1_048_576)
      : gigabytes(bytes)
  }

  public static func group(_ value: Int) -> String {
    let digits = Array(String(value))
    var out = ""
    for (index, digit) in digits.enumerated() {
      if index > 0, (digits.count - index) % 3 == 0 { out += " " }
      out.append(digit)
    }
    return out
  }

  public static func percent(_ fraction: Double) -> String {
    String(format: "%.0f%%", min(max(fraction, 0), 1) * 100)
  }

  public static func duration(_ seconds: Double) -> String {
    let total = Int(seconds)
    if total < 60 { return "\(total)s" }
    if total < 3600 { return "\(total / 60)m \(total % 60)s" }
    return "\(total / 3600)h \((total % 3600) / 60)m"
  }
}

extension APIServer {
  /// Everything the dashboard reports, gathered once so the terminal and the app cannot drift.
  public func readout() -> ServeReadout {
    let snapshot = stats.snapshot()
    let memory = Memory.snapshot()
    let weights = budget.weightBytes
    let held = max(weights, memory.activeMemory) + memory.cacheMemory
    peakHeld = max(peakHeld, held, max(weights, memory.peakMemory))

    let prefix: ServeReadout.Prefix? = {
      let lookups = sessions.hits + sessions.misses
      guard lookups > 0 || prefixStore != nil else { return nil }
      return ServeReadout.Prefix(
        hits: sessions.hits,
        misses: sessions.misses,
        branches: sessions.branches,
        diskHits: sessions.diskHits,
        evictions: sessions.evictions,
        slots: sessions.slotCount,
        ramBytes: sessions.cachedBytes,
        ramLimit: sessions.byteLimitBytes,
        diskBytes: prefixStore?.totalBytes,
        diskLimit: prefixStore?.byteLimit)
    }()

    return ServeReadout(
      modelName: modelName,
      isLoaded: isLoaded,
      inFlight: snapshot.inFlight,
      totals: snapshot.totals,
      load: ServeReadout.Load(
        held: held,
        ceiling: max(budget.ceiling, 1),
        weights: weights,
        peak: peakHeld,
        gpu: GPUMeter.utilization()),
      budgetSummary: budget.describe(),
      headroom: budget.headroom,
      context: ServeReadout.Context(
        peakTokens: snapshot.totals.peakContextTokens,
        reservedTokens: budget.tier.contextTokens,
        ceilingTokens: budget.maxContextTokens,
        kvHeldBytes: sessions.cachedBytes),
      prefix: prefix,
      state: ServeReadout.State(
        politeness: politeness,
        thermal: Politeness.thermalDescription,
        lowPower: Politeness.isLowPowerMode,
        idleSeconds: residency.options.idleSeconds,
        evictSeconds: residency.options.evictSeconds,
        uptime: snapshot.totals.uptime))
  }
}
