// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Darwin
import Foundation
import IshizukiKit
import MLX

final class ServeDashboard: @unchecked Sendable {
  private let server: APIServer
  private let header: [String]
  private let logLimit = 6

  private let lock = NSLock()
  private var logLines: [String] = []
  private var painted = 0
  private var stopped = false
  private var peakHeld = 0

  private var timer: DispatchSourceTimer?
  private var signals: [DispatchSourceSignal] = []
  private let queue = DispatchQueue(label: "ishizuki.dashboard")

  init(server: APIServer, header: [String]) {
    self.server = server
    self.header = header
  }

  private var stats: ServeStats { server.stats }
  private var budget: MemoryBudget { server.budget }

  static var isSupported: Bool { isatty(STDOUT_FILENO) == 1 && Style.depth != .none }

  func append(log line: String) {
    lock.lock()
    logLines.append(line)
    if logLines.count > logLimit { logLines.removeFirst(logLines.count - logLimit) }
    lock.unlock()
  }

  func start() {
    for line in header { print(line) }
    print("")
    write("\u{1B}[?25l")
    installSignalHandlers()

    let timer = DispatchSource.makeTimerSource(queue: queue)
    timer.schedule(deadline: .now(), repeating: .milliseconds(100))
    timer.setEventHandler { [weak self] in self?.render() }
    timer.resume()
    self.timer = timer
  }

  func stop() {
    lock.lock()
    guard !stopped else { return lock.unlock() }
    stopped = true
    lock.unlock()
    timer?.cancel()
    write("\u{1B}[?25h")
  }

  private func installSignalHandlers() {
    for number in [SIGINT, SIGTERM] {
      signal(number, SIG_IGN)
      let source = DispatchSource.makeSignalSource(signal: number, queue: queue)
      source.setEventHandler { [weak self] in
        guard let self else { exit(0) }
        let tokens = self.stats.snapshot().totals.totalTokens
        self.stop()
        print("")
        Farewell.print(tokens: tokens)
        Foundation.exit(0)
      }
      source.resume()
      signals.append(source)
    }
  }

  private func render() {
    lock.lock()
    guard !stopped else { return lock.unlock() }
    let recent = logLines
    lock.unlock()

    let snapshot = stats.snapshot()
    let width = terminalWidth()
    var lines: [String] = []

    lines.append(inFlightHeading(snapshot))
    if snapshot.inFlight.isEmpty {
      lines.append("  " + Style.faint("idle — waiting for requests"))
    } else {
      for request in snapshot.inFlight.prefix(10) {
        lines.append(requestLine(request, width: width))
      }
      if snapshot.inFlight.count > 10 {
        lines.append("  " + Style.faint("+\(snapshot.inFlight.count - 10) more"))
      }
    }

    lines.append("  " + Style.rule)
    lines.append(contentsOf: sessionLines(snapshot.totals))

    if !recent.isEmpty {
      lines.append("  " + Style.rule)
      for line in recent {
        lines.append("  " + Style.faint(truncate(line, to: max(10, width - 4))))
      }
    }

    paint(lines)
  }

  private func inFlightHeading(_ snapshot: ServeStats.Snapshot) -> String {
    var parts: [String] = []
    parts.append(Style.bright("\(snapshot.running)") + Style.muted(" running"))
    if snapshot.queued > 0 {
      parts.append(Style.warn("\(snapshot.queued)") + Style.muted(" queued"))
    }
    return "  " + Style.muted("in flight  ") + parts.joined(separator: Style.faint(" · "))
  }

  private func requestLine(_ request: ServeStats.Request, width: Int) -> String {
    var line = "  "
    line += Style.faint("#" + pad("\(request.id)", 4, right: false))
    line += " " + Style.accent(pad(request.api, 10))
    line += " " + Style.faint(pad(request.stream ? "stream" : "block", 6))
    line += " " + phaseLabel(request)
    line += " " + progressField(request)
    line += " " + Style.muted(pad(String(format: "%.1fs", request.elapsed), 7, right: false))
    return line
  }

  private func phaseLabel(_ request: ServeStats.Request) -> String {
    let text = pad(request.phase.rawValue, 9)
    switch request.phase {
    case .queued: return Style.warn(text)
    case .prefill: return Style.accent(text)
    case .decode: return Style.good(text)
    case .finishing: return Style.muted(text)
    }
  }

  private func progressField(_ request: ServeStats.Request) -> String {
    switch request.phase {
    case .queued:
      return Style.faint(pad("", 40))
    case .prefill:
      let total = max(request.prefillTotal, 1)
      let bar = meter(Double(request.prefilled) / Double(total))
      let counts = "\(group(request.prefilled))/\(group(total)) tok"
      var cached = ""
      if request.cachedTokens > 0 { cached = " +\(group(request.cachedTokens)) cached" }
      return bar + " "
        + Style.bright(pad(String(format: "%.0f", request.rate), 5, right: false))
        + Style.muted(" tok/s ")
        + Style.faint(pad(counts + cached, 20))
    case .decode:
      let total = max(request.maxTokens, 1)
      let bar = meter(Double(request.generated) / Double(total))
      let counts = "\(group(request.generated))/\(group(request.maxTokens)) tok"
      return bar + " "
        + Style.bright(pad(String(format: "%.1f", request.rate), 5, right: false))
        + Style.muted(" tok/s ")
        + Style.faint(pad(counts, 20))
    case .finishing:
      return Style.faint(pad("", 40))
    }
  }

  private func meter(_ fraction: Double, width: Int = 12) -> String {
    let clamped = min(max(fraction, 0), 1)
    let filled = Int((Double(width) * clamped).rounded())
    return Style.accent(String(repeating: "━", count: filled))
      + Style.faint(String(repeating: "╌", count: width - filled))
  }

  private func sessionLines(_ totals: ServeStats.Totals) -> [String] {
    var lines: [String] = []

    lines.append(
      "  "
        + Style.field(
          "prefill",
          Style.bright(String(format: "%.1f", totals.prefillRate))
            + Style.muted(" tok/s")
            + Style.faint(
              String(format: "  last %.1f", totals.lastPrefillRate))))
    lines.append(
      "  "
        + Style.field(
          "decode",
          Style.bright(String(format: "%.1f", totals.decodeRate))
            + Style.muted(" tok/s")
            + Style.faint(
              String(format: "  last %.1f", totals.lastDecodeRate))))
    lines.append(
      "  "
        + Style.field(
          "requests",
          Style.accent("\(totals.completed)") + Style.muted(" done")
            + Style.faint(" · ")
            + (totals.failed > 0
              ? Style.bad("\(totals.failed)") + Style.muted(" failed")
              : Style.faint("0 failed"))
            + (totals.cancelled > 0
              ? Style.faint(" · ") + Style.warn("\(totals.cancelled)")
                + Style.muted(" cancelled")
              : "")
            + Style.faint(" · \(totals.arrived) seen")))
    lines.append(
      "  "
        + Style.field(
          "tokens",
          Style.accent(group(totals.promptTokens)) + Style.muted(" in")
            + Style.faint(" · ")
            + Style.accent(group(totals.generatedTokens)) + Style.muted(" out")))
    lines.append(
      "  "
        + Style.field(
          "cache",
          Style.accent(String(format: "%.0f%%", totals.cacheRatio * 100))
            + Style.faint(
              "  \(totals.cacheHits) hit · \(totals.cacheMisses) miss · "
                + "\(group(totals.cachedTokens)) tok reused")))
    lines.append(contentsOf: loadLines(totals))

    let residency = server.residency.options
    var conditions = [Style.accent(server.politeness.rawValue)]
    conditions.append(Style.faint("thermal \(Politeness.thermalDescription)"))
    if Politeness.isLowPowerMode { conditions.append(Style.warn("low power")) }
    if residency.idleSeconds > 0 {
      conditions.append(Style.faint("pool freed at \(Int(residency.idleSeconds))s idle"))
    }
    if residency.evictSeconds > 0 {
      conditions.append(Style.faint("unload at \(Int(residency.evictSeconds))s idle"))
    }
    conditions.append(Style.faint("up \(duration(totals.uptime))"))
    lines.append("  " + Style.field("state", conditions.joined(separator: Style.faint(" · "))))

    return lines
  }

  private func loadLines(_ totals: ServeStats.Totals) -> [String] {
    let memory = Memory.snapshot()
    let weights = budget.weightBytes
    let held = max(weights, memory.activeMemory) + memory.cacheMemory
    let ceiling = max(budget.ceiling, 1)
    let tier = budget.tier
    peakHeld = max(peakHeld, held, max(weights, memory.peakMemory))

    var lines: [String] = []
    lines.append(
      "  "
        + Style.field(
          "memory",
          gauge(Double(held) / Double(ceiling))
            + " " + Style.bright(pad(percent(Double(held) / Double(ceiling)), 4, right: false))
            + " " + Style.faint("\(gigabytes(held)) / \(gigabytes(ceiling))")
            + Style.faint(
              "   weights \(gigabytes(weights))"
                + " · peak \(gigabytes(peakHeld))")))

    if let usage = GPUMeter.utilization() {
      lines.append(
        "  "
          + Style.field(
            "gpu",
            gauge(usage)
              + " " + Style.bright(pad(percent(usage), 4, right: false))
              + Style.muted(" busy")))
    }

    lines.append(
      "  "
        + Style.field(
          "budget",
          Style.accent(budget.describe())
            + Style.faint(" · \(gigabytes(budget.headroom)) spare")))
    lines.append(
      "  "
        + Style.field(
          "context",
          Style.accent(MemoryBudget.tokens(totals.peakContextTokens) + " peak")
            + Style.faint(
              " · \(MemoryBudget.tokens(tier.contextTokens)) reserved"
                + " · \(MemoryBudget.tokens(budget.maxContextTokens)) ceiling"
                + " · \(compact(server.sessions.cachedBytes)) kv held")))

    let sessions = server.sessions
    let lookups = sessions.hits + sessions.misses
    if lookups > 0 || server.prefixStore != nil {
      // What the prefix cache is holding, against what it is allowed to hold, in each tier it
      // uses. A hit rate with no denominator says nothing about whether it has room to work.
      var occupancy: [String] = []
      let ramLimit = sessions.byteLimitBytes
      occupancy.append(
        ramLimit > 0
          ? "\(compact(sessions.cachedBytes)) / \(compact(ramLimit)) ram"
          : "\(compact(sessions.cachedBytes)) ram")
      if let store = server.prefixStore {
        occupancy.append(
          "\(compact(store.totalBytes)) / \(compact(store.byteLimit)) disk")
      }

      var headline = Style.accent(occupancy.joined(separator: Style.faint(" · ")))
      if lookups > 0 {
        headline =
          Style.accent(percent(Double(sessions.hits) / Double(lookups)) + " hit")
          + Style.faint(" · ") + headline
      }
      lines.append("  " + Style.field("prefix", headline))

      var detail: [String] = []
      if lookups > 0 { detail.append("\(sessions.hits) of \(lookups) reused") }
      if sessions.branches > 0 { detail.append("\(sessions.branches) branched") }
      if sessions.diskHits > 0 { detail.append("\(sessions.diskHits) from disk") }
      if sessions.evictions > 0 { detail.append("\(sessions.evictions) evicted") }
      detail.append("\(sessions.slotCount) slot\(sessions.slotCount == 1 ? "" : "s")")
      lines.append("  " + Style.field("", Style.faint(detail.joined(separator: " · "))))
    }

    return lines
  }

  private func percent(_ fraction: Double) -> String {
    String(format: "%.0f%%", min(max(fraction, 0), 1) * 100)
  }

  private func gauge(_ fraction: Double, width: Int = 14) -> String {
    let clamped = min(max(fraction, 0), 1)
    let filled = Int((Double(width) * clamped).rounded())
    let bar = String(repeating: "━", count: filled)
    let colour: String
    if clamped >= 0.85 {
      colour = Style.bad(bar)
    } else if clamped >= 0.6 {
      colour = Style.warn(bar)
    } else {
      colour = Style.good(bar)
    }
    return colour + Style.faint(String(repeating: "╌", count: width - filled))
  }

  private func paint(_ lines: [String]) {
    let width = terminalWidth()
    var output = ""
    if painted > 0 { output += "\u{1B}[\(painted)A" }
    for line in lines { output += "\u{1B}[2K" + clip(line, to: width) + "\n" }
    if painted > lines.count {
      let extra = painted - lines.count
      output += String(repeating: "\u{1B}[2K\n", count: extra)
      output += "\u{1B}[\(extra)A"
    }
    painted = lines.count
    write(output)
  }

  private func write(_ text: String) {
    FileHandle.standardOutput.write(Data(text.utf8))
  }

  private func terminalWidth() -> Int {
    var size = winsize()
    if ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &size) == 0, size.ws_col > 0 {
      return Int(size.ws_col)
    }
    return 100
  }
}

func clip(_ line: String, to width: Int) -> String {
  var out = ""
  var visible = 0
  var index = line.startIndex
  while index < line.endIndex {
    let character = line[index]
    if character == "\u{1B}" {
      var end = line.index(after: index)
      while end < line.endIndex, !("@"..."~").contains(line[end]) {
        end = line.index(after: end)
      }
      if end < line.endIndex { end = line.index(after: end) }
      out += line[index..<end]
      index = end
      continue
    }
    if visible < width {
      out.append(character)
      visible += 1
    }
    index = line.index(after: index)
  }
  return out
}

private func pad(_ text: String, _ width: Int, right: Bool = true) -> String {
  let trimmed = truncate(text, to: width)
  let padding = String(repeating: " ", count: max(0, width - trimmed.count))
  return right ? trimmed + padding : padding + trimmed
}

private func truncate(_ text: String, to width: Int) -> String {
  guard text.count > width else { return text }
  guard width > 1 else { return String(text.prefix(width)) }
  return String(text.prefix(width - 1)) + "…"
}

private func group(_ value: Int) -> String {
  let digits = Array(String(value))
  var out = ""
  for (index, digit) in digits.enumerated() {
    if index > 0, (digits.count - index) % 3 == 0 { out += " " }
    out.append(digit)
  }
  return out
}

private func gigabytes(_ bytes: Int) -> String {
  String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
}

private func compact(_ bytes: Int) -> String {
  bytes < 1_073_741_824
    ? String(format: "%.0f MB", Double(bytes) / 1_048_576)
    : gigabytes(bytes)
}

private func duration(_ seconds: Double) -> String {
  let total = Int(seconds)
  if total < 60 { return "\(total)s" }
  if total < 3600 { return "\(total / 60)m \(total % 60)s" }
  return "\(total / 3600)h \((total % 3600) / 60)m"
}
