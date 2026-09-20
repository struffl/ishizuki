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

  private var timer: DispatchSourceTimer?
  private var signals: [DispatchSourceSignal] = []
  private let queue = DispatchQueue(label: "ishizuki.dashboard")

  init(server: APIServer, header: [String]) {
    self.server = server
    self.header = header
  }

  private var stats: ServeStats { server.stats }

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

    let readout = server.readout()
    let width = terminalWidth()
    var lines: [String] = []

    lines.append(inFlightHeading(readout))
    if readout.inFlight.isEmpty {
      lines.append("  " + Style.faint("idle — waiting for requests"))
    } else {
      for request in readout.inFlight.prefix(10) {
        lines.append(requestLine(request, width: width))
      }
      if readout.inFlight.count > 10 {
        lines.append("  " + Style.faint("+\(readout.inFlight.count - 10) more"))
      }
    }

    lines.append("  " + Style.rule)
    lines.append(contentsOf: sessionLines(readout))

    if !recent.isEmpty {
      lines.append("  " + Style.rule)
      for line in recent {
        lines.append("  " + Style.faint(truncate(line, to: max(10, width - 4))))
      }
    }

    paint(lines)
  }

  private func inFlightHeading(_ readout: ServeReadout) -> String {
    var parts: [String] = []
    parts.append(Style.bright("\(readout.running)") + Style.muted(" running"))
    if readout.queued > 0 {
      parts.append(Style.warn("\(readout.queued)") + Style.muted(" queued"))
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

  private func sessionLines(_ readout: ServeReadout) -> [String] {
    let totals = readout.totals
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
    lines.append(contentsOf: loadLines(readout))

    let state = readout.state
    var conditions = [Style.accent(state.politeness.rawValue)]
    conditions.append(Style.faint("thermal \(state.thermal)"))
    if state.lowPower { conditions.append(Style.warn("low power")) }
    if state.idleSeconds > 0 {
      conditions.append(Style.faint("pool freed at \(Int(state.idleSeconds))s idle"))
    }
    if state.evictSeconds > 0 {
      conditions.append(Style.faint("unload at \(Int(state.evictSeconds))s idle"))
    }
    conditions.append(Style.faint("up \(duration(state.uptime))"))
    lines.append("  " + Style.field("state", conditions.joined(separator: Style.faint(" · "))))

    return lines
  }

  private func loadLines(_ readout: ServeReadout) -> [String] {
    let load = readout.load
    let context = readout.context

    var lines: [String] = []
    lines.append(
      "  "
        + Style.field(
          "memory",
          gauge(load.fraction)
            + " " + Style.bright(pad(percent(load.fraction), 4, right: false))
            + " " + Style.faint("\(gigabytes(load.held)) / \(gigabytes(load.ceiling))")
            + Style.faint(
              "   weights \(gigabytes(load.weights))"
                + " · peak \(gigabytes(load.peak))")))

    if let usage = load.gpu {
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
          Style.accent(readout.budgetSummary)
            + Style.faint(" · \(gigabytes(readout.headroom)) spare")))
    lines.append(
      "  "
        + Style.field(
          "context",
          Style.accent(MemoryBudget.tokens(context.peakTokens) + " peak")
            + Style.faint(
              " · \(MemoryBudget.tokens(context.reservedTokens)) reserved"
                + " · \(MemoryBudget.tokens(context.ceilingTokens)) ceiling"
                + " · \(compact(context.kvHeldBytes)) kv held")))

    if let prefix = readout.prefix {
      // What the prefix cache is holding, against what it is allowed to hold, in each tier it
      // uses. A hit rate with no denominator says nothing about whether it has room to work.
      var occupancy: [String] = []
      occupancy.append(
        prefix.ramLimit > 0
          ? "\(compact(prefix.ramBytes)) / \(compact(prefix.ramLimit)) ram"
          : "\(compact(prefix.ramBytes)) ram")
      if let bytes = prefix.diskBytes, let limit = prefix.diskLimit {
        occupancy.append("\(compact(bytes)) / \(compact(limit)) disk")
      }

      var headline = Style.accent(occupancy.joined(separator: Style.faint(" · ")))
      if prefix.lookups > 0 {
        headline =
          Style.accent(percent(prefix.hitRate) + " hit")
          + Style.faint(" · ") + headline
      }
      lines.append("  " + Style.field("prefix", headline))

      var detail: [String] = []
      if prefix.lookups > 0 { detail.append("\(prefix.hits) of \(prefix.lookups) reused") }
      if prefix.branches > 0 { detail.append("\(prefix.branches) branched") }
      if prefix.diskHits > 0 { detail.append("\(prefix.diskHits) from disk") }
      if prefix.evictions > 0 { detail.append("\(prefix.evictions) evicted") }
      detail.append("\(prefix.slots) slot\(prefix.slots == 1 ? "" : "s")")
      lines.append("  " + Style.field("", Style.faint(detail.joined(separator: " · "))))
    }

    return lines
  }

  private func percent(_ fraction: Double) -> String { ReadoutFormat.percent(fraction) }

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
      // `[` is itself in the final-byte range, so the CSI introducer has to be stepped over
      // before the terminator search, or style bytes count against the visible width.
      if end < line.endIndex, line[end] == "[" {
        end = line.index(after: end)
      }
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

private func group(_ value: Int) -> String { ReadoutFormat.group(value) }
private func gigabytes(_ bytes: Int) -> String { ReadoutFormat.gigabytes(bytes) }
private func compact(_ bytes: Int) -> String { ReadoutFormat.compact(bytes) }
private func duration(_ seconds: Double) -> String { ReadoutFormat.duration(seconds) }
