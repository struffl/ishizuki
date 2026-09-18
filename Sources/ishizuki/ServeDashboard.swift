// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Darwin
import Foundation
import IshizukiKit
import MLX

final class ServeDashboard: @unchecked Sendable {
  private let stats: ServeStats
  private let header: [String]
  private let logLimit = 6

  private let lock = NSLock()
  private var logLines: [String] = []
  private var painted = 0
  private var stopped = false

  private var timer: DispatchSourceTimer?
  private var signals: [DispatchSourceSignal] = []
  private let queue = DispatchQueue(label: "ishizuki.dashboard")

  init(stats: ServeStats, header: [String]) {
    self.stats = stats
    self.header = header
  }

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
    lines.append("  " + Style.field("memory", Style.faint(ResidencyManager.describeMemory())))

    var conditions = [Style.faint("thermal \(Politeness.thermalDescription)")]
    if Politeness.isLowPowerMode { conditions.append(Style.warn("low power")) }
    conditions.append(Style.faint("up \(duration(totals.uptime))"))
    lines.append("  " + Style.field("state", conditions.joined(separator: Style.faint(" · "))))

    return lines
  }

  private func paint(_ lines: [String]) {
    var output = ""
    if painted > 0 { output += "\u{1B}[\(painted)A" }
    for line in lines { output += "\u{1B}[2K" + line + "\n" }
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

private func duration(_ seconds: Double) -> String {
  let total = Int(seconds)
  if total < 60 { return "\(total)s" }
  if total < 3600 { return "\(total / 60)m \(total % 60)s" }
  return "\(total / 3600)h \((total % 3600) / 60)m"
}
