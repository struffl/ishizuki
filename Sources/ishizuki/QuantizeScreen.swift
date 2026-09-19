// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Darwin
import Foundation
import IshizukiKit

/// A long job rendered in place: the same handful of lines repainted, never scrolled.
///
/// Quantizing a 52 GB checkpoint takes long enough that a scrolling log is unreadable and
/// tells you nothing about where it is. This keeps a fixed frame — phase, bar, elapsed, and
/// the last thing the worker said — so the screen is a status, not a history.
final class QuantizeScreen: @unchecked Sendable {
  struct Finished {
    let output: String
    let bytes: Int
    let seconds: Double
  }

  private let source: String
  private let destination: String
  private let profile: QuantProfile
  private let sourceBytes: Int
  private let estimateBytes: Int

  private var progress = Quantizer.Progress(
    phase: .scanning, done: 0, total: 1, detail: "", elements: 0, note: "")
  private var started = Date()
  private var phaseStarted = Date()
  private var lastPhase = Quantizer.Progress.Phase.scanning
  private var painted = 0
  private var active = false
  private let lock = NSLock()
  private var timer: DispatchSourceTimer?

  private(set) var finished: Finished?
  private(set) var failure: String?

  init(
    source: String, destination: String, profile: QuantProfile,
    sourceBytes: Int, estimateBytes: Int
  ) {
    self.source = source
    self.destination = destination
    self.profile = profile
    self.sourceBytes = sourceBytes
    self.estimateBytes = estimateBytes
  }

  var isSupported: Bool { isatty(STDOUT_FILENO) == 1 && Style.depth != .none }

  func begin() {
    lock.lock()
    defer { lock.unlock() }
    guard !active else { return }
    active = true
    started = Date()
    phaseStarted = started
    guard isSupported else {
      print(
        "quantizing \(source) -> \(destination) [\(profile.name), "
          + String(format: "~%.1f bpw]", profile.targetBpw))
      return
    }
    write("\u{1B}[?25l")
    render()
    // Elapsed has to keep moving even when the worker is quiet for minutes at a stretch,
    // otherwise a working process looks like a hung one.
    let timer = DispatchSource.makeTimerSource(queue: .global())
    timer.schedule(deadline: .now() + 1, repeating: 1)
    timer.setEventHandler { [weak self] in self?.tick() }
    timer.resume()
    self.timer = timer
  }

  func end() {
    lock.lock()
    defer { lock.unlock() }
    guard active else { return }
    active = false
    timer?.cancel()
    timer = nil
    guard isSupported else { return }
    render()
    write("\u{1B}[?25h")
  }

  func apply(_ update: Quantizer.Progress) {
    lock.lock()
    defer { lock.unlock() }
    if update.phase != lastPhase {
      lastPhase = update.phase
      phaseStarted = Date()
    }
    progress = update
    render()
  }

  func complete(_ outcome: Quantizer.Outcome) {
    lock.lock()
    defer { lock.unlock() }
    finished = Finished(
      output: outcome.directory.lastPathComponent, bytes: outcome.byteCount,
      seconds: outcome.seconds)
    render()
  }

  func fail(_ message: String) {
    lock.lock()
    defer { lock.unlock() }
    failure = message
    render()
  }

  private func tick() {
    lock.lock()
    defer { lock.unlock() }
    render()
  }

  // MARK: - Painting

  private func render() {
    guard isSupported else { return }
    let elapsed = -started.timeIntervalSinceNow
    let inPhase = -phaseStarted.timeIntervalSinceNow
    let width = terminalWidth()

    var lines: [String] = []
    lines.append(Style.banner("quantize"))
    lines.append("")
    lines.append(
      "  "
        + Style.field(
          "source",
          Style.accent(source)
            + (sourceBytes > 0 ? Style.faint("  " + Format.bytes(sourceBytes)) : "")))
    lines.append(
      "  "
        + Style.field(
          "profile",
          Style.accent(profile.name)
            + Style.faint(
              String(
                format: "  %d-bit base → %@  ·  target ~%.1f bpw", profile.baseBits,
                profile.boostBits.map(String.init).joined(separator: "/"),
                profile.targetBpw))))
    lines.append(
      "  "
        + Style.field(
          "output",
          Style.accent(destination)
            + (estimateBytes > 0 ? Style.faint("  ≈" + Format.bytes(estimateBytes)) : "")))
    lines.append("")
    lines.append("  " + bar(width: max(width - 22, 20)))
    lines.append("")

    // What it is doing and, for the long pass, why that takes as long as it does.
    lines.append(
      "  "
        + Style.field(
          "phase",
          (failure == nil ? Style.accent(progress.phase.rawValue) : Style.bad("failed"))
            + Style.faint("  \(progress.done)/\(progress.total) modules")
            + Style.faint("  ·  \(Self.duration(inPhase)) in phase")))
    lines.append(
      "  "
        + Style.field(
          "module", Style.faint(clip(progress.detail, to: max(width - 16, 20)))))

    var footer: [String] = ["\(Self.duration(elapsed)) elapsed"]
    if progress.elements > 0 {
      footer.append(String(format: "%.1fB weights read", Double(progress.elements) / 1e9))
      if inPhase > 1, progress.phase == .surveying {
        footer.append(
          String(format: "%.0fM weights/s", Double(progress.elements) / inPhase / 1e6))
      }
    }
    if progress.fraction > 0.01, progress.fraction < 1 {
      let total = inPhase / progress.fraction
      footer.append("eta \(Self.duration(max(total - inPhase, 0)))")
    }
    lines.append("  " + Style.field("", Style.faint(footer.joined(separator: "  ·  "))))

    if let failure {
      lines.append("  " + Style.field("error", Style.bad(clip(failure, to: width - 16))))
    } else if !progress.note.isEmpty {
      lines.append("  " + Style.field("", Style.faint(clip(progress.note, to: width - 16))))
    } else {
      lines.append("")
    }
    paint(lines)
  }

  private func bar(width: Int) -> String {
    let fraction = progress.fraction
    let filled = Int(fraction * Double(width))
    let body = String(repeating: "━", count: max(filled, 0))
    let rest = String(repeating: "╌", count: max(width - filled, 0))
    let colour = failure == nil ? Style.good(body) : Style.bad(body)
    return colour + Style.faint(rest)
      + Style.accent(String(format: "  %5.1f%%", fraction * 100))
  }

  static func duration(_ seconds: Double) -> String {
    let total = Int(seconds.rounded())
    if total >= 3600 { return String(format: "%dh%02dm", total / 3600, (total % 3600) / 60) }
    if total >= 60 { return String(format: "%dm%02ds", total / 60, total % 60) }
    return "\(total)s"
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

  private func terminalWidth() -> Int {
    var size = winsize()
    if ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &size) == 0, size.ws_col > 0 {
      return Int(size.ws_col)
    }
    return 100
  }

  private func write(_ text: String) {
    FileHandle.standardOutput.write(Data(text.utf8))
  }
}
