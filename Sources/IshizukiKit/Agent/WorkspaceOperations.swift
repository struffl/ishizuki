// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What the tools actually do. Kept clear of Foundation Models so the same operations serve a
// build that has no session to hang them off, and so they can be tested on their own.

import Foundation

extension Workspace {
  /// A slice, numbered, with a line saying what was withheld and where to pick it up.
  public func readSlice(path: String, offset: Int? = nil, limit: Int? = nil) async throws
    -> String
  {
    let file = try await read(path)
    let total = file.lines.count
    let start = max(1, offset ?? 1)
    guard start <= total else {
      return "\(file.display) has \(total) lines; \(start) is past the end"
    }
    let count = max(1, limit ?? sliceLines)
    let end = min(total, start + count - 1)

    await ledger.noteRead(
      path: file.display, fingerprint: file.fingerprint, lineCount: total, slice: start...end)

    var out = "\(file.display) \(start)-\(end)/\(total)\n"
    for number in start...end {
      out += "\(number)→\(file.lines[number - 1])\n"
    }
    if end < total {
      out += "… \(total - end) more (offset \(end + 1))"
    }
    return out
  }

  public func writeWhole(path: String, contents: String) async throws -> String {
    let url = try host.resolve(path)
    let display = host.display(url)

    let existing = try? await read(path)
    try await ledger.checkWrite(
      path: display, fingerprint: existing?.fingerprint,
      lineCount: existing?.lines.count ?? 0)

    try await host.write(Data(contents.utf8), to: url)
    await ledger.invalidate(path: display)

    let lines = contents.components(separatedBy: "\n").count
    return "\(existing == nil ? "created" : "wrote") \(display), \(lines) lines"
  }

  public func edit(path: String, old: String, new: String, all: Bool = false) async throws
    -> String
  {
    let file = try await read(path)
    let contents = file.lines.joined(separator: "\n")

    guard !old.isEmpty else {
      throw LedgerRefusal(message: "refused: old is empty; use write to create a file")
    }

    var matches: [Range<String.Index>] = []
    var cursor = contents.startIndex
    while let found = contents.range(of: old, range: cursor..<contents.endIndex) {
      matches.append(found)
      cursor = found.upperBound
    }

    guard let first = matches.first else {
      throw LedgerRefusal(
        message: "no match in \(file.display). Read the slice you mean and copy it exactly.")
    }
    guard all || matches.count == 1 else {
      throw LedgerRefusal(
        message: "\(matches.count) matches in \(file.display); lengthen old or pass all")
    }

    for match in (all ? matches : [first]) {
      try await ledger.checkEdit(
        path: file.display, fingerprint: file.fingerprint,
        lines: Workspace.lineRange(of: match, in: contents))
    }

    let updated =
      all
      ? contents.replacingOccurrences(of: old, with: new)
      : contents.replacingCharacters(in: first, with: new)

    try await host.write(Data(updated.utf8), to: file.url)
    await ledger.invalidate(path: file.display)

    let touched = Workspace.lineRange(of: first, in: contents)
    return "edited \(file.display):\(touched.lowerBound)"
      + (all && matches.count > 1 ? " and \(matches.count - 1) more" : "")
  }

  public func grep(
    pattern: String, glob: String? = nil, path: String? = nil, ignoreCase: Bool = false,
    limit: Int = 40
  ) async throws -> String {
    let target = try path.map { try host.resolve($0) } ?? root
    let rg = await searchProgram()

    var command: [String]
    if let rg {
      command = [rg, "--line-number", "--no-heading", "--color", "never", "--max-columns", "200"]
      if ignoreCase { command.append("--ignore-case") }
      if let glob { command += ["--glob", shellQuoted(glob)] }
      command += [shellQuoted(pattern), shellQuoted(target.path)]
    } else {
      // Nothing to install into a sandbox: plain grep says the same thing, more slowly.
      command = ["grep", "-r", "-n", "-I", "-E"]
      if ignoreCase { command.append("-i") }
      if let glob { command += ["--include", shellQuoted(glob)] }
      command += ["-e", shellQuoted(pattern), shellQuoted(target.path)]
    }

    let result = try await host.run(
      command.joined(separator: " "), cwd: root, timeout: 30, byteLimit: 256 * 1024)

    // rg exits 1 on no matches, which is an answer rather than a failure.
    guard result.exitCode != 1 else { return "no matches" }
    guard result.succeeded else {
      throw LedgerRefusal(message: result.stderr.isEmpty ? "grep failed" : result.stderr)
    }

    let prefix = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
    let hits =
      result.stdout
      .components(separatedBy: "\n")
      .filter { !$0.isEmpty }
      .map { $0.hasPrefix(prefix) ? String($0.dropFirst(prefix.count)) : $0 }

    guard !hits.isEmpty else { return "no matches" }
    var out = hits.prefix(limit).joined(separator: "\n")
    if hits.count > limit {
      out += "\n… \(hits.count - limit) more matches; narrow the pattern"
    }
    return out
  }

  public func glob(pattern: String, limit: Int = 60) async throws -> String {
    let command =
      if let rg = await searchProgram() {
        "\(rg) --files --glob \(shellQuoted(pattern))"
      } else {
        "find . -type f | sed 's|^\\./||' | grep -E \(shellQuoted(Workspace.regex(for: pattern)))"
      }
    let result = try await host.run(
      command, cwd: root, timeout: 30, byteLimit: 256 * 1024)

    let paths = result.stdout.components(separatedBy: "\n").filter { !$0.isEmpty }
    guard !paths.isEmpty else { return "no files match \(pattern)" }

    var out = paths.prefix(limit).joined(separator: "\n")
    if paths.count > limit { out += "\n… \(paths.count - limit) more" }
    return out
  }

  /// Runs a command and waits a short while for it. A command that is still going when the
  /// wait runs out is not killed: it is left running and answered for with a job id, which is
  /// the only way a build or a server belongs in a turn.
  public func shell(
    command: String, timeout: Double = 15, byteLimit: Int = 8 * 1024, background: Bool = false
  ) async throws -> String {
    guard host.isAvailable else { throw ShellError.noHost }

    let job: ShellJob
    do {
      job = try await host.start(command, cwd: root)
    } catch ShellError.noJobs {
      return try await foreground(command: command, timeout: timeout, byteLimit: byteLimit)
    }

    if background {
      return
        "\(job.id) started: \(command)\nRead it with output \(job.id), stop it with kill \(job.id)."
    }

    let seen = try await host.read(job: job.id, wait: max(0, timeout), byteLimit: byteLimit)
    guard seen.job.isRunning else { return settled(seen) }

    var out = "\(job.id) is still running after \(whole(timeout))s, so it was left in the "
    out += "background: \(command)"
    let written = body(of: seen)
    if !written.isEmpty { out += "\n\(written)" }
    out += "\nRead the rest with output \(job.id), stop it with kill \(job.id)."
    return out
  }

  /// Every job this workspace has going, and the ones that have finished with output nobody
  /// has read yet.
  public func jobs() async throws -> String {
    let jobs = try await host.jobs()
    guard !jobs.isEmpty else { return "no background jobs" }
    return jobs.map(line(for:)).joined(separator: "\n")
  }

  public func jobOutput(_ id: String, wait: Double = 0, byteLimit: Int = 8 * 1024) async throws
    -> String
  {
    let seen = try await host.read(job: id, wait: min(max(0, wait), 120), byteLimit: byteLimit)
    guard seen.job.isRunning else { return settled(seen) }
    let written = body(of: seen)
    return written.isEmpty
      ? "\(id) is still running, \(whole(seen.job.seconds))s in, with nothing new to show"
      : "\(id) is still running, \(whole(seen.job.seconds))s in\n\(written)"
  }

  public func killJob(_ id: String, force: Bool = false) async throws -> String {
    let job = try await host.stop(job: id, force: force)
    guard job.isRunning else { return "\(id) had already finished" }
    return force
      ? "\(id) was killed after \(whole(job.seconds))s"
      : "\(id) was asked to stop after \(whole(job.seconds))s; it is killed if it stays up"
  }

  /// Ripgrep where the workspace actually is, which on a sandbox is a question for the
  /// sandbox rather than for this Mac.
  private func searchProgram() async -> String? {
    await host.locate("rg")
  }

  /// A glob as a regular expression, for a host with no ripgrep to read it for us.
  static func regex(for pattern: String) -> String {
    var out = "^"
    var rest = Substring(pattern)
    while let next = rest.first {
      switch next {
      case "*":
        if rest.hasPrefix("**/") {
          out += "(.*/)?"
          rest = rest.dropFirst(3)
          continue
        }
        if rest.hasPrefix("**") {
          out += ".*"
          rest = rest.dropFirst(2)
          continue
        }
        out += "[^/]*"
      case "?": out += "[^/]"
      case ".", "+", "(", ")", "|", "[", "]", "{", "}", "^", "$", "\\":
        out += "\\" + String(next)
      default: out.append(next)
      }
      rest = rest.dropFirst()
    }
    return out + "$"
  }

  /// Stops everything this workspace still has running, which is what a conversation being
  /// thrown away owes the machine it was working on.
  public func stopAllJobs() async {
    guard let running = try? await host.jobs() else { return }
    for job in running where job.isRunning {
      _ = try? await host.stop(job: job.id, force: true)
    }
  }

  private func foreground(command: String, timeout: Double, byteLimit: Int) async throws -> String {
    let result = try await host.run(
      command, cwd: root, timeout: max(timeout, 120), byteLimit: byteLimit)
    var out = ""
    if !result.stdout.isEmpty { out += result.stdout }
    if !result.stderr.isEmpty {
      if !out.isEmpty { out += "\n" }
      out += result.stderr
    }
    if result.truncated { out += "\n… output truncated" }
    if !result.succeeded { out += "\nexit \(result.exitCode)" }
    return out.isEmpty ? "exit \(result.exitCode), no output" : out
  }

  private func settled(_ output: ShellJobOutput) -> String {
    var out = body(of: output)
    let code = output.job.exitCode ?? 0
    if output.job.state == .killed {
      out += out.isEmpty ? "" : "\n"
      out += "\(output.job.id) was stopped after \(whole(output.job.seconds))s"
      return out
    }
    if code != 0 {
      out += out.isEmpty ? "" : "\n"
      out += "exit \(code)"
    }
    return out.isEmpty ? "exit \(code), no output" : out
  }

  private func body(of output: ShellJobOutput) -> String {
    var out = output.stdout
    if !output.stderr.isEmpty {
      if !out.isEmpty, !out.hasSuffix("\n") { out += "\n" }
      out += output.stderr
    }
    if output.skipped > 0 { out += "\n… \(size(output.skipped)) of earlier output was dropped" }
    if output.remaining > 0 { out += "\n… \(size(output.remaining)) more waiting" }
    return out
  }

  private func line(for job: ShellJob) -> String {
    var out = job.id + "  "
    switch job.state {
    case .running: out += "running \(whole(job.seconds))s"
    case .exited: out += "exit \(job.exitCode ?? 0) after \(whole(job.seconds))s"
    case .killed: out += "stopped after \(whole(job.seconds))s"
    }
    out += "  \(job.command)"
    if job.pending > 0 { out += "  (\(size(job.pending)) unread)" }
    return out
  }

  private func whole(_ seconds: Double) -> String {
    String(Int(seconds.rounded()))
  }

  private func size(_ bytes: Int) -> String {
    bytes < 1024 ? "\(bytes) B" : String(format: "%.1f KB", Double(bytes) / 1024)
  }
}

/// Where `rg` is. A bundled copy wins, since a shipped app cannot count on a Homebrew install.
public enum Ripgrep {
  public static let candidates = [
    "/opt/homebrew/bin/rg", "/usr/local/bin/rg", "/usr/bin/rg",
  ]

  public static func locate(
    bundled: URL? = Bundle.main.url(forAuxiliaryExecutable: "rg")
  ) -> String? {
    if let bundled, FileManager.default.isExecutableFile(atPath: bundled.path) {
      return bundled.path
    }
    return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
  }
}

/// Quoting is ours to do, because a host runs a command line rather than an argument vector.
func shellQuoted(_ value: String) -> String {
  "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}
