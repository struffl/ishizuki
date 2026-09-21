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
    let file = try read(path)
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

    let existing = try? read(path)
    try await ledger.checkWrite(
      path: display, fingerprint: existing?.fingerprint,
      lineCount: existing?.lines.count ?? 0)

    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(contents.utf8).write(to: url, options: .atomic)
    await ledger.invalidate(path: display)

    let lines = contents.components(separatedBy: "\n").count
    return "\(existing == nil ? "created" : "wrote") \(display), \(lines) lines"
  }

  public func edit(path: String, old: String, new: String, all: Bool = false) async throws
    -> String
  {
    let file = try read(path)
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

    try Data(updated.utf8).write(to: file.url, options: .atomic)
    await ledger.invalidate(path: file.display)

    let touched = Workspace.lineRange(of: first, in: contents)
    return "edited \(file.display):\(touched.lowerBound)"
      + (all && matches.count > 1 ? " and \(matches.count - 1) more" : "")
  }

  public func grep(
    pattern: String, glob: String? = nil, path: String? = nil, ignoreCase: Bool = false,
    limit: Int = 40
  ) async throws -> String {
    guard let rg = Ripgrep.locate() else { throw ShellError.missingProgram("rg") }
    let target = try path.map { try host.resolve($0) } ?? root

    var command = [rg, "--line-number", "--no-heading", "--color", "never", "--max-columns", "200"]
    if ignoreCase { command.append("--ignore-case") }
    if let glob { command += ["--glob", shellQuoted(glob)] }
    command += [shellQuoted(pattern), shellQuoted(target.path)]

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
    guard let rg = Ripgrep.locate() else { throw ShellError.missingProgram("rg") }
    let result = try await host.run(
      "\(rg) --files --glob \(shellQuoted(pattern))",
      cwd: root, timeout: 30, byteLimit: 256 * 1024)

    let paths = result.stdout.components(separatedBy: "\n").filter { !$0.isEmpty }
    guard !paths.isEmpty else { return "no files match \(pattern)" }

    var out = paths.prefix(limit).joined(separator: "\n")
    if paths.count > limit { out += "\n… \(paths.count - limit) more" }
    return out
  }

  public func shell(command: String, timeout: Double = 120, byteLimit: Int = 8 * 1024)
    async throws -> String
  {
    guard host.isAvailable else { throw ShellError.noHost }
    let result = try await host.run(
      command, cwd: root, timeout: timeout, byteLimit: byteLimit)

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
