// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What the chosen folder is, as git sees it: which branch, how far from its upstream, what is
// uncommitted, and which worktrees hang off it.

import Foundation
import IshizukiKit
import Observation

/// A folder's standing in git, read in one call so the strip above the composer never shows
/// half a picture.
struct GitStatus: Equatable, Sendable {
  var root: URL
  var branch: String
  /// A head that is not on a branch, which is worth saying rather than showing a bare hash.
  var detached: Bool
  var ahead: Int
  var behind: Int
  var hasUpstream: Bool
  /// Tracked files with changes, staged or not.
  var changed: Int
  var untracked: Int
  var conflicted: Int
  /// Whether this folder is a linked worktree rather than the repository's own.
  var isLinkedWorktree: Bool

  var isClean: Bool { changed == 0 && untracked == 0 && conflicted == 0 }

  var pendingCount: Int { changed + untracked + conflicted }

  /// The short form the strip draws: branch first, then only what is actually true of it.
  var summary: String {
    var parts = [detached ? "detached" : branch]
    if behind > 0 { parts.append("↓\(behind)") }
    if ahead > 0 { parts.append("↑\(ahead)") }
    if pendingCount > 0 { parts.append("•\(pendingCount)") }
    return parts.joined(separator: " ")
  }
}

/// One entry of `git worktree list`: where it is and what it has checked out.
struct GitWorktree: Identifiable, Equatable, Sendable {
  var path: URL
  var branch: String?
  var isPrimary: Bool

  var id: String { path.path }
  var name: String { path.lastPathComponent }
}

/// Git as a handful of one-shot commands. Everything here runs off the main actor, because a
/// status call on a cold repository is tens of milliseconds and the window is drawing tokens.
enum Git {
  static let executable = "/usr/bin/git"

  /// Where ishizuki puts the worktrees it makes, kept outside the repository so a checkout it
  /// created is never mistaken for the user's own working copy.
  static var worktreeRoot: URL {
    IshizukiPaths.applicationSupport.appending(path: "Ishizuki/worktrees")
  }

  static func isRepository(_ folder: URL) -> Bool {
    root(of: folder) != nil
  }

  /// The top of the working tree a folder belongs to, or nil when it belongs to none.
  static func root(of folder: URL) -> URL? {
    guard let out = run(["rev-parse", "--show-toplevel"], in: folder), !out.isEmpty else {
      return nil
    }
    return URL(filePath: out)
  }

  static func status(of folder: URL) -> GitStatus? {
    guard let root = root(of: folder) else { return nil }
    guard
      let out = run(
        ["status", "--porcelain=v2", "--branch", "--untracked-files=normal"], in: root)
    else { return nil }

    var branch = "—"
    var detached = false
    var hasUpstream = false
    var ahead = 0
    var behind = 0
    var changed = 0
    var untracked = 0
    var conflicted = 0

    for line in out.split(separator: "\n", omittingEmptySubsequences: true) {
      if line.hasPrefix("# branch.head ") {
        let name = String(line.dropFirst("# branch.head ".count))
        detached = name == "(detached)"
        branch = detached ? "detached" : name
      } else if line.hasPrefix("# branch.upstream ") {
        hasUpstream = true
      } else if line.hasPrefix("# branch.ab ") {
        for field in line.dropFirst("# branch.ab ".count).split(separator: " ") {
          let value = Int(field.dropFirst()) ?? 0
          if field.hasPrefix("+") { ahead = value }
          if field.hasPrefix("-") { behind = value }
        }
      } else if line.hasPrefix("1 ") || line.hasPrefix("2 ") {
        changed += 1
      } else if line.hasPrefix("u ") {
        conflicted += 1
      } else if line.hasPrefix("? ") {
        untracked += 1
      }
    }

    let common = run(["rev-parse", "--git-common-dir"], in: root) ?? ""
    let own = run(["rev-parse", "--absolute-git-dir"], in: root) ?? ""
    let linked =
      !common.isEmpty && !own.isEmpty
      && URL(filePath: common).standardizedFileURL
        != URL(filePath: own).standardizedFileURL

    return GitStatus(
      root: root, branch: branch, detached: detached, ahead: ahead, behind: behind,
      hasUpstream: hasUpstream, changed: changed, untracked: untracked,
      conflicted: conflicted, isLinkedWorktree: linked)
  }

  static func worktrees(of folder: URL) -> [GitWorktree] {
    guard let root = root(of: folder),
      let out = run(["worktree", "list", "--porcelain"], in: root)
    else { return [] }

    var result: [GitWorktree] = []
    var path: URL?
    var branch: String?

    func flush() {
      guard let path else { return }
      result.append(
        GitWorktree(path: path, branch: branch, isPrimary: result.isEmpty))
      branch = nil
    }

    for line in out.split(separator: "\n", omittingEmptySubsequences: false) {
      if line.hasPrefix("worktree ") {
        flush()
        path = URL(filePath: String(line.dropFirst("worktree ".count)))
      } else if line.hasPrefix("branch ") {
        branch = String(line.dropFirst("branch ".count))
          .replacingOccurrences(of: "refs/heads/", with: "")
      }
    }
    flush()
    return result
  }

  /// A fresh worktree off this repository, on a branch of its own. The branch is reused when it
  /// already exists, so asking twice for the same name checks it out rather than failing.
  static func addWorktree(of folder: URL, named name: String) throws -> GitWorktree {
    guard let root = root(of: folder) else {
      throw GitFailure("\(folder.lastPathComponent) is not a git repository")
    }
    let slug = slugify(name)
    let branch = "ishizuki/\(slug)"
    let destination =
      worktreeRoot
      .appending(path: root.lastPathComponent)
      .appending(path: slug)

    try FileManager.default.createDirectory(
      at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

    if FileManager.default.fileExists(atPath: destination.path) {
      throw GitFailure("a worktree is already checked out at \(destination.lastPathComponent)")
    }

    let existing = run(["rev-parse", "--verify", "--quiet", branch], in: root)
    let arguments =
      (existing?.isEmpty == false)
      ? ["worktree", "add", destination.path, branch]
      : ["worktree", "add", "-b", branch, destination.path]

    let outcome = capture(arguments, in: root)
    guard outcome.code == 0 else {
      throw GitFailure(
        outcome.error.isEmpty ? "git worktree add failed" : outcome.error)
    }
    return GitWorktree(path: destination, branch: branch, isPrimary: false)
  }

  static func removeWorktree(_ worktree: GitWorktree, of folder: URL) throws {
    guard let root = root(of: folder) else {
      throw GitFailure("\(folder.lastPathComponent) is not a git repository")
    }
    let outcome = capture(["worktree", "remove", "--force", worktree.path.path], in: root)
    guard outcome.code == 0 else {
      throw GitFailure(outcome.error.isEmpty ? "git worktree remove failed" : outcome.error)
    }
  }

  /// A branch name from whatever someone typed: git refuses most of what a chat title contains.
  static func slugify(_ name: String) -> String {
    let lowered = name.lowercased()
    var out = ""
    var lastWasDash = false
    for character in lowered {
      if character.isLetter || character.isNumber {
        out.append(character)
        lastWasDash = false
      } else if !lastWasDash, !out.isEmpty {
        out.append("-")
        lastWasDash = true
      }
    }
    while out.hasSuffix("-") { out.removeLast() }
    let trimmed = String(out.prefix(40))
    return trimmed.isEmpty ? "work-\(UUID().uuidString.prefix(6))" : trimmed
  }

  private static func run(_ arguments: [String], in folder: URL) -> String? {
    let outcome = capture(arguments, in: folder)
    guard outcome.code == 0 else { return nil }
    return outcome.output.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  private static func capture(
    _ arguments: [String], in folder: URL
  ) -> (output: String, error: String, code: Int32) {
    guard FileManager.default.isExecutableFile(atPath: executable) else {
      return ("", "git is not installed", 127)
    }
    let process = Process()
    process.executableURL = URL(filePath: executable)
    process.arguments = arguments
    process.currentDirectoryURL = folder

    var environment = ProcessInfo.processInfo.environment
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    process.environment = environment

    let out = Pipe()
    let err = Pipe()
    process.standardOutput = out
    process.standardError = err

    do {
      try process.run()
    } catch {
      return ("", String(describing: error), 127)
    }
    let outData = (try? out.fileHandleForReading.readToEnd()) ?? Data()
    let errData = (try? err.fileHandleForReading.readToEnd()) ?? Data()
    process.waitUntilExit()

    return (
      String(decoding: outData, as: UTF8.self),
      String(decoding: errData, as: UTF8.self)
        .trimmingCharacters(in: .whitespacesAndNewlines),
      process.terminationStatus
    )
  }
}

struct GitFailure: Error, LocalizedError {
  let message: String

  init(_ message: String) {
    self.message = message
  }

  var errorDescription: String? { message }
}

/// Watches one folder's git state on a slow tick, so the strip stays current without a command
/// per keystroke. A folder that is not a repository parks the poll rather than retrying it.
@MainActor
@Observable
final class GitProbe {
  private(set) var status: GitStatus?
  private(set) var worktrees: [GitWorktree] = []

  @ObservationIgnored private var folder: URL?
  @ObservationIgnored private var ticker: Task<Void, Never>?

  var isRepository: Bool { status != nil }

  func watch(_ url: URL?) {
    guard url != folder else { return }
    folder = url
    status = nil
    worktrees = []
    ticker?.cancel()
    guard let url else { return }
    ticker = Task { [weak self] in
      while !Task.isCancelled {
        await self?.reload(url)
        try? await Task.sleep(for: .seconds(4))
      }
    }
  }

  func refresh() {
    guard let folder else { return }
    Task { await reload(folder) }
  }

  private func reload(_ url: URL) async {
    let read = await Task.detached(priority: .utility) {
      (Git.status(of: url), Git.worktrees(of: url))
    }.value
    guard folder == url else { return }
    if status != read.0 { status = read.0 }
    if worktrees != read.1 { worktrees = read.1 }
  }
}
