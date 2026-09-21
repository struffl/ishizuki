// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Where a command actually runs. In the direct build that is this process; under the sandbox it
// has to be someone else, so nothing above this protocol is allowed to care which.

import Foundation

public struct ShellResult: Sendable {
  public var stdout: String
  public var stderr: String
  public var exitCode: Int32
  public var truncated: Bool

  public var succeeded: Bool { exitCode == 0 }

  public init(stdout: String, stderr: String, exitCode: Int32, truncated: Bool = false) {
    self.stdout = stdout
    self.stderr = stderr
    self.exitCode = exitCode
    self.truncated = truncated
  }
}

public enum ShellError: Error, LocalizedError, Sendable {
  case outsideWorkspace(String)
  case noHost
  case missingProgram(String)
  case timedOut(Double)

  public var errorDescription: String? {
    switch self {
    case .outsideWorkspace(let path):
      return "\(path) is outside the workspace"
    case .noHost:
      return "no shell is attached to this workspace"
    case .missingProgram(let name):
      return "\(name) was not found"
    case .timedOut(let seconds):
      return String(format: "timed out after %.0fs", seconds)
    }
  }
}

public protocol ShellHost: Sendable {
  /// The one directory the agent is allowed to touch. Every path a tool takes resolves against
  /// it and is refused if it lands outside.
  var workspace: URL { get }
  var isAvailable: Bool { get }

  func run(
    _ command: String, cwd: URL?, timeout: Double, byteLimit: Int
  ) async throws -> ShellResult
}

extension ShellHost {
  public func run(_ command: String) async throws -> ShellResult {
    try await run(command, cwd: nil, timeout: 120, byteLimit: 64 * 1024)
  }

  /// Resolves a path the model handed over, and refuses anything that leaves the workspace —
  /// after following symlinks, since a link is the easy way out of a directory.
  public func resolve(_ path: String) throws -> URL {
    let expanded = (path as NSString).expandingTildeInPath
    let candidate =
      expanded.hasPrefix("/")
      ? URL(fileURLWithPath: expanded)
      : workspace.appending(path: expanded)

    let root = workspace.resolvingSymlinksInPath().standardizedFileURL.path
    let resolved = candidate.resolvingSymlinksInPath().standardizedFileURL.path
    guard resolved == root || resolved.hasPrefix(root + "/") else {
      throw ShellError.outsideWorkspace(path)
    }
    return candidate.standardizedFileURL
  }

  /// How a path is spelled back to the model: relative to the workspace, so the transcript does
  /// not carry a home directory on every line.
  public func display(_ url: URL) -> String {
    let root = workspace.standardizedFileURL.path
    let path = url.standardizedFileURL.path
    guard path.hasPrefix(root + "/") else { return path }
    return String(path.dropFirst(root.count + 1))
  }
}

#if os(macOS)

  /// Runs commands in this process. Correct for a Developer ID build; under the App Sandbox the
  /// child inherits the sandbox and only the container is reachable. There is no process to spawn
  /// on iOS, where a workspace is reached over the link instead.
  public final class LocalShellHost: ShellHost {
    public let workspace: URL
    public let shell: String
    public let extraPaths: [String]

    public init(
      workspace: URL, shell: String = "/bin/zsh",
      extraPaths: [String] = ["/opt/homebrew/bin", "/usr/local/bin"]
    ) {
      self.workspace = workspace
      self.shell = shell
      self.extraPaths = extraPaths
    }

    public var isAvailable: Bool { FileManager.default.isExecutableFile(atPath: shell) }

    public func run(
      _ command: String, cwd: URL?, timeout: Double, byteLimit: Int
    ) async throws -> ShellResult {
      guard isAvailable else { throw ShellError.missingProgram(shell) }
      let directory = try cwd.map { try resolve($0.path) } ?? workspace

      let process = Process()
      process.executableURL = URL(fileURLWithPath: shell)
      process.arguments = ["-l", "-c", command]
      process.currentDirectoryURL = directory

      var environment = ProcessInfo.processInfo.environment
      let path = environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
      environment["PATH"] = (extraPaths + [path]).joined(separator: ":")
      environment["TERM"] = "dumb"
      process.environment = environment

      let out = Pipe()
      let err = Pipe()
      process.standardOutput = out
      process.standardError = err

      let expiry = Expiry()
      try process.run()

      // Both pipes drain at once: draining one to completion first lets the child block on the
      // other's full buffer, and neither of us would ever move again.
      async let stdout = Self.drain(out, limit: byteLimit)
      async let stderr = Self.drain(err, limit: byteLimit)

      let deadline = Task {
        try await Task.sleep(for: .seconds(timeout))
        expiry.expire()
        process.terminate()
      }

      let captured = await (stdout, stderr)
      process.waitUntilExit()
      deadline.cancel()

      if expiry.hasExpired { throw ShellError.timedOut(timeout) }

      return ShellResult(
        stdout: captured.0.text,
        stderr: captured.1.text,
        exitCode: process.terminationStatus,
        truncated: captured.0.truncated || captured.1.truncated)
    }

    private struct Capture: Sendable {
      var text: String
      var truncated: Bool
    }

    /// Reads a pipe to a ceiling, because a runaway build log is not something a 27B model on a
    /// laptop should be made to read. Reading continues past the ceiling and is thrown away, so
    /// a chatty command is never left blocked on a pipe nobody is emptying.
    private static func drain(_ pipe: Pipe, limit: Int) async -> Capture {
      await withCheckedContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async {
          var data = Data()
          var truncated = false
          let handle = pipe.fileHandleForReading
          while let chunk = try? handle.read(upToCount: 16 * 1024), !chunk.isEmpty {
            if data.count < limit {
              data.append(chunk)
            } else {
              truncated = true
            }
          }
          if data.count > limit {
            data = data.prefix(limit)
            truncated = true
          }
          continuation.resume(
            returning: Capture(
              text: String(decoding: data, as: UTF8.self), truncated: truncated))
        }
      }
    }
  }

  /// One bit, shared between the command and the clock that may cut it short.
  private final class Expiry: @unchecked Sendable {
    private let lock = NSLock()
    private var expired = false

    var hasExpired: Bool {
      lock.lock()
      defer { lock.unlock() }
      return expired
    }

    func expire() {
      lock.lock()
      expired = true
      lock.unlock()
    }
  }

#endif
