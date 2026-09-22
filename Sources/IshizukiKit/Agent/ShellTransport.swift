// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// How a command gets to somewhere that is not this process: a container on this Mac, a pod on
// a cluster, a machine over ssh. Every one of them is the same thing — a script handed to a
// shell on the far side — so the agent above never learns which.

import Foundation

/// One way to reach a shell. A transport runs a whole script rather than an argument vector,
/// because that is the only thing every far side agrees on.
public protocol ExecTransport: Sendable {
  /// How the environment is named in a readout: "container ishizuki-a1b2", "pod agent-1".
  var describes: String { get }
  /// Whether the tool this transport needs is installed here. Not whether the far side is up.
  var isAvailable: Bool { get }
  /// Brings the far side up if it is not already. Called before the first command and cheap
  /// to call again.
  func prepare() async throws
  func exec(_ script: String, stdin: Data?, timeout: Double, byteLimit: Int) async throws
    -> ShellResult
  /// The same, when the answer is bytes rather than text.
  func capture(_ script: String, stdin: Data?, timeout: Double) async throws -> (
    data: Data, exitCode: Int32
  )
}

extension ExecTransport {
  public func prepare() async throws {}

  public func exec(_ script: String) async throws -> ShellResult {
    try await exec(script, stdin: nil, timeout: 120, byteLimit: 64 * 1024)
  }
}

public enum SandboxError: Error, LocalizedError, Sendable {
  case missingTool(String)
  case failed(String, String)

  public var errorDescription: String? {
    switch self {
    case .missingTool(let name):
      return "\(name) is not installed on this Mac"
    case .failed(let what, let why):
      return why.isEmpty ? "\(what) failed" : "\(what) failed: \(why)"
    }
  }
}

#if os(macOS)

  /// Where a command line tool is. The same search the rest of the app does, since a shipped
  /// app has no PATH worth the name.
  public enum Tooling {
    public static let searchPaths = [
      "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin",
    ]

    public static func locate(_ name: String) -> String? {
      searchPaths
        .map { $0 + "/" + name }
        .first { FileManager.default.isExecutableFile(atPath: $0) }
    }
  }

  /// A transport that spawns a local command and lets it carry the script the rest of the way.
  /// Every sandbox we have is one of these with a different prefix.
  public struct CommandTransport: ExecTransport {
    public var program: String
    public var prefix: [String]
    /// Whether the script needs quoting because a shell on the far side will parse the line
    /// again. True for ssh, false for anything that takes an argument vector.
    public var quotesScript: Bool
    public var describes: String

    public init(
      program: String, prefix: [String], quotesScript: Bool = false, describes: String
    ) {
      self.program = program
      self.prefix = prefix
      self.quotesScript = quotesScript
      self.describes = describes
    }

    public var isAvailable: Bool { FileManager.default.isExecutableFile(atPath: program) }

    public func exec(_ script: String, stdin: Data?, timeout: Double, byteLimit: Int)
      async throws -> ShellResult
    {
      try await ProcessRunner.run(
        argv: argv(for: script), stdin: stdin, timeout: timeout, byteLimit: byteLimit)
    }

    public func capture(_ script: String, stdin: Data?, timeout: Double) async throws -> (
      data: Data, exitCode: Int32
    ) {
      try await ProcessRunner.capture(
        argv: argv(for: script), stdin: stdin, timeout: timeout)
    }

    /// The script is the last argument, every time. What differs between a container, a pod
    /// and a machine over ssh is only what comes before it.
    private func argv(for script: String) -> [String] {
      [program] + prefix + [quotesScript ? shellQuoted(script) : script]
    }

    /// A shell on this Mac, reached the way a sandbox is. The spool, the cursors and the job
    /// ids are then exercised by the same code a container will run.
    public static func localShell() -> CommandTransport {
      CommandTransport(program: "/bin/sh", prefix: ["-c"], describes: "this Mac")
    }

    public static func kubectl(
      pod: String, namespace: String?, context: String?, container: String?
    ) throws -> CommandTransport {
      guard let tool = Tooling.locate("kubectl") else {
        throw SandboxError.missingTool("kubectl")
      }
      var prefix = ["exec", "-i"]
      if let context, !context.isEmpty { prefix += ["--context", context] }
      if let namespace, !namespace.isEmpty { prefix += ["--namespace", namespace] }
      prefix.append(pod)
      if let container, !container.isEmpty { prefix += ["--container", container] }
      prefix += ["--", "sh", "-c"]
      return CommandTransport(program: tool, prefix: prefix, describes: "pod \(pod)")
    }

    public static func ssh(destination: String, port: Int?, identity: String?) throws
      -> CommandTransport
    {
      guard let tool = Tooling.locate("ssh") else { throw SandboxError.missingTool("ssh") }
      var prefix = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
      if let port { prefix += ["-p", String(port)] }
      if let identity, !identity.isEmpty { prefix += ["-i", identity] }
      prefix += [destination, "sh", "-c"]
      return CommandTransport(
        program: tool, prefix: prefix, quotesScript: true, describes: "ssh \(destination)")
    }
  }

  /// Spawning, draining and timing out a local process. The one place that knows how to do it,
  /// so a transport is only ever an argument vector.
  public enum ProcessRunner {
    public static func run(
      argv: [String], stdin: Data?, timeout: Double, byteLimit: Int
    ) async throws -> ShellResult {
      let captured = try await pipeline(argv: argv, stdin: stdin, timeout: timeout)
      func text(_ data: Data) -> (String, Bool) {
        data.count > byteLimit
          ? (String(decoding: data.prefix(byteLimit), as: UTF8.self), true)
          : (String(decoding: data, as: UTF8.self), false)
      }
      let out = text(captured.stdout)
      let err = text(captured.stderr)
      return ShellResult(
        stdout: out.0, stderr: err.0, exitCode: captured.exitCode,
        truncated: out.1 || err.1)
    }

    public static func capture(argv: [String], stdin: Data?, timeout: Double) async throws -> (
      data: Data, exitCode: Int32
    ) {
      let captured = try await pipeline(argv: argv, stdin: stdin, timeout: timeout)
      return (captured.stdout, captured.exitCode)
    }

    private static func pipeline(argv: [String], stdin: Data?, timeout: Double) async throws -> (
      stdout: Data, stderr: Data, exitCode: Int32
    ) {
      guard let program = argv.first else { throw SandboxError.missingTool("nothing to run") }
      let process = Process()
      process.executableURL = URL(fileURLWithPath: program)
      process.arguments = Array(argv.dropFirst())

      var environment = ProcessInfo.processInfo.environment
      let path = environment["PATH"] ?? "/usr/bin:/bin"
      environment["PATH"] = (Tooling.searchPaths + [path]).joined(separator: ":")
      process.environment = environment

      let out = Pipe()
      let err = Pipe()
      let input = Pipe()
      process.standardOutput = out
      process.standardError = err
      process.standardInput = input

      try process.run()

      if let stdin {
        let handle = input.fileHandleForWriting
        DispatchQueue.global(qos: .utility).async {
          try? handle.write(contentsOf: stdin)
          try? handle.close()
        }
      } else {
        try? input.fileHandleForWriting.close()
      }

      async let stdout = drain(out)
      async let stderr = drain(err)

      let killer = Task {
        try await Task.sleep(for: .seconds(timeout))
        process.terminate()
      }
      let both = await (stdout, stderr)
      process.waitUntilExit()
      killer.cancel()

      return (both.0, both.1, process.terminationStatus)
    }

    private static func drain(_ pipe: Pipe) async -> Data {
      await withCheckedContinuation { continuation in
        DispatchQueue.global(qos: .utility).async {
          var data = Data()
          let handle = pipe.fileHandleForReading
          while case let chunk = handle.availableData, !chunk.isEmpty {
            data.append(chunk)
          }
          continuation.resume(returning: data)
        }
      }
    }
  }

#endif
