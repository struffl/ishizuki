// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A workspace on another machine, presented as the shell host the agent tools already expect.
// This is what lets a turn thought out on the phone change files on the Mac.

import Foundation
import IshizukiKit

public final class RemoteShellHost: ShellHost {
  public let workspace: URL
  private let client: LinkClient

  public init(client: LinkClient, workspace path: String) {
    self.client = client
    self.workspace = URL(filePath: path)
  }

  public var isAvailable: Bool { true }

  public func run(
    _ command: String, cwd: URL?, timeout: Double, byteLimit: Int
  ) async throws -> ShellResult {
    let outcome = try await client.shell(
      ShellRequest(command: command, cwd: cwd?.path, timeout: timeout))
    return ShellResult(
      stdout: outcome.stdout, stderr: outcome.stderr, exitCode: outcome.exitCode,
      truncated: outcome.truncated)
  }

  public func start(_ command: String, cwd: URL?) async throws -> ShellJob {
    try await client.start(ShellRequest(command: command, cwd: cwd?.path ?? workspace.path))
  }

  public func jobs() async throws -> [ShellJob] {
    try await client.jobs()
  }

  public func read(job id: String, wait: Double, byteLimit: Int) async throws -> ShellJobOutput {
    try await client.jobOutput(id, wait: wait, limit: byteLimit)
  }

  public func stop(job id: String, force: Bool) async throws -> ShellJob {
    try await client.stopJob(id, force: force)
  }
}
