// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

/// The sandbox host, driven against a shell on this Mac. Every script here is the one a VM or
/// a pod would run, so the spool is tested without one being up.
@Suite("Sandbox shell")
struct SpooledShellTests {
  private func sandbox() throws -> (Workspace, SpooledShellHost, URL) {
    let root = URL(filePath: NSTemporaryDirectory())
      .appending(path: "spool-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let host = SpooledShellHost(
      workspace: root, transport: CommandTransport.localShell(),
      spool: root.appending(path: ".jobs").path)
    return (Workspace(host: host, sliceLines: 3), host, root)
  }

  @Test("a command runs where the workspace is and its output comes back")
  func runs() async throws {
    let (workspace, _, root) = try sandbox()
    try Data("hello".utf8).write(to: root.appending(path: "a.txt"))

    let out = try await workspace.shell(command: "ls", timeout: 20)
    #expect(out.contains("a.txt"))

    let failed = try await workspace.shell(command: "exit 3", timeout: 20)
    #expect(failed.contains("exit 3"))
  }

  @Test("files are read and written through the sandbox, not around it")
  func files() async throws {
    let (workspace, host, root) = try sandbox()

    let made = try await workspace.writeWhole(path: "notes/b.txt", contents: "one\ntwo\n")
    #expect(made.contains("created notes/b.txt"))

    let onDisk = try String(
      contentsOf: root.appending(path: "notes/b.txt"), encoding: .utf8)
    #expect(onDisk == "one\ntwo\n")

    let slice = try await workspace.readSlice(path: "notes/b.txt")
    #expect(slice.contains("1→one"))

    #expect(try await host.contents(at: root.appending(path: "nope.txt")) == nil)
  }

  @Test("a path that leaves the workspace is refused without a filesystem to ask")
  func staysInside() async throws {
    let (_, host, _) = try sandbox()
    #expect(throws: ShellError.self) { try host.resolve("../escape") }
    #expect(throws: ShellError.self) { try host.resolve("/etc/passwd") }
    #expect(throws: ShellError.self) { try host.resolve("~/secrets") }
    #expect(throws: Never.self) { try host.resolve("Sources/deep/./file.swift") }
  }

  @Test("a job can be killed in the sandbox")
  func killed() async throws {
    let (workspace, _, _) = try sandbox()
    let started = try await workspace.shell(command: "sleep 30", background: true)
    #expect(started.contains("job1 started"))

    _ = try await workspace.killJob("job1", force: true)
    try await Task.sleep(for: .milliseconds(400))
    let after = try await workspace.jobOutput("job1")
    #expect(after.contains("was stopped") || after.contains("exit"))
  }

  @Test("a glob reads as a regular expression for a sandbox with no ripgrep")
  func globRegex() {
    #expect(Workspace.regex(for: "Sources/**/*.swift") == "^Sources/(.*/)?[^/]*\\.swift$")
    #expect(Workspace.regex(for: "*.md") == "^[^/]*\\.md$")
  }
}
