// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

@Suite("Agent shell")
struct AgentShellTests {
  private func sandbox() throws -> (Workspace, URL) {
    let root = URL(filePath: NSTemporaryDirectory())
      .appending(path: "agent-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let host = LocalShellHost(workspace: root)
    return (Workspace(host: host, sliceLines: 3), root)
  }

  private func seed(_ root: URL, _ name: String, lines: Int) throws -> URL {
    let url = root.appending(path: name)
    let body = (1...lines).map { "line \($0)" }.joined(separator: "\n")
    try Data(body.utf8).write(to: url)
    return url
  }

  @Test("a read hands back a slice and says what it withheld")
  func readSlices() async throws {
    let (workspace, root) = try sandbox()
    _ = try seed(root, "a.txt", lines: 10)

    let out = try await workspace.readSlice(path: "a.txt")

    #expect(out.hasPrefix("a.txt 1-3/10"))
    #expect(out.contains("1→line 1"))
    #expect(out.contains("3→line 3"))
    #expect(!out.contains("4→line 4"))
    #expect(out.contains("… 7 more (offset 4)"))
  }

  @Test("an unread file cannot be overwritten")
  func writeNeedsARead() async throws {
    let (workspace, root) = try sandbox()
    _ = try seed(root, "a.txt", lines: 4)

    await #expect(throws: LedgerRefusal.self) {
      try await workspace.writeWhole(path: "a.txt", contents: "replaced")
    }
  }

  @Test("a new file needs no read")
  func writeCreates() async throws {
    let (workspace, _) = try sandbox()
    let out = try await workspace.writeWhole(path: "new/b.txt", contents: "hello")
    #expect(out.contains("created new/b.txt"))
  }

  @Test("a partial read is not enough to overwrite, but is enough to edit its own lines")
  func sliceLetsEdit() async throws {
    let (workspace, root) = try sandbox()
    _ = try seed(root, "a.txt", lines: 10)
    _ = try await workspace.readSlice(path: "a.txt", offset: 1, limit: 3)

    await #expect(throws: LedgerRefusal.self) {
      try await workspace.writeWhole(path: "a.txt", contents: "all gone")
    }

    let edited = try await workspace.edit(path: "a.txt", old: "line 2", new: "line two")
    #expect(edited == "edited a.txt:2")
    let body = try String(contentsOf: root.appending(path: "a.txt"), encoding: .utf8)
    #expect(body.contains("line two"))
  }

  @Test("an edit to lines that were never read is refused")
  func editNeedsThoseLines() async throws {
    let (workspace, root) = try sandbox()
    _ = try seed(root, "a.txt", lines: 10)

    _ = try await workspace.readSlice(path: "a.txt", offset: 1, limit: 3)

    await #expect(throws: LedgerRefusal.self) {
      try await workspace.edit(path: "a.txt", old: "line 9", new: "nine")
    }
  }

  @Test("a file that moved under the model is refused until it is read again")
  func staleReadIsRefused() async throws {
    let (workspace, root) = try sandbox()
    let url = try seed(root, "a.txt", lines: 3)

    _ = try await workspace.readSlice(path: "a.txt", offset: 1, limit: 3)
    try Data("line 1\nline 2\nline 3\nline 4".utf8).write(to: url)

    await #expect(throws: LedgerRefusal.self) {
      try await workspace.edit(path: "a.txt", old: "line 2", new: "two")
    }
  }

  @Test("paths outside the workspace are refused, symlinks included")
  func staysInside() async throws {
    let (workspace, root) = try sandbox()
    let outside = URL(filePath: NSTemporaryDirectory())
      .appending(path: "outside-\(UUID().uuidString).txt")
    try Data("secret".utf8).write(to: outside)
    try FileManager.default.createSymbolicLink(
      at: root.appending(path: "link.txt"), withDestinationURL: outside)

    #expect(throws: ShellError.self) { try workspace.host.resolve("../escape") }
    #expect(throws: ShellError.self) { try workspace.host.resolve(outside.path) }
    #expect(throws: ShellError.self) { try workspace.host.resolve("link.txt") }
  }

  @Test("a command runs in the workspace and its output comes back")
  func shellRuns() async throws {
    let (workspace, root) = try sandbox()
    _ = try seed(root, "a.txt", lines: 2)

    let out = try await workspace.shell(command: "ls", timeout: 30)
    #expect(out.contains("a.txt"))

    let failed = try await workspace.shell(command: "exit 3", timeout: 30)
    #expect(failed.contains("exit 3"))
  }

  @Test("grep answers with path:line:text and caps the hits")
  func grepFinds() async throws {
    guard Ripgrep.locate() != nil else { return }
    let (workspace, root) = try sandbox()
    _ = try seed(root, "a.txt", lines: 10)

    let out = try await workspace.grep(pattern: "line", limit: 2)
    #expect(out.contains("a.txt:1:line 1"))
    #expect(out.contains("more matches"))

    let none = try await workspace.grep(pattern: "zzzz")
    #expect(none == "no matches")
  }
}
