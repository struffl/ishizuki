// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The one place a tool goes for a path, a fingerprint and a shell. Holding them together is
// what lets every tool answer in the same few lines.

import Foundation

public final class Workspace: Sendable {
  public let host: any ShellHost
  public let ledger: ReadLedger
  /// How many lines a read hands back when the model does not say. Small on purpose: the model
  /// is told what it did not get, and asks again if it wants it.
  public let sliceLines: Int

  public init(host: any ShellHost, ledger: ReadLedger = ReadLedger(), sliceLines: Int = 120) {
    self.host = host
    self.ledger = ledger
    self.sliceLines = sliceLines
  }

  public var root: URL { host.workspace }

  struct FileRead: Sendable {
    var url: URL
    var display: String
    var lines: [String]
    var fingerprint: Int
  }

  func read(_ path: String) throws -> FileRead {
    let url = try host.resolve(path)
    guard let data = FileManager.default.contents(atPath: url.path) else {
      throw LedgerRefusal(message: "no such file: \(host.display(url))")
    }
    guard !isBinary(data) else {
      throw LedgerRefusal(message: "\(host.display(url)) is not text")
    }
    let contents = String(decoding: data, as: UTF8.self)
    return FileRead(
      url: url,
      display: host.display(url),
      lines: contents.components(separatedBy: "\n"),
      fingerprint: ReadLedger.fingerprint(contents))
  }

  private func isBinary(_ data: Data) -> Bool {
    data.prefix(1024).contains { $0 == 0 }
  }

  /// Which lines a stretch of the file falls on, so an edit can be checked against what was read.
  static func lineRange(of range: Range<String.Index>, in contents: String) -> ClosedRange<Int> {
    let before = contents[contents.startIndex..<range.lowerBound]
    let start = before.reduce(1) { $1 == "\n" ? $0 + 1 : $0 }
    let inside = contents[range]
    let end = inside.reduce(start) { $1 == "\n" ? $0 + 1 : $0 }
    return start...end
  }
}
