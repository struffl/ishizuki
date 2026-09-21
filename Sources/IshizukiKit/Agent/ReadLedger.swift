// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What the model has actually looked at. A write it cannot justify from something it read is
// refused, and the refusal says which slice to go and read.

import Foundation

/// A refusal is not a failure: it is a turn of the loop. The message is the whole instruction,
/// so a small model can act on it without being told twice.
public struct LedgerRefusal: Error, LocalizedError, Sendable {
  public let message: String
  public var errorDescription: String? { message }
}

public actor ReadLedger {
  private struct Record {
    var fingerprint: Int
    var lineCount: Int
    var seen: [ClosedRange<Int>]

    var sawEverything: Bool {
      guard lineCount > 0 else { return true }
      return covers(1...lineCount)
    }

    func covers(_ range: ClosedRange<Int>) -> Bool {
      var cursor = range.lowerBound
      // The slices arrive in whatever order the model asked for them, so walk the merged set.
      for slice in seen.sorted(by: { $0.lowerBound < $1.lowerBound }) {
        if slice.lowerBound > cursor { return false }
        cursor = max(cursor, slice.upperBound + 1)
        if cursor > range.upperBound { return true }
      }
      return cursor > range.upperBound
    }
  }

  private var records: [String: Record] = [:]

  public init() {}

  public static func fingerprint(_ contents: String) -> Int {
    var hasher = Hasher()
    hasher.combine(contents)
    return hasher.finalize()
  }

  public func noteRead(
    path: String, fingerprint: Int, lineCount: Int, slice: ClosedRange<Int>
  ) {
    var record = records[path] ?? Record(fingerprint: fingerprint, lineCount: lineCount, seen: [])
    if record.fingerprint != fingerprint {
      record = Record(fingerprint: fingerprint, lineCount: lineCount, seen: [])
    }
    record.lineCount = lineCount
    record.seen.append(slice)
    records[path] = record
  }

  /// A whole-file overwrite needs the whole file seen, and unchanged since it was seen.
  public func checkWrite(path: String, fingerprint: Int?, lineCount: Int) throws {
    guard let fingerprint else { return }
    guard let record = records[path] else {
      throw LedgerRefusal(
        message: "refused: \(path) exists and has not been read. Read it first "
          + "(\(lineCount) lines).")
    }
    guard record.fingerprint == fingerprint else {
      throw LedgerRefusal(
        message: "refused: \(path) changed since you read it. Read it again.")
    }
    guard record.sawEverything else {
      throw LedgerRefusal(
        message: "refused: you have seen \(describe(record.seen)) of \(path)'s "
          + "\(record.lineCount) lines. Read the rest, or use edit instead of write.")
    }
  }

  /// An edit needs only the lines it touches, which is the point: a slice is cheap and the
  /// model can decide to pull more.
  public func checkEdit(
    path: String, fingerprint: Int, lines: ClosedRange<Int>
  ) throws {
    guard let record = records[path] else {
      throw LedgerRefusal(
        message: "refused: \(path) has not been read. Read the lines you mean to change.")
    }
    guard record.fingerprint == fingerprint else {
      throw LedgerRefusal(
        message: "refused: \(path) changed since you read it. Read it again.")
    }
    guard record.covers(lines) else {
      throw LedgerRefusal(
        message: "refused: you have not read \(path):\(lines.lowerBound)-\(lines.upperBound). "
          + "Read that slice first.")
    }
  }

  public func invalidate(path: String) {
    records[path] = nil
  }

  public func seen(path: String) -> Bool {
    records[path] != nil
  }

  private func describe(_ ranges: [ClosedRange<Int>]) -> String {
    var total = 0
    var cursor = 0
    for slice in ranges.sorted(by: { $0.lowerBound < $1.lowerBound }) {
      let start = max(slice.lowerBound, cursor + 1)
      if slice.upperBound >= start { total += slice.upperBound - start + 1 }
      cursor = max(cursor, slice.upperBound)
    }
    return "\(total) line\(total == 1 ? "" : "s")"
  }
}
