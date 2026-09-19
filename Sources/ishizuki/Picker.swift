// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Darwin
import Foundation
import IshizukiKit

/// A list you move through with the arrow keys. Used where a command would otherwise make the
/// reader retype something it already knows — which model to serve, which cached prefix to drop.
///
/// Falls back to a numbered prompt when there is no terminal to take over, so the same call
/// works over a pipe or in a script.
enum Picker {
  struct Row {
    let title: String
    let detail: String
    /// Rows that describe rather than offer — a total, a note — are skipped by the cursor.
    let selectable: Bool

    init(title: String, detail: String = "", selectable: Bool = true) {
      self.title = title
      self.detail = detail
      self.selectable = selectable
    }
  }

  enum Outcome {
    case chose(Int)
    case delete(Int)
    case cancelled
  }

  static var isInteractive: Bool {
    isatty(STDIN_FILENO) == 1 && isatty(STDOUT_FILENO) == 1
  }

  /// `deletable` adds d/x/delete as a second verb, reported back as `.delete`.
  static func run(
    title: String, rows: [Row], deletable: Bool = false, initial: Int = 0
  ) -> Outcome {
    guard !rows.isEmpty else { return .cancelled }
    guard isInteractive else { return prompt(title: title, rows: rows) }

    var cursor =
      rows.indices.contains(initial) && rows[initial].selectable
      ? initial : (rows.firstIndex { $0.selectable } ?? 0)

    var original = termios()
    tcgetattr(STDIN_FILENO, &original)
    var raw = original
    raw.c_lflag &= ~(UInt(ECHO) | UInt(ICANON))
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
    write(STDOUT_FILENO, "\u{1B}[?25l", 6)

    defer {
      var restore = original
      tcsetattr(STDIN_FILENO, TCSAFLUSH, &restore)
      FileHandle.standardOutput.write(Data("\u{1B}[?25h".utf8))
    }

    var drawn = 0
    func render() {
      var out = ""
      if drawn > 0 { out += "\u{1B}[\(drawn)A" }
      out += "\u{1B}[0J"
      out += Style.banner(title) + "\n\n"
      for (index, row) in rows.enumerated() {
        if !row.selectable {
          out += "    " + Style.faint(row.title) + "\n"
          continue
        }
        let marker = index == cursor ? Style.accent("❯ ") : "  "
        let name = index == cursor ? Style.bright(row.title) : row.title
        out += "  " + marker + name
        if !row.detail.isEmpty { out += "  " + Style.faint(row.detail) }
        out += "\n"
      }
      out += "\n"
      out += Style.faint(
        deletable
          ? "  ↑↓ move · d delete · enter choose · q quit"
          : "  ↑↓ move · enter choose · q quit")
      out += "\n"
      drawn = rows.count + 4
      FileHandle.standardOutput.write(Data(out.utf8))
    }

    func step(_ direction: Int) {
      var next = cursor
      for _ in 0..<rows.count {
        next = (next + direction + rows.count) % rows.count
        if rows[next].selectable {
          cursor = next
          return
        }
      }
    }

    render()
    while true {
      var byte: UInt8 = 0
      guard read(STDIN_FILENO, &byte, 1) == 1 else { return .cancelled }

      switch byte {
      case 0x1B:
        // Either a bare escape or an arrow's CSI sequence.
        var sequence: [UInt8] = [0, 0]
        guard read(STDIN_FILENO, &sequence, 2) == 2, sequence[0] == 0x5B else {
          return .cancelled
        }
        switch sequence[1] {
        case 0x41: step(-1)
        case 0x42: step(1)
        case 0x33:
          var trailing: UInt8 = 0
          _ = read(STDIN_FILENO, &trailing, 1)
          if deletable { return .delete(cursor) }
        default: break
        }
      case 0x0A, 0x0D: return .chose(cursor)
      case UInt8(ascii: "k"): step(-1)
      case UInt8(ascii: "j"): step(1)
      case UInt8(ascii: "d"), UInt8(ascii: "x"), 0x7F:
        if deletable { return .delete(cursor) }
      case UInt8(ascii: "q"), 0x03: return .cancelled
      default: break
      }
      render()
    }
  }

  /// No terminal to take over: ask for a number instead.
  private static func prompt(title: String, rows: [Row]) -> Outcome {
    print(Style.banner(title))
    print("")
    for (index, row) in rows.enumerated() where row.selectable {
      print("  \(index + 1). \(row.title)  \(Style.faint(row.detail))")
    }
    print("")
    print("  choose [1-\(rows.count)]: ", terminator: "")
    guard let line = readLine(strippingNewline: true), let choice = Int(line),
      rows.indices.contains(choice - 1), rows[choice - 1].selectable
    else { return .cancelled }
    return .chose(choice - 1)
  }
}
