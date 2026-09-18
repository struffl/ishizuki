// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation

public enum Style {
  public enum Depth { case none, ansi256, trueColor }

  public nonisolated(unsafe) static var depth: Depth = detect()

  public static func disable() { depth = .none }

  public static func detect(stream: Int32 = 1) -> Depth {
    let environment = ProcessInfo.processInfo.environment

    if environment["NO_COLOR"] != nil { return .none }
    if environment["CLICOLOR"] == "0" { return .none }

    let forced = environment["CLICOLOR_FORCE"].map { $0 != "0" } ?? false
    if !forced && isatty(stream) == 0 { return .none }

    let term = environment["TERM"] ?? ""
    if term == "dumb" { return .none }

    let colorTerm = environment["COLORTERM"] ?? ""
    let program = environment["TERM_PROGRAM"] ?? ""

    if colorTerm == "truecolor" || colorTerm == "24bit" { return .trueColor }
    if program == "ghostty" || term.contains("ghostty") { return .trueColor }
    if program == "iTerm.app" || program == "WezTerm" || program == "vscode" {
      return .trueColor
    }
    if term.contains("256color") { return .ansi256 }
    if program == "Apple_Terminal" { return .ansi256 }
    return term.isEmpty ? .none : .ansi256
  }

  public struct Colour: Sendable {
    let red: Int, green: Int, blue: Int, fallback: Int

    public static let accent = Colour(red: 122, green: 162, blue: 196, fallback: 110)
    public static let accentBright = Colour(red: 158, green: 194, blue: 222, fallback: 153)
    public static let text = Colour(red: 198, green: 202, blue: 208, fallback: 252)
    public static let muted = Colour(red: 134, green: 140, blue: 148, fallback: 245)
    public static let faint = Colour(red: 98, green: 104, blue: 112, fallback: 240)
    public static let good = Colour(red: 140, green: 176, blue: 150, fallback: 108)
    public static let warn = Colour(red: 202, green: 172, blue: 124, fallback: 179)
    public static let bad = Colour(red: 198, green: 132, blue: 132, fallback: 167)

    public static let barkLight = Colour(red: 176, green: 144, blue: 110, fallback: 180)
    public static let bark = Colour(red: 146, green: 116, blue: 88, fallback: 137)
    public static let barkDeep = Colour(red: 108, green: 84, blue: 64, fallback: 95)
    public static let leaf = Colour(red: 118, green: 166, blue: 114, fallback: 108)
    public static let leafDeep = Colour(red: 84, green: 128, blue: 90, fallback: 71)
    public static let leafBright = Colour(red: 160, green: 200, blue: 142, fallback: 150)
    public static let stone = Colour(red: 132, green: 146, blue: 162, fallback: 103)
    public static let stoneDeep = Colour(red: 94, green: 108, blue: 124, fallback: 60)
    public static let blossom = Colour(red: 226, green: 164, blue: 192, fallback: 175)
    public static let blossomDeep = Colour(red: 198, green: 92, blue: 112, fallback: 168)
    public static let blossomPlum = Colour(red: 172, green: 136, blue: 206, fallback: 140)
  }

  public static func paint(_ text: String, _ colour: Colour, bold: Bool = false) -> String {
    switch depth {
    case .none:
      return text
    case .ansi256:
      return "\u{1B}[\(bold ? "1;" : "")38;5;\(colour.fallback)m\(text)\u{1B}[0m"
    case .trueColor:
      return
        "\u{1B}[\(bold ? "1;" : "")38;2;\(colour.red);\(colour.green);\(colour.blue)m\(text)\u{1B}[0m"
    }
  }

  public static func accent(_ t: String) -> String { paint(t, .accent) }
  public static func bright(_ t: String) -> String { paint(t, .accentBright, bold: true) }
  public static func muted(_ t: String) -> String { paint(t, .muted) }
  public static func faint(_ t: String) -> String { paint(t, .faint) }
  public static func good(_ t: String) -> String { paint(t, .good) }
  public static func warn(_ t: String) -> String { paint(t, .warn) }
  public static func bad(_ t: String) -> String { paint(t, .bad) }

  public static func field(_ label: String, _ value: String, width: Int = 10) -> String {
    let padded = label.padding(toLength: max(width, label.count), withPad: " ", startingAt: 0)
    return muted(padded) + " " + value
  }

  public static var rule: String {
    paint(String(repeating: "─", count: 46), .faint)
  }

  public static func banner(_ subtitle: String) -> String {
    bright("石付き") + muted("  ishizuki") + faint("  " + BuildInfo.version)
      + "\n" + faint(subtitle)
  }
}
