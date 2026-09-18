// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Darwin
import Foundation
import IshizukiKit
import MLX

struct Demo: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "demo",
    abstract: "Talk to 石付き in the terminal.")

  @Option(name: .long, help: "Path to the MLX pack directory.")
  var model: String = defaultModelPath

  @Option(name: .long, help: "HuggingFace repo to fetch the model from if missing.")
  var repo: String = defaultRepo

  @Flag(name: .long, help: "Skip the model download/repair check.") var offline = false

  @Option(name: .long, help: "System prompt.")
  var system: String = "You are an assistant named Ishizuki."

  @Option(name: .shortAndLong, help: "Maximum tokens per reply.")
  var maxTokens: Int = 640

  @Option(name: .long) var temperature: Float = 0.7
  @Option(name: .long) var topP: Float = 1.0
  @Option(name: .long) var topK: Int = 0
  @Option(name: .long) var minP: Float = 0.0
  @Option(name: .long) var repetitionPenalty: Float = 1.0
  @Option(name: .long) var seed: UInt64?

  @Option(name: .long, help: "Reasoning effort: none, low, medium, high, xhigh.")
  var reasoning: String = "medium"

  @Flag(name: .long, help: "Disable the model's thinking block.")
  var noThinking = false

  @Flag(name: .long, help: "Fold the thinking block to a single line instead of showing it.")
  var foldThinking = false

  @Option(
    name: .long,
    help: "Scheduling: adaptive (default), polite, normal, background.")
  var politeness: String = "adaptive"

  @Option(name: .long, help: "GPU cache limit in MB (0 leaves MLX's default).")
  var cacheLimit: Int = 0

  @Flag(name: .long, help: "Disable coloured output.")
  var noColor = false

  @Flag(name: .long, help: "Skip the bonsai drawn at the top of the session.")
  var noArt = false

  @Flag(name: .long, help: "Draw a bonsai and leave, without loading the model.")
  var art = false

  func run() throws {
    if noColor { Style.disable() }
    if art {
      for line in BonsaiArt.lines(seed: seed) { print("  " + line) }
      return
    }

    let level = Politeness.Level(rawValue: politeness) ?? .adaptive
    Politeness.apply(level)
    if cacheLimit > 0 { Memory.cacheLimit = cacheLimit * 1024 * 1024 }

    if !noArt {
      for line in BonsaiArt.lines(seed: seed) { print("  " + line) }
      print("")
      fflush(stdout)
    }

    let packURL = URL(filePath: resolvedModelPath(model, repo: repo))
    if !offline { try ModelDownloader.ensure(directory: packURL, repo: repo) }

    let loadStart = Date()
    let bonsai = try BonsaiModel(directory: packURL)
    let template = try ChatTemplate(directory: packURL)
    let generator = Generator(model: bonsai, prefillChunkSize: Politeness.prefillChunk(for: level))
    generator.politeness = level
    let sessions = SessionCache(capacity: 1)

    let effort = ReasoningEffort(rawValue: reasoning)
    var session = ChatSession(
      system: system,
      thinking: !noThinking && effort != ReasoningEffort.none,
      effort: effort == ReasoningEffort.none ? nil : effort)

    print(Style.banner("🌸 a conversation with " + Style.bright("石付き")))
    print("")
    print(
      "  "
        + Style.field("model", Style.accent(packURL.lastPathComponent))
        + Style.faint(String(format: "  loaded in %.1fs", -loadStart.timeIntervalSinceNow)))
    print("  " + Style.field("sched", Style.faint(Politeness.describe(level))))
    print("  " + Style.field("thinking", Style.faint(session.describeThinking)))
    print(
      "  "
        + Style.field(
          "commands",
          Style.faint("/new  /think  /system  /stats  /help  /quit  ·  ⌃D to leave")))
    print("")

    let options = SamplingOptions(
      temperature: temperature, topP: topP, topK: topK, minP: minP,
      repetitionPenalty: repetitionPenalty, seed: seed)

    var totals = ChatTotals()

    while true {
      guard let line = ask() else { break }
      let entry = line.trimmingCharacters(in: .whitespacesAndNewlines)
      if entry.isEmpty { continue }

      if entry.hasPrefix("/") {
        switch session.handle(command: entry, totals: totals) {
        case .handled: continue
        case .reset:
          sessions.reset()
          continue
        case .quit: return exit(totals: totals)
        }
      }

      session.messages.append(.user(entry))
      let prompt = try template.render(
        messages: session.transcript,
        addGenerationPrompt: true,
        enableThinking: session.thinking,
        reasoningEffort: session.thinking ? session.effort : nil)
      let promptTokens = bonsai.tokenizer.encode(prompt)

      let lease = sessions.prepare(for: promptTokens, model: bonsai)
      var stream = ReplyStream(thinking: session.thinking, folded: foldThinking)

      let result = generator.generate(
        promptTokens: promptTokens, options: options, maxTokens: maxTokens,
        cache: lease.cache, cachedPrefixLength: lease.reused,
        onProgress: { stream.progress($0) },
        onToken: { stream.push($0) })
      sessions.commit(lease, generated: result.tokens)
      stream.finish()

      let reply = ToolCallParser.parse(
        session.thinking ? "<think>" + result.text : result.text)
      session.messages.append(.assistant(reply.content))
      totals.absorb(result.stats, reused: lease.reused)

      print("")
      print(
        "  "
          + Style.faint("▔ ")
          + Style.bright(String(format: "%.0f", result.stats.promptTokensPerSecond))
          + Style.muted(" tok/s in")
          + Style.faint(" · ")
          + Style.bright(String(format: "%.1f", result.stats.generationTokensPerSecond))
          + Style.muted(" tok/s out")
          + Style.faint(
            "  ·  \(group(promptTokens.count + result.tokens.count)) tok of context"
              + (lease.reused > 0 ? ", \(group(lease.reused)) kept warm" : "")))
      print("")
    }

    exit(totals: totals)
  }

  private func ask() -> String? {
    let label = Style.bright("anata") + Style.faint(" ▸ ")
    if isatty(STDIN_FILENO) == 1 {
      fputs(label, stdout)
      fflush(stdout)
      return readLine(strippingNewline: true)
    }
    guard let line = readLine(strippingNewline: true) else { return nil }
    print(label + line)
    return line
  }

  private func exit(totals: ChatTotals) {
    print("")
    print(
      "  "
        + Style.field(
          "session",
          Style.accent("\(totals.turns)") + Style.muted(totals.turns == 1 ? " turn" : " turns")
            + Style.faint(" · ")
            + Style.accent(group(totals.promptTokens)) + Style.muted(" tok in")
            + Style.faint(" · ")
            + Style.accent(group(totals.generatedTokens)) + Style.muted(" tok out")
            + Style.faint(
              String(format: "  ·  %.1f tok/s average", totals.decodeRate))))
    if let line = Farewell.line(tokens: totals.promptTokens + totals.generatedTokens) {
      print(line)
    }
    print(Style.faint("mata ne.") + " 🌸")
  }
}

private struct ChatSession {
  enum Outcome { case handled, reset, quit }

  var system: String
  var thinking: Bool
  var effort: ReasoningEffort?
  var messages: [ChatMessage] = []

  var transcript: [ChatMessage] {
    system.isEmpty ? messages : [.system(system)] + messages
  }

  var describeThinking: String {
    guard thinking else { return "off" }
    return effort.map { "on, \($0.rawValue) effort" } ?? "on"
  }

  mutating func handle(command: String, totals: ChatTotals) -> Outcome {
    let parts = command.dropFirst().split(separator: " ", maxSplits: 1)
    let verb = parts.first.map(String.init) ?? ""
    let argument = parts.count > 1 ? String(parts[1]) : ""

    switch verb {
    case "quit", "exit", "q":
      return .quit

    case "new", "clear":
      messages.removeAll()
      say("the thread is empty again")
      return .reset

    case "system":
      guard !argument.isEmpty else {
        say(Style.faint(system))
        return .handled
      }
      system = argument
      messages.removeAll()
      say("system prompt replaced, thread cleared")
      return .reset

    case "think":
      switch argument {
      case "", "on": thinking = true
      case "off", "none": thinking = false
      default:
        guard let level = ReasoningEffort(rawValue: argument) else {
          say(Style.warn("effort is one of: none, low, medium, high, xhigh"))
          return .handled
        }
        thinking = level != ReasoningEffort.none
        effort = thinking ? level : nil
      }
      say("thinking " + describeThinking)
      return .handled

    case "stats":
      say(
        Style.accent("\(totals.turns)") + Style.muted(totals.turns == 1 ? " turn" : " turns")
          + Style.faint(" · ")
          + Style.accent(group(totals.promptTokens)) + Style.muted(" tok in")
          + Style.faint(" · ")
          + Style.accent(group(totals.generatedTokens)) + Style.muted(" tok out")
          + Style.faint(String(format: "  ·  %.1f tok/s", totals.decodeRate)))
      return .handled

    case "help", "?":
      for (name, blurb) in Self.help {
        say(
          Style.accent(name.padding(toLength: 16, withPad: " ", startingAt: 0)) + Style.faint(blurb)
        )
      }
      return .handled

    default:
      say(Style.warn("no such command — try /help"))
      return .handled
    }
  }

  private static let help: [(String, String)] = [
    ("/new", "forget the thread, keep the system prompt"),
    ("/system <text>", "replace the system prompt"),
    ("/think <effort>", "on, off, or none/low/medium/high/xhigh"),
    ("/stats", "tokens and speed so far"),
    ("/quit", "leave"),
  ]

  private func say(_ line: String) {
    print("  " + Style.faint("· ") + line)
  }
}

private struct ChatTotals {
  var turns = 0
  var promptTokens = 0
  var generatedTokens = 0
  var generationSeconds = 0.0

  var decodeRate: Double {
    generationSeconds > 0 ? Double(generatedTokens) / generationSeconds : 0
  }

  mutating func absorb(_ stats: GenerationStats, reused: Int) {
    turns += 1
    promptTokens += stats.promptTokens + reused
    generatedTokens += stats.generatedTokens
    generationSeconds += stats.generationSeconds
  }
}

private struct ReplyStream {
  private let folded: Bool
  private let animated = isatty(STDOUT_FILENO) == 1 && Style.depth != .none
  private let frames = Array("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
  private let guardLength = 12

  private var inThinking: Bool
  private var thoughtStart = Date()
  private var thoughtTokens = 0
  private var buffer = ""
  private var emitted = 0
  private var frame = 0
  private var lastPaint = Date.distantPast
  private var opened = false
  private var box = ThoughtBox()
  private var markdown = MarkdownStream()

  init(thinking: Bool, folded: Bool) {
    self.inThinking = thinking
    self.folded = folded
  }

  mutating func progress(_ step: GenerationProgress) {
    guard case .prefill(let done, let total) = step, total > 0 else { return }
    spin(Style.muted("reading ") + Style.faint("\(group(done)) / \(group(total)) tok"))
  }

  mutating func push(_ fragment: String) -> Bool {
    guard inThinking else {
      emit(reply: fragment)
      return true
    }

    thoughtTokens += 1
    buffer += fragment

    guard let end = buffer.range(of: "</think>") else {
      if folded {
        spin(
          Style.bright("石付き") + Style.muted(" is thinking")
            + Style.faint(
              String(format: "  %d tok · %.1fs", thoughtTokens, -thoughtStart.timeIntervalSinceNow))
        )
      } else {
        openBox()
        box.push(take(upTo: buffer.count - guardLength), to: write)
      }
      return true
    }

    let tail = String(buffer[end.upperBound...])
    if !folded {
      openBox()
      box.push(take(upTo: buffer.distance(from: buffer.startIndex, to: end.lowerBound)), to: write)
    }
    clear()
    inThinking = false
    closeBox(
      Style.muted("thought for ")
        + Style.faint(
          String(format: "%.1fs · %d tok", -thoughtStart.timeIntervalSinceNow, thoughtTokens)))
    emit(reply: tail)
    return true
  }

  mutating func finish() {
    clear()
    write(markdown.finish())
    if inThinking {
      closeBox(
        Style.warn("cut off ")
          + Style.faint(
            String(
              format: "at the %d token budget after %.1fs — raise --max-tokens",
              thoughtTokens, -thoughtStart.timeIntervalSinceNow)))
    }
    write("\n")
  }

  private mutating func openBox() {
    guard !box.open else { return }
    box.open = true
    write("\n" + Style.faint("  ╭ ") + Style.muted("思考") + Style.faint("  thinking") + "\n")
    write(ThoughtBox.edge)
  }

  private mutating func closeBox(_ label: String) {
    if box.open {
      box.close(to: write)
      box.open = false
      write(Style.faint("  ╰ ") + label + "\n")
    } else {
      write("  " + Style.faint("✻ ") + label + "\n")
    }
    opened = false
  }

  private mutating func emit(reply fragment: String) {
    var text = fragment
    if !opened {
      text = String(text.drop(while: { $0.isWhitespace }))
      guard !text.isEmpty else { return }
      opened = true
      write("\n" + Style.bright("石付き") + "\n")
    }
    write(markdown.push(text))
  }

  private mutating func take(upTo limit: Int) -> String {
    let characters = Array(buffer)
    let available = max(0, min(limit, characters.count))
    guard available > emitted else { return "" }
    let slice = String(characters[emitted..<available])
    emitted = available
    return slice
  }

  private mutating func spin(_ label: String) {
    guard animated, -lastPaint.timeIntervalSinceNow >= 0.08 else { return }
    lastPaint = Date()
    let mark = String(frames[frame % frames.count])
    frame += 1
    write("\r\u{1B}[2K  " + Style.accent(mark) + " " + label)
  }

  private func clear() {
    guard animated else { return }
    write("\r\u{1B}[2K")
  }

  private func write(_ text: String) {
    fputs(text, stdout)
    fflush(stdout)
  }
}

private struct ThoughtBox {
  static let edge = Style.faint("  │ ")

  var open = false
  private let width = max(32, terminalColumns() - 6)
  private var column = 0
  private var word = ""
  private var breaks = 0
  private var space = false

  mutating func push(_ text: String, to write: (String) -> Void) {
    guard !text.isEmpty else { return }
    var line = ""
    for character in text {
      if character.isNewline {
        line += flushWord()
        breaks += 1
        column = 0
      } else if character == " " || character == "\t" {
        line += flushWord()
        if column > 0 { space = true }
      } else {
        word.append(character)
        if column + word.count + (space ? 1 : 0) > width {
          breaks = max(breaks, 1)
          space = false
          column = 0
        }
      }
    }
    if !line.isEmpty { write(line) }
  }

  mutating func close(to write: (String) -> Void) {
    write(flushWord() + "\n")
    column = 0
    breaks = 0
  }

  private mutating func flushWord() -> String {
    guard !word.isEmpty else { return "" }
    var text = String(repeating: "\n" + Self.edge, count: breaks)
    if space, column > 0 {
      text += Style.faint(" ")
      column += 1
    }
    breaks = 0
    space = false
    column += word.count
    text += Style.faint(word)
    word = ""
    return text
  }
}

func terminalColumns() -> Int {
  var size = winsize()
  if ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &size) == 0, size.ws_col > 0 {
    return Int(size.ws_col)
  }
  if let declared = ProcessInfo.processInfo.environment["COLUMNS"], let columns = Int(declared) {
    return columns
  }
  return 80
}

private func group(_ value: Int) -> String {
  let formatter = NumberFormatter()
  formatter.numberStyle = .decimal
  return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
}
