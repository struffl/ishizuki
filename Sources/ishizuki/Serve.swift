// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct Serve: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "serve",
    abstract: "Serve the model over the OpenAI and Anthropic APIs.",
    discussion: """
      Exposes both API shapes on one port:

        POST /v1/chat/completions   OpenAI  - Hermes, Pi, most agent frameworks
        POST /v1/messages           Anthropic - Claude Code
        POST /v1/messages/count_tokens
        GET  /v1/models, GET /health

      Point a harness at it:
        export OPENAI_BASE_URL=http://127.0.0.1:8128/v1
        export ANTHROPIC_BASE_URL=http://127.0.0.1:8128
      """)

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long, help: "HuggingFace repo to fetch the model from if missing.")
  var repo: String = defaultRepo
  @Flag(name: .long, help: "Skip the model download/repair check.") var offline = false
  @Option(name: .shortAndLong) var port: UInt16 = 8128
  @Option(name: .long, help: "Name reported to clients.")
  var servedName: String = "ternary-bonsai-2-27b"

  @Option(name: .long) var temperature: Float = 0.7
  @Option(name: .long) var topP: Float = 1.0
  @Option(name: .long) var topK: Int = 0
  @Option(name: .long) var minP: Float = 0.0

  @Flag(
    name: .long,
    help:
      "Let the model think before answering. Off by default: it is slow and most harnesses only want the answer."
  )
  var thinking = false

  @Option(
    name: .long,
    help:
      "Quantize the KV cache to this many bits (3.5 = 3-bit keys / 4-bit values). Pass 16 for fp16."
  )
  var kvBits: Float = 3.5

  @Option(name: .long, help: "Tokens kept unquantized at the head of the KV cache.")
  var kvWindow: Int = 128

  @Option(
    name: .long,
    help: "Prefix caches kept for reuse. More slots survive interleaved requests.")
  var cacheSlots: Int = 4

  @Option(name: .long, help: "Seconds idle before caches are released. 0 disables.")
  var idleTimeout: Double = 120

  @Option(name: .long, help: "Seconds idle before the model is unloaded entirely. 0 disables.")
  var evictTimeout: Double = 0

  @Option(name: .long, help: "Ask the OS to keep this many GB wired while serving. 0 disables.")
  var wireGB: Double = 0

  @Option(name: .long, help: "Cap MLX's reusable buffer cache, in GB. 0 leaves the default.")
  var cacheLimitGB: Double = 0

  @Option(
    name: .long,
    help: "Scheduling: adaptive (default), polite, normal, background.")
  var politeness: String = "adaptive"

  @Option(name: .long, help: "Stretch context past 262144 by this factor, e.g. 2 for ~512K.")
  var contextScale: Float = 1

  @Flag(name: .long, help: "Do not load the model until the first request arrives.")
  var lazyLoad = false

  @Flag(name: .long, help: "Load the model immediately at startup. Wins over --lazy-load.")
  var hot = false

  @Flag(name: .long, help: "Disable coloured output.")
  var noColor = false

  @Flag(name: .long, help: "Log plain lines instead of the live dashboard.")
  var disableDashboard = false

  func run() throws {
    let kvConfig = KVCacheConfig(bits: kvBits, residualWindow: kvWindow)
    try kvConfig.validate()

    if noColor { Style.disable() }
    let level = Politeness.Level(rawValue: politeness) ?? .adaptive
    Politeness.apply(level)

    let modelURL = URL(filePath: model)
    if !offline {
      try ModelDownloader.ensure(directory: modelURL, repo: repo)
    }

    let residency = ResidencyManager.Options(
      wiredBytes: Int(wireGB * 1_073_741_824),
      cacheLimit: Int(cacheLimitGB * 1_073_741_824),
      idleSeconds: idleTimeout,
      evictSeconds: evictTimeout)

    let server = try APIServer(
      directory: modelURL,
      modelName: servedName,
      thinking: thinking,
      samplingOptions: SamplingOptions(
        temperature: temperature, topP: topP, topK: topK, minP: minP),
      kvConfig: kvConfig,
      residency: residency,
      politeness: level,
      ropeScaling: contextScale > 1
        ? RopeScaling(method: .yarn, factor: contextScale) : .none,
      cacheSlots: cacheSlots,
      preload: hot || !lazyLoad)

    var header = [
      Style.banner("serving \(servedName) on http://127.0.0.1:\(port)"),
      "",
      "  " + Style.field("OpenAI", Style.faint("export OPENAI_BASE_URL=http://127.0.0.1:\(port)/v1")),
      "  " + Style.field("Anthropic", Style.faint("export ANTHROPIC_BASE_URL=http://127.0.0.1:\(port)")),
      "  " + Style.field("scheduling", Style.faint(Politeness.describe(level))),
    ]
    if idleTimeout > 0 {
      header.append(
        "  " + Style.field("idle", Style.faint("caches released after \(Int(idleTimeout))s")))
    }
    if evictTimeout > 0 {
      header.append(
        "  " + Style.field("evict", Style.faint("model unloaded after \(Int(evictTimeout))s")))
    }
    if wireGB > 0 {
      header.append(
        "  " + Style.field("wired", Style.faint("\(wireGB) GB requested while serving")))
    }

    if disableDashboard || !ServeDashboard.isSupported {
      server.log = { message in
        FileHandle.standardError.write(Data("[bonsai] \(message)\n".utf8))
      }
      try server.listen(port: port)
      for line in header { print(line) }
      print("  " + Style.field("memory", Style.faint(ResidencyManager.describeMemory())))
      fflush(stdout)
      installFarewell(for: server)
    } else {
      let dashboard = ServeDashboard(stats: server.stats, header: header)
      server.log = { [dashboard] message in dashboard.append(log: message) }
      try server.listen(port: port)
      dashboard.start()
    }

    dispatchMain()
  }

  private func installFarewell(for server: APIServer) {
    for number in [SIGINT, SIGTERM] {
      signal(number, SIG_IGN)
      let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
      source.setEventHandler {
        Farewell.print(tokens: server.stats.snapshot().totals.totalTokens)
        Foundation.exit(0)
      }
      source.resume()
      farewellSources.append(source)
    }
  }
}

private nonisolated(unsafe) var farewellSources: [DispatchSourceSignal] = []

struct InstallAgent: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "install-agent",
    abstract: "Write a launchd LaunchAgent that runs `ishizuki serve` at login.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .shortAndLong) var port: UInt16 = 8128
  @Option(name: .long) var label: String = "studio.bonsai.server"
  @Option(name: .long, help: "Seconds idle before the model is unloaded.")
  var evictTimeout: Double = 900
  @Option(name: .long, help: "KV cache bits, e.g. 3.5.")
  var kvBits: Float = 3.5
  @Flag(name: .long, help: "Print the plist instead of writing it.")
  var dryRun = false

  func run() throws {
    let executable = URL(filePath: CommandLine.arguments[0])
      .standardizedFileURL.path
    let logs = NSHomeDirectory() + "/Library/Logs"
    let plistURL = URL(
      filePath: NSHomeDirectory() + "/Library/LaunchAgents/\(label).plist")

    var arguments = [
      executable, "serve",
      "--model", model,
      "--port", "\(port)",
      "--evict-timeout", "\(Int(evictTimeout))",
      "--lazy-load",
    ]
    arguments += ["--kv-bits", "\(kvBits)"]

    let argumentXML =
      arguments
      .map { "        <string>\($0)</string>" }
      .joined(separator: "\n")

    let plist = """
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
      <dict>
          <key>Label</key>
          <string>\(label)</string>
          <key>ProgramArguments</key>
          <array>
      \(argumentXML)
          </array>
          <key>RunAtLoad</key>
          <true/>
          <key>KeepAlive</key>
          <dict>
              <key>SuccessfulExit</key>
              <false/>
          </dict>
          <key>ProcessType</key>
          <string>Adaptive</string>
          <key>StandardOutPath</key>
          <string>\(logs)/\(label).log</string>
          <key>StandardErrorPath</key>
          <string>\(logs)/\(label).err.log</string>
      </dict>
      </plist>
      """

    if dryRun {
      print(plist)
      return
    }

    try FileManager.default.createDirectory(
      at: plistURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try plist.write(to: plistURL, atomically: true, encoding: .utf8)

    print("wrote \(plistURL.path)")
    print("")
    print("Load it:")
    print("  launchctl bootstrap gui/$(id -u) \(plistURL.path)")
    print("Unload it:")
    print("  launchctl bootout gui/$(id -u)/\(label)")
    print("Logs:")
    print("  tail -f \(logs)/\(label).err.log")
    print("")
    print(
      "With --lazy-load and --evict-timeout \(Int(evictTimeout)), the process stays up but")
    print("holds no weights until a request arrives, and releases them again when idle.")
  }
}
