// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

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
  @Option(
    name: .long,
    help: "Name reported to clients. Defaults to the pack's own name in the catalog.")
  var servedName: String?

  @OptionGroup var neural: ANEOption

  @Option(name: .long) var temperature: Float = 0.7
  @Option(name: .long) var topP: Float = 1.0
  @Option(name: .long) var topK: Int = 0
  @Option(name: .long) var minP: Float = 0.0

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
    help:
      "Pin the number of prefix caches kept for reuse. Default starts at 1 and doubles under pressure."
  )
  var cacheSlots: Int?

  @Option(
    name: .long,
    help:
      "Seconds idle before the reusable buffer pool is released. Prefix caches are kept. 0 disables."
  )
  var idleTimeout: Double = 120

  @Option(name: .long, help: "Seconds idle before the model is unloaded entirely. 0 disables.")
  var evictTimeout: Double = 0

  @Option(name: .long, help: "Ask the OS to keep this many GB wired while serving. 0 disables.")
  var wireGB: Double = 0

  @Option(
    name: .long,
    help:
      "Pin MLX's reusable buffer cache, in GB. Default starts at 0.5 and doubles under pressure; 0 leaves it to MLX."
  )
  var cacheLimitGB: Double?

  @Option(
    name: .long,
    help: "Scheduling: adaptive (default), polite, normal, background.")
  var politeness: String = "adaptive"

  @Option(name: .long, help: "Stretch context past 262144 by this factor, e.g. 2 for ~512K.")
  var contextScale: Float = 1

  @Flag(name: .long, help: "Do not load the model until the first request arrives.")
  var lazyLoad = false

  @Flag(
    name: .long,
    help:
      "Load the model and its vision tower at startup rather than on demand. Wins over --lazy-load."
  )
  var hot = false

  @Flag(name: .long, help: "Disable coloured output.")
  var noColor = false

  @Flag(name: .long, help: "Log plain lines instead of the live dashboard.")
  var disableDashboard = false

  @Option(
    name: .long,
    help:
      "Disk budget for prefixes kept between runs, in GB. 0 turns the disk cache off."
  )
  var prefixCacheGB: Double = 8

  @Option(
    name: .long,
    help: "Shortest prefix worth archiving to disk, in tokens.")
  var prefixCacheMinimum: Int = 2048

  func run() throws {
    let kvConfig = KVCacheConfig(bits: kvBits, residualWindow: kvWindow)
    try kvConfig.validate()

    if noColor { Style.disable() }
    let level = Politeness.Level(rawValue: politeness) ?? .adaptive
    Politeness.apply(level)

    let modelURL = URL(filePath: resolvedModelPath(model, repo: repo))
    try neural.apply(pack: modelURL)
    if !offline {
      try ModelDownloader.ensure(directory: modelURL, repo: repo)
    }

    // Naming the active pack as the catalog does is what lets a client switch to it by name.
    let catalog = ModelCatalog.discover(in: modelSearchRoots)
    let activeName =
      servedName
      ?? catalog.entries.first { $0.directory.standardizedFileURL == modelURL.standardizedFileURL }?
      .id
      ?? modelURL.lastPathComponent

    let budget = MemoryBudget(
      kvBits: kvConfig.bits,
      maxContextTokens: Int(262_144 * max(contextScale, 1)),
      weights: MemoryBudget.weightBytes(in: modelURL) ?? MemoryBudget.defaultWeights,
      slots: cacheSlots,
      bufferCache: cacheLimitGB.map { Int($0 * 1_073_741_824) })
    let residency = ResidencyManager.Options(
      wiredBytes: Int(wireGB * 1_073_741_824),
      idleSeconds: idleTimeout,
      evictSeconds: evictTimeout)

    let server = try APIServer(
      directory: modelURL,
      modelName: activeName,
      samplingOptions: SamplingOptions(
        temperature: temperature, topP: topP, topK: topK, minP: minP),
      kvConfig: kvConfig,
      residency: residency,
      politeness: level,
      ropeScaling: contextScale > 1
        ? RopeScaling(method: .yarn, factor: contextScale) : .none,
      budget: budget,
      prefixStore: prefixCacheGB > 0
        ? PrefixStore(
          directory: prefixCacheDirectory,
          byteLimit: Int(prefixCacheGB * 1_073_741_824),
          minimumTokens: prefixCacheMinimum)
        : nil,
      catalog: catalog,
      preload: hot || !lazyLoad,
      hot: hot)

    var header = [
      Style.banner("serving \(activeName) on http://127.0.0.1:\(port)"),
      "",
      "  "
        + Style.field("OpenAI", Style.faint("export OPENAI_BASE_URL=http://127.0.0.1:\(port)/v1")),
      "  "
        + Style.field(
          "Anthropic", Style.faint("export ANTHROPIC_BASE_URL=http://127.0.0.1:\(port)")),
    ]
    if !budget.fitsFullContext {
      header.append(
        "  "
          + Style.field(
            "warning",
            Style.warn(
              "the ceiling holds about \(MemoryBudget.tokens(budget.maxContextThatFits)) tokens "
                + "of context, not the full \(MemoryBudget.tokens(budget.maxContextTokens)); "
                + "longer sessions fall back to re-prefill")))
    }

    if let notice = SelfUpdate.notice() {
      header.append("  " + Style.field("update", Style.warn(notice)))
    }

    var settings = [
      "  " + Style.field("scheduling", Style.faint(Politeness.describe(level))),
      "  "
        + Style.field(
          "budget",
          Style.faint(
            "starts at \(budget.describe()), doubling on demand "
              + "within a \(gigabytes(budget.ceiling)) ceiling")),
    ]
    if idleTimeout > 0 {
      settings.append(
        "  "
          + Style.field("idle", Style.faint("buffer pool released after \(Int(idleTimeout))s")))
    }
    if evictTimeout > 0 {
      settings.append(
        "  " + Style.field("evict", Style.faint("model unloaded after \(Int(evictTimeout))s")))
    }
    if wireGB > 0 {
      settings.append(
        "  " + Style.field("wired", Style.faint("\(wireGB) GB requested while serving")))
    }

    if disableDashboard || !ServeDashboard.isSupported {
      server.log = { message in
        FileHandle.standardError.write(Data("[bonsai] \(message)\n".utf8))
      }
      try server.listen(port: port)
      for line in header + settings { print(line) }
      print("  " + Style.field("memory", Style.faint(ResidencyManager.describeMemory())))
      fflush(stdout)
      installFarewell(for: server)
    } else {
      let dashboard = ServeDashboard(server: server, header: header)
      server.log = { [dashboard] message in dashboard.append(log: message) }
      try server.listen(port: port)
      dashboard.start()
    }

    dispatchMain()
  }

  private func gigabytes(_ bytes: Int) -> String {
    bytes == 0 ? "unbounded" : String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
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
  @Option(name: .long) var label: String = "studio.ishizuki.server"
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
