// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct Launch: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "launch",
    abstract: "Launch a coding tool wired to a local ishizuki server.",
    discussion: """
      Starts the server (or attaches to one already listening), points the tool at it,
      and hands the terminal over. The server shuts down when the tool exits.

        ishizuki launch hermes
        ishizuki launch claude
        ishizuki launch pi
        ishizuki launch list

      claude is configured with environment variables only. hermes and pi need a named
      provider entry in their own config, so those files are edited additively, with a
      .bak copy written first. Pass --print-config to see the changes without applying
      them or starting anything.
      """)

  @Argument(help: "Tool to launch: claude, hermes, pi, or list.")
  var tool: String

  @Argument(parsing: .postTerminator, help: "Arguments forwarded to the tool, after --.")
  var arguments: [String] = []

  @Option(
    name: .long,
    help: "Pack to serve. Omit to pick from what is on this machine, when there is a choice.")
  var model: String?
  @Option(name: .long, help: "HuggingFace repo to fetch the model from if missing.")
  var repo: String = defaultRepo
  @Flag(name: .long, help: "Skip the model download/repair check.") var offline = false

  @Option(name: .shortAndLong) var port: UInt16 = 8128
  @Option(name: .long) var host: String = "127.0.0.1"
  @Option(
    name: .long,
    help: "Name reported to the tool. Defaults to the pack's own name in the catalog.")
  var servedName: String?
  @Option(name: .long, help: "API key the tool sends. Any value works; it is not checked.")
  var apiKey: String = "ishizuki"

  @Option(name: .long, help: "Quantize the KV cache to this many bits. Pass 16 for fp16.")
  var kvBits: Float = 3.5
  @Option(name: .long, help: "Seconds idle before the model is unloaded entirely. 0 disables.")
  var evictTimeout: Double = 0
  @Option(
    name: .long,
    help:
      "Pin MLX's reusable buffer cache, in GB. Default starts at 0.5 and doubles under pressure; 0 leaves it to MLX."
  )
  var cacheLimitGB: Double?
  @Option(name: .long, help: "Scheduling: adaptive (default), polite, normal, background.")
  var politeness: String = "adaptive"
  @Option(name: .long, help: "Context window advertised to the tool.")
  var contextWindow: Int = 262_144

  @Flag(name: .long, help: "Print the configuration changes and exit.")
  var printConfig = false
  @Flag(name: .long, help: "Do not start a server; assume one is already listening.")
  var noServe = false
  @Flag(name: .long, help: "Disable coloured output.")
  var noColor = false

  private var baseURL: String { "http://\(host):\(port)" }

  func run() throws {
    if noColor { Style.disable() }

    if tool == "list" {
      print(Style.banner("tools ishizuki can launch"))
      print("")
      for integration in Integration.all {
        print(
          "  "
            + Style.accent(
              integration.name.padding(
                toLength: 10, withPad: " ", startingAt: 0))
            + Style.faint(integration.summary))
      }
      print("")
      print(Style.faint("  ishizuki launch <tool> [-- args...]"))
      return
    }

    guard let integration = Integration.named(tool) else {
      throw ValidationError(
        "unknown tool '\(tool)'. Try: "
          + Integration.all.map(\.name).joined(separator: ", ") + ", or list")
    }
    guard let executable = which(integration.executable) else {
      throw BonsaiError.missingComponent(
        "\(integration.executable) is not on PATH — install it first")
    }

    // The tool's config carries the model name, so the pack has to be settled before the plan
    // is written — but --print-config changes nothing and should never stop to ask.
    let catalog = ModelCatalog.discover(in: modelSearchRoots)
    let chosen = try choose(from: catalog, mayAsk: !printConfig && !noServe)
    let activeName = servedName ?? chosen?.id ?? defaultServedName

    let plan = try integration.plan(baseURL, activeName, apiKey, contextWindow)

    if printConfig {
      print(Style.banner("\(integration.name) configuration"))
      print("")
      for line in plan.describe() { print(line) }
      return
    }

    var server: APIServer?
    if !noServe && !isServing() {
      server = try startServer(chosen: chosen, name: activeName)
    } else {
      note(Style.field("server", Style.faint("already listening on \(baseURL)")))
    }

    for change in plan.fileChanges {
      let wrote = try change.apply()
      note(
        Style.field(
          "config",
          Style.accent(shortPath(change.path))
            + Style.faint(wrote ? "  provider added" : "  already configured")))
    }

    if let notice = SelfUpdate.notice() {
      note(Style.field("update", Style.warn(notice)))
    }
    note(Style.field("launch", Style.accent(integration.executable) + Style.faint("  \(baseURL)")))
    note("")

    let process = Process()
    process.executableURL = URL(filePath: executable)
    process.arguments = plan.arguments + arguments
    process.environment = ProcessInfo.processInfo.environment.merging(plan.environment) { _, new in
      new
    }
    try process.run()
    let terminal = TerminalHandoff(to: process.processIdentifier)
    process.waitUntilExit()
    terminal.restore()

    if let server { Farewell.print(tokens: server.stats.snapshot().totals.totalTokens) }
    server?.stop()
    throw ExitCode(process.terminationStatus == 0 ? 0 : Int32(process.terminationStatus))
  }

  /// Which pack to serve. An explicit --model wins; one pack needs no asking; several with a
  /// terminal to ask in gets a selector. Returns nil when the old path applies — nothing
  /// discovered, so fall back to the default pack and fetch it if it is missing.
  private func choose(
    from catalog: ModelCatalog, mayAsk: Bool
  ) throws -> ModelCatalog.Entry? {
    if let model {
      // A catalog name is as good as a path here, so either spelling works.
      return catalog[model]
    }
    if catalog.entries.count == 1 { return catalog.entries[0] }
    guard catalog.entries.count > 1 else { return nil }
    guard mayAsk else { return catalog.entries.first }

    let rows = catalog.entries.map {
      Picker.Row(title: $0.id, detail: ModelsCommand.describe($0))
    }
    switch Picker.run(title: "which model should \(tool) talk to?", rows: rows) {
    case .chose(let index): return catalog.entries[index]
    case .delete, .cancelled: throw ExitCode.failure
    }
  }

  private func startServer(
    chosen: ModelCatalog.Entry?, name activeName: String
  ) throws -> APIServer {
    let kvConfig = KVCacheConfig(bits: kvBits, residualWindow: 128)
    try kvConfig.validate()
    let level = Politeness.Level(rawValue: politeness) ?? .adaptive
    Politeness.apply(level)

    let catalog = ModelCatalog.discover(in: modelSearchRoots)
    let modelURL: URL
    if let chosen {
      modelURL = chosen.directory
    } else {
      modelURL = URL(filePath: resolvedModelPath(model ?? defaultModelPath, repo: repo))
      if !offline {
        try ModelDownloader.ensure(directory: modelURL, repo: repo)
      }
    }

    note(Style.banner("starting ishizuki for \(tool)"))
    note("")
    let start = Date()
    let server = try APIServer(
      directory: modelURL,
      modelName: activeName,
      kvConfig: kvConfig,
      residency: ResidencyManager.Options(idleSeconds: 0, evictSeconds: evictTimeout),
      politeness: level,
      budget: MemoryBudget(
        kvBits: kvConfig.bits,
        maxContextTokens: contextWindow,
        weights: MemoryBudget.weightBytes(in: modelURL) ?? MemoryBudget.defaultWeights,
        bufferCache: cacheLimitGB.map { Int($0 * 1_073_741_824) }),
      catalog: catalog,
      catalogRoots: modelSearchRoots,
      preload: true)
    server.log = { _ in }
    try server.listen(port: port)
    note(
      Style.field("server", Style.accent(baseURL))
        + Style.faint(String(format: "  ready in %.1fs", -start.timeIntervalSinceNow)))
    return server
  }

  private func isServing() -> Bool {
    guard let url = URL(string: baseURL + "/health") else { return false }
    var request = URLRequest(url: url)
    request.timeoutInterval = 1.5
    let done = DispatchSemaphore(value: 0)
    let reachable = Reachable()
    URLSession.shared.dataTask(with: request) { _, response, _ in
      reachable.value = (response as? HTTPURLResponse)?.statusCode == 200
      done.signal()
    }.resume()
    _ = done.wait(timeout: .now() + 3)
    return reachable.value
  }

  private func which(_ name: String) -> String? {
    let path = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
    for directory in path.split(separator: ":") {
      let candidate = String(directory) + "/" + name
      if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
    }
    return nil
  }

  private func shortPath(_ path: String) -> String {
    path.hasPrefix(NSHomeDirectory()) ? "~" + path.dropFirst(NSHomeDirectory().count) : path
  }
}

private final class Reachable: @unchecked Sendable {
  var value = false
}

/// Lends the controlling terminal to the spawned tool.
///
/// Foundation spawns children into a process group of their own, which leaves them in the
/// background: the first raw-mode or tty read a full-screen tool makes stops it on SIGTTIN or
/// SIGTTOU before it ever paints. Handing the foreground over for the duration fixes that.
private final class TerminalHandoff {
  private let previous: pid_t
  private let lent: Bool

  init(to child: pid_t) {
    guard isatty(STDIN_FILENO) == 1 else {
      previous = -1
      lent = false
      return
    }
    signal(SIGTTOU, SIG_IGN)
    signal(SIGTTIN, SIG_IGN)
    previous = tcgetpgrp(STDIN_FILENO)
    lent = tcsetpgrp(STDIN_FILENO, child) == 0
    if lent { kill(-child, SIGCONT) }
  }

  func restore() {
    if lent, previous > 0 { tcsetpgrp(STDIN_FILENO, previous) }
    guard previous != -1 else { return }
    signal(SIGTTOU, SIG_DFL)
    signal(SIGTTIN, SIG_DFL)
  }
}

struct LaunchPlan {
  var environment: [String: String] = [:]
  var arguments: [String] = []
  var fileChanges: [ConfigChange] = []

  func describe() -> [String] {
    var lines: [String] = []
    for (key, value) in environment.sorted(by: { $0.key < $1.key }) {
      lines.append("  " + Style.muted("export ") + Style.accent(key) + "=" + Style.faint(value))
    }
    if !arguments.isEmpty {
      lines.append("  " + Style.muted("args   ") + Style.faint(arguments.joined(separator: " ")))
    }
    for change in fileChanges {
      lines.append("")
      lines.append("  " + Style.muted("file   ") + Style.accent(change.path))
      for line in change.snippet.split(separator: "\n", omittingEmptySubsequences: false) {
        lines.append("    " + Style.faint(String(line)))
      }
    }
    return lines
  }
}

struct ConfigChange {
  var path: String
  var snippet: String
  var apply: () throws -> Bool
}

struct Integration: Sendable {
  var name: String
  var executable: String
  var summary: String
  var plan:
    @Sendable (_ baseURL: String, _ model: String, _ apiKey: String, _ contextWindow: Int) throws ->
      LaunchPlan

  static var all: [Integration] { [hermes, claude, pi] }

  static func named(_ name: String) -> Integration? {
    all.first { $0.name == name.lowercased() }
  }

  static let claude = Integration(
    name: "claude", executable: "claude",
    summary: "Claude Code — Anthropic API, environment only"
  ) { baseURL, model, apiKey, contextWindow in
    LaunchPlan(
      environment: [
        "ANTHROPIC_BASE_URL": baseURL,
        "ANTHROPIC_AUTH_TOKEN": apiKey,
        "ANTHROPIC_API_KEY": apiKey,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "\(contextWindow)",
      ])
  }

  static let hermes = Integration(
    name: "hermes", executable: "hermes",
    summary: "Hermes Agent — OpenAI API, provider entry in config.yaml"
  ) { baseURL, model, apiKey, contextWindow in
    let home = ProcessInfo.processInfo.environment["HERMES_HOME"] ?? NSHomeDirectory() + "/.hermes"
    let path = home + "/config.yaml"
    let block = """
        ishizuki:
          name: Ishizuki
          base_url: \(baseURL)/v1
          model: \(model)
          default_model: \(model)
          key_env: ISHIZUKI_API_KEY
          api_mode: chat_completions
          context_length: \(contextWindow)
      """
    return LaunchPlan(
      environment: ["ISHIZUKI_API_KEY": apiKey],
      arguments: ["--provider", "ishizuki", "--model", model],
      fileChanges: [
        ConfigChange(path: path, snippet: "providers:\n" + block) {
          try mergeYAMLProvider(path: path, key: "ishizuki", block: block)
        }
      ])
  }

  static let pi = Integration(
    name: "pi", executable: "pi",
    summary: "Pi — OpenAI API, provider entry in models.json"
  ) { baseURL, model, apiKey, contextWindow in
    let directory =
      ProcessInfo.processInfo.environment["PI_CODING_AGENT_DIR"]
      ?? NSHomeDirectory() + "/.pi/agent"
    let path = directory + "/models.json"
    let entry: [String: Any] = [
      "baseUrl": baseURL + "/v1",
      "api": "openai-completions",
      "apiKey": apiKey,
      "models": [
        [
          "id": model,
          "name": model,
          "reasoning": false,
          "input": ["text", "image"],
          "contextWindow": contextWindow,
          "maxTokens": 32768,
          "cost": ["input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0],
        ]
      ],
    ]
    let snippet =
      (try? String(
        data: JSONSerialization.data(
          withJSONObject: ["providers": ["ishizuki": entry]],
          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]), encoding: .utf8)) ?? ""
    return LaunchPlan(
      arguments: ["--provider", "ishizuki", "--model", model],
      fileChanges: [
        ConfigChange(path: path, snippet: snippet) {
          try mergeJSONProvider(path: path, key: "ishizuki", entry: entry)
        }
      ])
  }
}

private func backup(_ path: String) throws {
  guard FileManager.default.fileExists(atPath: path) else { return }
  let destination = path + ".bak"
  try? FileManager.default.removeItem(atPath: destination)
  try FileManager.default.copyItem(atPath: path, toPath: destination)
}

func mergeJSONProvider(path: String, key: String, entry: [String: Any]) throws -> Bool {
  try FileManager.default.createDirectory(
    at: URL(filePath: path).deletingLastPathComponent(), withIntermediateDirectories: true)

  var root: [String: Any] = [:]
  if let data = FileManager.default.contents(atPath: path),
    let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
  {
    root = parsed
  }
  var providers = root["providers"] as? [String: Any] ?? [:]
  if let existing = providers[key] as? [String: Any],
    existing["baseUrl"] as? String == entry["baseUrl"] as? String
  {
    return false
  }
  try backup(path)
  providers[key] = entry
  root["providers"] = providers
  let data = try JSONSerialization.data(
    withJSONObject: root, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
  try data.write(to: URL(filePath: path), options: .atomic)
  return true
}

func mergeYAMLProvider(path: String, key: String, block: String) throws -> Bool {
  try FileManager.default.createDirectory(
    at: URL(filePath: path).deletingLastPathComponent(), withIntermediateDirectories: true)

  guard let existing = try? String(contentsOfFile: path, encoding: .utf8) else {
    try ("providers:\n" + block + "\n").write(
      toFile: path, atomically: true, encoding: .utf8)
    return true
  }

  var lines = existing.components(separatedBy: "\n")
  if let providersIndex = lines.firstIndex(where: {
    $0 == "providers:" || $0.hasPrefix("providers:")
  }) {
    let alreadyPresent = lines[providersIndex...]
      .prefix { $0 == lines[providersIndex] || $0.hasPrefix(" ") || $0.isEmpty }
      .contains { $0.trimmingCharacters(in: .whitespaces) == "\(key):" }
    if alreadyPresent { return false }
    try backup(path)
    lines.insert(
      contentsOf: block.components(separatedBy: "\n"), at: providersIndex + 1)
  } else {
    try backup(path)
    if lines.last?.isEmpty == true { lines.removeLast() }
    lines.append("providers:")
    lines.append(contentsOf: block.components(separatedBy: "\n"))
  }
  try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
  return true
}
