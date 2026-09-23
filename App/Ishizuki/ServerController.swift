// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Owns the in-process APIServer: start, stop, switch pack, and poll its readout.

import AppKit
import Foundation
import FoundationModels
import IshizukiKit
import Observation

@MainActor
@Observable
final class ServerController {
  enum Phase: Equatable {
    case stopped
    case starting(String)
    case running
    case failed(String)

    var isBusy: Bool { if case .starting = self { true } else { false } }
    var isRunning: Bool { self == .running }
  }

  private(set) var phase: Phase = .stopped
  private(set) var readout: ServeReadout?
  private(set) var log: [String] = []
  private(set) var catalog = ModelCatalog(entries: [])
  private(set) var appleIntelligenceUsable = SystemLanguageModel.default.isAvailable
  /// The pack being swapped in, while the swap waits its turn on the generation queue.
  private(set) var switching: String?

  var offeredAppleModels: [AppleFoundationModel] {
    appleIntelligenceUsable ? AppleFoundationModel.offered : []
  }

  let settings = ServerSettings()
  let library = ModelLibrary()
  let samplerSettings = SamplerSettingsStore()

  private var server: APIServer?
  /// Made once per loaded server, so the chat borrows the same weights the port is serving.
  private(set) var engine: AgentEngine?
  private var idleStore: PrefixStore?
  private var ticker: Task<Void, Never>?
  private let logLimit = 200

  private var bootstrapped = false
  /// The folders the catalog is built from, watched so a pack pulled in a terminal or thrown
  /// away in the Finder reaches the list on its own.
  private var watcher: FolderWatcher?
  /// A signal to quit becomes an ordinary quit, so the cache is written on the way out rather
  /// than lost with the process.
  private var signals: [DispatchSourceSignal] = []

  init() {
    rescan()
    let watcher = FolderWatcher { [weak self] in
      MainActor.assumeIsolated { self?.rescan() }
    }
    self.watcher = watcher
    watcher.watch(library.searchRoots())

    // A pack that arrived while the window was in the background is worth a look on the way
    // back in: a watcher can miss a move the file system reports to nobody.
    NotificationCenter.default.addObserver(
      forName: NSApplication.didBecomeActiveNotification, object: nil, queue: nil
    ) { [weak self] _ in
      MainActor.assumeIsolated {
        self?.rescan()
        self?.probeAppleIntelligence()
      }
    }
    probeAppleIntelligence()

    NotificationCenter.default.addObserver(
      forName: NSApplication.willTerminateNotification, object: nil, queue: nil
    ) { [weak self] _ in
      MainActor.assumeIsolated { self?.stop() }
    }
    for number in [SIGTERM, SIGINT, SIGHUP] {
      signal(number, SIG_IGN)
      let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
      source.setEventHandler { NSApp.terminate(nil) }
      source.resume()
      signals.append(source)
    }
  }

  func probeAppleIntelligence() {
    Task { [weak self] in
      let usable = await AppleIntelligenceProbe.isUsable()
      self?.setAppleIntelligenceUsable(usable)
    }
  }

  func setAppleIntelligenceUsable(_ usable: Bool) {
    appleIntelligenceUsable = usable
    if !usable, settings.appleModel != nil { settings.appleModel = nil }
  }

  func bootstrap() {
    guard !bootstrapped else { return }
    bootstrapped = true
    if settings.startOnLaunch, activeEntry != nil { start() }
  }

  var activeEntry: ModelCatalog.Entry? {
    catalog[settings.activeModelID] ?? catalog.entries.first
  }

  var baseURL: String { "http://127.0.0.1:\(settings.port)" }

  /// The running server's store when there is one, so archives are never deleted out from
  /// under the instance holding them.
  var prefixStore: PrefixStore {
    if let store = server?.prefixStore { return store }
    if let idle = idleStore { return idle }
    let store = PrefixStore(
      directory: IshizukiPaths.prefixCache,
      byteLimit: Int(settings.prefixCacheGB * 1_073_741_824))
    idleStore = store
    return store
  }

  func rescan() {
    let roots = library.searchRoots()
    let found = ModelCatalog.discover(in: roots)
    // Assigning an identical catalog would still redraw every row that reads it, and this now
    // runs whenever anything under a root is written to.
    if found.entries != catalog.entries { catalog = found }
    if catalog[settings.activeModelID] == nil {
      settings.activeModelID = catalog.entries.first?.id ?? ""
    }
    watcher?.watch(roots)
  }

  func start() {
    guard !phase.isBusy, !phase.isRunning else { return }
    guard let entry = activeEntry else {
      phase = .failed("No model installed yet.")
      return
    }

    phase = .starting(entry.displayName)
    log.removeAll()

    let settings = self.settings
    let port = UInt16(clamping: settings.port)
    let kvConfig = settings.kvConfig
    let level = settings.politenessLevel
    let maxContext = settings.maxContextTokens
    let contextScale = settings.contextScale
    let residency = ResidencyManager.Options(
      wiredBytes: Int(settings.wireGB * 1_073_741_824),
      idleSeconds: settings.idleTimeout,
      evictSeconds: settings.evictTimeout)
    let prefixGB = settings.prefixCacheGB
    let preload = settings.preload
    let hot = settings.hot
    let roots = library.searchRoots()
    let catalog = self.catalog
    let url = entry.url
    let name = entry.id

    let neural = settings.neuralEngine
    // Read as the pack is opened, so a streamed model's slot budget is whatever the dial said
    // when it was started rather than whatever it says now.
    BonsaiRuntime.expertSlots = settings.expertSlots

    Task {
      do {
        try kvConfig.validate()
        do {
          try ANEOffload.apply(neural ? .automatic : nil, pack: url)
        } catch {
          // A pack without slices is a reason to serve on the GPU alone, not to refuse.
          self.append("neural engine off: \(error)")
        }
        let server = try await Self.build(
          url: url, name: name, kvConfig: kvConfig, level: level, maxContext: maxContext,
          contextScale: contextScale, residency: residency, prefixGB: prefixGB,
          catalog: catalog, roots: roots, preload: preload, hot: hot)
        server.log = { [weak self] message in
          Task { @MainActor in self?.append(message) }
        }
        server.samplingOptions = self.samplerSettings.samplingOptions(for: name)
        try server.listen(port: port)
        self.server = server
        self.engine = AgentEngine(server: server)
        self.phase = .running
        self.startTicking()
      } catch {
        self.phase = .failed(String(describing: error))
      }
    }
  }

  func stop() {
    ticker?.cancel()
    ticker = nil
    server?.stop()
    server = nil
    engine = nil
    readout = nil
    phase = .stopped
  }

  func restart() {
    stop()
    start()
  }

  func activate(_ id: String) {
    settings.appleModel = nil
    settings.activeModelID = id
    guard phase.isRunning, let server, let engine else { return }
    let options = samplerSettings.samplingOptions(for: id)
    switching = id
    Task { [weak self] in
      do {
        try await engine.activate(id)
        server.samplingOptions = options
      } catch {
        self?.append("switch to \(id) failed: \(error)")
      }
      if self?.switching == id { self?.switching = nil }
    }
  }

  /// Called as the sampler sheet is edited, so a pack already answering picks up the new
  /// numbers on its next turn rather than waiting for the next activation.
  func updateSamplerSettings(_ new: SamplerSettings, for id: String) {
    samplerSettings.set(new, for: id)
    if id == settings.activeModelID { server?.samplingOptions = new.samplingOptions }
  }

  private func append(_ message: String) {
    log.append(message)
    if log.count > logLimit { log.removeFirst(log.count - logLimit) }
  }

  private func startTicking() {
    ticker?.cancel()
    ticker = Task { [weak self] in
      while !Task.isCancelled {
        guard let self, let server = self.server else { return }
        self.readout = server.readout()
        try? await Task.sleep(for: .milliseconds(250))
      }
    }
  }

  /// The pack is read on a background thread: a 27B load holds for seconds and must not
  /// stall the status bar.
  private nonisolated static func build(
    url: URL, name: String, kvConfig: KVCacheConfig, level: Politeness.Level, maxContext: Int,
    contextScale: Double, residency: ResidencyManager.Options, prefixGB: Double,
    catalog: ModelCatalog, roots: [URL], preload: Bool, hot: Bool
  ) async throws -> APIServer {
    try await Task.detached(priority: .userInitiated) {
      let budget = MemoryBudget(
        kvBits: kvConfig.bits,
        maxContextTokens: maxContext,
        weights: StreamedPlan.residentBytes(in: url) ?? MemoryBudget.defaultWeights)
      return try APIServer(
        directory: url,
        modelName: name,
        kvConfig: kvConfig,
        residency: residency,
        politeness: level,
        ropeScaling: contextScale > 1
          ? RopeScaling(method: .yarn, factor: Float(contextScale)) : .none,
        budget: budget,
        prefixStore: prefixGB > 0
          ? PrefixStore(
            directory: IshizukiPaths.prefixCache,
            byteLimit: Int(prefixGB * 1_073_741_824))
          : nil,
        catalog: catalog,
        catalogRoots: roots,
        preload: preload,
        hot: hot)
    }.value
  }
}
