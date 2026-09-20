// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Owns the in-process APIServer: start, stop, switch pack, and poll its readout.

import Foundation
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

  let settings = ServerSettings()
  let library = ModelLibrary()

  private var server: APIServer?
  private var idleStore: PrefixStore?
  private var ticker: Task<Void, Never>?
  private let logLimit = 200

  private var bootstrapped = false

  init() {
    rescan()
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
    catalog = ModelCatalog.discover(in: library.searchRoots())
    if catalog[settings.activeModelID] == nil {
      settings.activeModelID = catalog.entries.first?.id ?? ""
    }
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

    Task {
      do {
        try kvConfig.validate()
        let server = try await Self.build(
          url: url, name: name, kvConfig: kvConfig, level: level, maxContext: maxContext,
          contextScale: contextScale, residency: residency, prefixGB: prefixGB,
          catalog: catalog, roots: roots, preload: preload, hot: hot)
        server.log = { [weak self] message in
          Task { @MainActor in self?.append(message) }
        }
        try server.listen(port: port)
        self.server = server
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
    readout = nil
    phase = .stopped
  }

  func restart() {
    stop()
    start()
  }

  func activate(_ id: String) {
    settings.activeModelID = id
    guard phase.isRunning, let server else { return }
    Task.detached { [weak self] in
      do {
        try server.activate(id)
      } catch {
        await MainActor.run { self?.append("switch to \(id) failed: \(error)") }
      }
    }
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
        weights: MemoryBudget.weightBytes(in: url) ?? MemoryBudget.defaultWeights)
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
