// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation

public final class ServeStats: @unchecked Sendable {
  public enum Phase: String, Sendable {
    case queued
    case prefill
    case decode
    case finishing
  }

  public struct Request: Sendable {
    public var id: Int
    public var api: String
    public var stream = false
    public var phase: Phase = .queued
    public var arrived = Date()
    public var phaseStart = Date()
    public var promptTokens = 0
    public var cachedTokens = 0
    public var prefilled = 0
    public var prefillTotal = 0
    public var generated = 0
    public var maxTokens = 0

    public var elapsed: Double { -arrived.timeIntervalSinceNow }
    public var phaseSeconds: Double { -phaseStart.timeIntervalSinceNow }

    public var rate: Double {
      let seconds = phaseSeconds
      guard seconds > 0 else { return 0 }
      switch phase {
      case .prefill: return Double(prefilled) / seconds
      case .decode: return Double(generated) / seconds
      case .queued, .finishing: return 0
      }
    }
  }

  public struct Totals: Sendable {
    public var started = Date()
    public var arrived = 0
    public var completed = 0
    public var failed = 0
    public var cancelled = 0
    public var promptTokens = 0
    public var prefilledTokens = 0
    public var cachedTokens = 0
    public var generatedTokens = 0
    public var prefillSeconds = 0.0
    public var decodeSeconds = 0.0
    public var cacheHits = 0
    public var cacheMisses = 0
    public var lastPrefillRate = 0.0
    public var lastDecodeRate = 0.0

    public var prefillRate: Double {
      prefillSeconds > 0 ? Double(prefilledTokens) / prefillSeconds : 0
    }
    public var decodeRate: Double {
      decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : 0
    }
    public var cacheRatio: Double {
      let seen = cacheHits + cacheMisses
      return seen > 0 ? Double(cacheHits) / Double(seen) : 0
    }
    public var totalTokens: Int { promptTokens + generatedTokens }
    public var uptime: Double { -started.timeIntervalSinceNow }
  }

  public struct Snapshot: Sendable {
    public var inFlight: [Request]
    public var totals: Totals

    public var running: Int { inFlight.filter { $0.phase != .queued }.count }
    public var queued: Int { inFlight.filter { $0.phase == .queued }.count }
  }

  private let lock = NSLock()
  private var requests: [Int: Request] = [:]
  private var order: [Int] = []
  private var totals = Totals()
  private var nextID = 1
  private var recorded: Set<Int> = []

  public init() {}

  public func enqueue(for path: String) -> Int? {
    let route = path.split(separator: "?").first.map(String.init) ?? path
    switch route {
    case "/v1/chat/completions": return enqueue(api: "openai")
    case "/v1/messages": return enqueue(api: "anthropic")
    default: return nil
    }
  }

  public func enqueue(api: String) -> Int {
    lock.lock()
    defer { lock.unlock() }
    let id = nextID
    nextID += 1
    requests[id] = Request(id: id, api: api)
    order.append(id)
    totals.arrived += 1
    return id
  }

  public func update(_ id: Int?, _ mutate: (inout Request) -> Void) {
    guard let id else { return }
    lock.lock()
    defer { lock.unlock() }
    guard var request = requests[id] else { return }
    mutate(&request)
    requests[id] = request
  }

  public func enter(_ id: Int?, phase: Phase) {
    update(id) { request in
      guard request.phase != phase else { return }
      request.phase = phase
      request.phaseStart = Date()
    }
  }

  public func record(_ id: Int?, generation: GenerationStats, cached: Int, reused: Bool) {
    lock.lock()
    defer { lock.unlock() }
    totals.completed += 1
    totals.promptTokens += generation.promptTokens + cached
    totals.prefilledTokens += generation.promptTokens
    totals.cachedTokens += cached
    totals.generatedTokens += generation.generatedTokens
    totals.prefillSeconds += generation.promptSeconds
    totals.decodeSeconds += generation.generationSeconds
    totals.lastPrefillRate = generation.promptTokensPerSecond
    totals.lastDecodeRate = generation.generationTokensPerSecond
    if reused { totals.cacheHits += 1 } else { totals.cacheMisses += 1 }
    if let id { recorded.insert(id) }
  }

  public func cancel(_ id: Int?) {
    guard let id else { return }
    lock.lock()
    defer { lock.unlock() }
    totals.cancelled += 1
    recorded.insert(id)
  }

  public func end(_ id: Int?) {
    guard let id else { return }
    lock.lock()
    defer { lock.unlock() }
    if requests.removeValue(forKey: id) != nil, !recorded.contains(id) {
      totals.failed += 1
    }
    recorded.remove(id)
    order.removeAll { $0 == id }
  }

  public func snapshot() -> Snapshot {
    lock.lock()
    defer { lock.unlock() }
    return Snapshot(inFlight: order.compactMap { requests[$0] }, totals: totals)
  }
}
