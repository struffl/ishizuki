// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public final class ResidencyManager: @unchecked Sendable {
  public struct Options: Sendable {
    public var wiredBytes: Int
    public var cacheLimit: Int
    public var idleSeconds: Double
    public var evictSeconds: Double

    public init(
      wiredBytes: Int = 0, cacheLimit: Int = 0,
      idleSeconds: Double = 60, evictSeconds: Double = 0
    ) {
      self.wiredBytes = wiredBytes
      self.cacheLimit = cacheLimit
      self.idleSeconds = idleSeconds
      self.evictSeconds = evictSeconds
    }
  }

  public private(set) var options: Options
  public private(set) var isWired = false

  private var ticket: WiredMemoryTicket?
  private let lock = NSLock()
  private var lastActivity = Date()
  private var timer: DispatchSourceTimer?

  public var onIdle: (@Sendable () -> Void)?
  public var onEvict: (@Sendable () -> Void)?

  public init(options: Options) {
    self.options = options
    if options.cacheLimit > 0 {
      Memory.cacheLimit = options.cacheLimit
    }
  }

  public func beginRequest() {
    lock.lock()
    lastActivity = Date()
    lock.unlock()
    wire()
  }

  public func endRequest() {
    lock.lock()
    lastActivity = Date()
    lock.unlock()
  }

  public func startMonitoring(on queue: DispatchQueue) {
    guard options.idleSeconds > 0 || options.evictSeconds > 0 else { return }
    let timer = DispatchSource.makeTimerSource(queue: queue)
    timer.schedule(deadline: .now() + 5, repeating: 5)
    timer.setEventHandler { [weak self] in self?.checkIdle() }
    timer.resume()
    self.timer = timer
  }

  public func stopMonitoring() {
    timer?.cancel()
    timer = nil
  }

  private func checkIdle() {
    lock.lock()
    let idleFor = Date().timeIntervalSince(lastActivity)
    lock.unlock()

    if options.evictSeconds > 0, idleFor >= options.evictSeconds {
      unwire()
      Memory.clearCache()
      onEvict?()
      return
    }
    if options.idleSeconds > 0, idleFor >= options.idleSeconds {
      unwire()
      Memory.clearCache()
      onIdle?()
    }
  }

  public func wire() {
    guard options.wiredBytes > 0 else { return }
    lock.lock()
    defer { lock.unlock() }
    guard !isWired else { return }

    let ticket = WiredMemoryTicket(
      size: options.wiredBytes, policy: WiredSumPolicy(), manager: .shared, kind: .active)
    let semaphore = DispatchSemaphore(value: 0)
    Task {
      _ = await ticket.start()
      semaphore.signal()
    }
    _ = semaphore.wait(timeout: .now() + 2)
    self.ticket = ticket
    isWired = true
  }

  public func unwire() {
    lock.lock()
    defer { lock.unlock() }
    guard isWired, let ticket else { return }
    let semaphore = DispatchSemaphore(value: 0)
    Task {
      _ = await ticket.end()
      semaphore.signal()
    }
    _ = semaphore.wait(timeout: .now() + 2)
    self.ticket = nil
    isWired = false
  }

  public static func describeMemory() -> String {
    let snapshot = Memory.snapshot()
    func gigabytes(_ bytes: Int) -> String {
      String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
    }
    return "active \(gigabytes(snapshot.activeMemory)), "
      + "cache \(gigabytes(snapshot.cacheMemory)), "
      + "peak \(gigabytes(snapshot.peakMemory))"
  }
}
