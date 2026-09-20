// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One long job at a time — a check, a benchmark, a quantize — streaming its lines to a console.

import Foundation
import Observation

/// The handle the work writes through. It runs off the actor, so the flag and the sink are
/// both lock-guarded rather than isolated.
final class JobLog: @unchecked Sendable {
  private let lock = NSLock()
  private var cancelled = false
  private let sink: @Sendable (String) -> Void
  private let meter: @Sendable (Double?) -> Void

  init(
    sink: @escaping @Sendable (String) -> Void,
    meter: @escaping @Sendable (Double?) -> Void
  ) {
    self.sink = sink
    self.meter = meter
  }

  var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }

  func cancel() {
    lock.lock()
    cancelled = true
    lock.unlock()
  }

  func line(_ text: String) {
    for piece in text.split(separator: "\n", omittingEmptySubsequences: false) {
      sink(String(piece))
    }
  }

  func progress(_ fraction: Double?) {
    meter(fraction)
  }

  struct Cancelled: Error {}

  func checkCancellation() throws {
    if isCancelled { throw Cancelled() }
  }
}

@MainActor
@Observable
final class JobRunner {
  struct Job {
    var name: String
    var lines: [String] = []
    var progress: Double?
    var started = Date()
    var finished: Date?
    var failure: String?

    var isRunning: Bool { finished == nil }
    var elapsed: Double { (finished ?? Date()).timeIntervalSince(started) }
  }

  private(set) var job: Job?

  private var log: JobLog?
  private let lineLimit = 2000

  var isRunning: Bool { job?.isRunning == true }

  func run(_ name: String, _ body: @escaping @Sendable (JobLog) throws -> Void) {
    guard !isRunning else { return }
    job = Job(name: name)

    let log = JobLog(
      sink: { text in
        Task { @MainActor [weak self] in self?.append(text) }
      },
      meter: { fraction in
        Task { @MainActor [weak self] in self?.job?.progress = fraction }
      })
    self.log = log

    Task.detached(priority: .userInitiated) { [weak self] in
      var failure: String?
      do {
        try body(log)
      } catch is JobLog.Cancelled {
        failure = nil
      } catch {
        failure = String(describing: error)
      }
      let outcome = failure
      await MainActor.run { self?.finish(failure: outcome) }
    }
  }

  func cancel() {
    log?.cancel()
  }

  func clear() {
    guard !isRunning else { return }
    job = nil
  }

  private func append(_ text: String) {
    guard job != nil else { return }
    job?.lines.append(text)
    if let count = job?.lines.count, count > lineLimit {
      job?.lines.removeFirst(count - lineLimit)
    }
  }

  private func finish(failure: String?) {
    job?.finished = Date()
    job?.failure = failure
    job?.progress = nil
    log = nil
  }
}
