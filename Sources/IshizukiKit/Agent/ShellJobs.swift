// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Commands that outlive the call that started them. A turn waits a short while and then walks
// away with a job id, which is the only shape a build, a test run or a dev server has that
// fits inside a conversation.

import Foundation

public struct ShellJob: Codable, Sendable, Equatable, Identifiable {
  public enum State: String, Codable, Sendable {
    case running
    case exited
    case killed
  }

  public var id: String
  public var command: String
  public var pid: Int32
  public var started: Date
  public var state: State
  public var exitCode: Int32?
  public var seconds: Double
  /// Bytes the job has written that nobody has read yet, across both streams.
  public var pending: Int

  public var isRunning: Bool { state == .running }

  public init(
    id: String, command: String, pid: Int32, started: Date, state: State,
    exitCode: Int32? = nil, seconds: Double, pending: Int
  ) {
    self.id = id
    self.command = command
    self.pid = pid
    self.started = started
    self.state = state
    self.exitCode = exitCode
    self.seconds = seconds
    self.pending = pending
  }
}

/// What a job has written since it was last read, and where that leaves it.
public struct ShellJobOutput: Codable, Sendable, Equatable {
  public var job: ShellJob
  public var stdout: String
  public var stderr: String
  /// Bytes that fell off the front of the buffer before anyone asked for them.
  public var skipped: Int
  /// Bytes still waiting after this slice, because the slice hit its ceiling.
  public var remaining: Int

  public init(
    job: ShellJob, stdout: String, stderr: String, skipped: Int = 0, remaining: Int = 0
  ) {
    self.job = job
    self.stdout = stdout
    self.stderr = stderr
    self.skipped = skipped
    self.remaining = remaining
  }
}

#if os(macOS)

  /// The jobs one workspace has running. Held apart from the host so a shell that is built per
  /// request still finds the processes the last request left behind.
  public actor ShellJobs {
    private var jobs: [String: Job] = [:]
    private var order: [String] = []
    private var sequence = 0
    private let retain: Int
    private let finishedLimit: Int

    public init(retain: Int = 128 * 1024, finishedLimit: Int = 16) {
      self.retain = retain
      self.finishedLimit = finishedLimit
    }

    public func start(
      command: String, cwd: URL, shell: String, environment: [String: String]
    ) throws -> ShellJob {
      let process = Process()
      process.executableURL = URL(fileURLWithPath: shell)
      process.arguments = ["-l", "-c", command]
      process.currentDirectoryURL = cwd
      process.environment = environment

      let out = Pipe()
      let err = Pipe()
      process.standardOutput = out
      process.standardError = err
      process.standardInput = FileHandle.nullDevice

      sequence += 1
      let id = "job\(sequence)"
      let job = Job(id: id, command: command, process: process, retain: retain)

      try process.run()
      Live.shared.add(process.processIdentifier)
      job.drain(out, into: .out)
      job.drain(err, into: .err)

      jobs[id] = job
      order.append(id)
      prune()
      return job.snapshot()
    }

    public func list() -> [ShellJob] {
      order.compactMap { jobs[$0]?.snapshot() }
    }

    /// Waits up to `wait` seconds for the job to finish, then hands over what it has written
    /// since the last read. A job that is finished and fully read is forgotten here.
    public func read(_ id: String, wait: Double, byteLimit: Int) async throws -> ShellJobOutput {
      guard let job = jobs[id] else { throw ShellError.noSuchJob(id) }
      await job.settle(within: wait)
      let taken = job.take(byteLimit)
      if !taken.job.isRunning, taken.remaining == 0 { forget(id) }
      return taken
    }

    @discardableResult
    public func stop(_ id: String, force: Bool) throws -> ShellJob {
      guard let job = jobs[id] else { throw ShellError.noSuchJob(id) }
      job.signal(force ? SIGKILL : SIGTERM)
      if !force {
        Task { [weak self] in
          try? await Task.sleep(for: .seconds(3))
          await self?.escalate(id)
        }
      }
      return job.snapshot()
    }

    public func stopAll() {
      for job in jobs.values where job.isRunning { job.signal(SIGKILL) }
    }

    /// Every background command this process has running, stopped at once and without an
    /// await to get to. A child of ours is not something to leave behind at quit.
    public nonisolated static func stopEverything() {
      Live.shared.stopAll()
    }

    private func escalate(_ id: String) {
      guard let job = jobs[id], job.isRunning else { return }
      job.signal(SIGKILL)
    }

    private func forget(_ id: String) {
      jobs[id] = nil
      order.removeAll { $0 == id }
    }

    private func prune() {
      var finished = order.filter { jobs[$0]?.isRunning == false }
      while finished.count > finishedLimit {
        forget(finished.removeFirst())
      }
    }
  }

  /// The pids of every command still running anywhere in this process, kept so quitting can
  /// take them with it. A pid leaves the set the moment its process is reaped, so nothing here
  /// can name a pid the system has since handed to someone else.
  private final class Live: @unchecked Sendable {
    static let shared = Live()

    private let lock = NSLock()
    private var pids: Set<Int32> = []

    func add(_ pid: Int32) {
      lock.lock()
      pids.insert(pid)
      lock.unlock()
    }

    func remove(_ pid: Int32) {
      lock.lock()
      pids.remove(pid)
      lock.unlock()
    }

    func stopAll() {
      lock.lock()
      let doomed = pids
      pids.removeAll()
      lock.unlock()

      for pid in doomed { Job.signal(pid, SIGTERM) }
      guard !doomed.isEmpty else { return }
      usleep(200_000)
      for pid in doomed where kill(pid, 0) == 0 { Job.signal(pid, SIGKILL) }
    }
  }

  /// One running command: the process, the two buffers its output lands in, and the bookkeeping
  /// that says how much of it has been handed over.
  private final class Job: @unchecked Sendable {
    enum Which {
      case out
      case err
    }

    let id: String
    let command: String
    let started = Date()

    private let process: Process
    private let out: Buffer
    private let err: Buffer
    private let lock = NSLock()
    private var open = 0
    private var signalled = false
    private var ended: Date?

    init(id: String, command: String, process: Process, retain: Int) {
      self.id = id
      self.command = command
      self.process = process
      self.out = Buffer(retain: retain)
      self.err = Buffer(retain: retain)
    }

    var isRunning: Bool { process.isRunning }

    func drain(_ pipe: Pipe, into which: Which) {
      lock.lock()
      open += 1
      lock.unlock()

      let buffer = which == .out ? out : err
      DispatchQueue.global(qos: .utility).async { [self] in
        let handle = pipe.fileHandleForReading
        while case let chunk = handle.availableData, !chunk.isEmpty {
          buffer.append(chunk)
        }
        lock.lock()
        open -= 1
        lock.unlock()
        if which == .out {
          process.waitUntilExit()
          Live.shared.remove(process.processIdentifier)
        }
      }
    }

    /// Polls rather than waits on the process: a job is read from several places and a clock
    /// that can be walked away from is worth more here than a wakeup that cannot.
    func settle(within seconds: Double) async {
      let deadline = Date().addingTimeInterval(max(0, seconds))
      while process.isRunning, Date() < deadline, !Task.isCancelled {
        try? await Task.sleep(for: .milliseconds(20))
      }
      let grace = Date().addingTimeInterval(0.25)
      while !process.isRunning, hasOpenStreams, Date() < grace, !Task.isCancelled {
        try? await Task.sleep(for: .milliseconds(10))
      }
    }

    func take(_ limit: Int) -> ShellJobOutput {
      let first = out.take(limit)
      let second = err.take(max(0, limit - first.text.utf8.count))
      return ShellJobOutput(
        job: snapshot(),
        stdout: first.text,
        stderr: second.text,
        skipped: first.skipped + second.skipped,
        remaining: first.remaining + second.remaining)
    }

    func signal(_ number: Int32) {
      guard process.isRunning else { return }
      lock.lock()
      signalled = true
      lock.unlock()
      Job.signal(process.processIdentifier, number)
    }

    /// Signals the whole process group where the child leads one, since a shell that spawned a
    /// server is not the process anybody wanted stopped.
    static func signal(_ pid: Int32, _ number: Int32) {
      let group = getpgid(pid)
      if group > 0, group != getpgrp() {
        kill(-group, number)
      } else {
        kill(pid, number)
      }
    }

    func snapshot() -> ShellJob {
      let running = process.isRunning
      lock.lock()
      if !running, ended == nil { ended = Date() }
      let finishedAt = ended
      let killed = signalled
      lock.unlock()

      let state: ShellJob.State = running ? .running : (killed ? .killed : .exited)
      return ShellJob(
        id: id,
        command: command,
        pid: process.processIdentifier,
        started: started,
        state: state,
        exitCode: running ? nil : process.terminationStatus,
        seconds: (finishedAt ?? Date()).timeIntervalSince(started),
        pending: out.pending + err.pending)
    }

    private var hasOpenStreams: Bool {
      lock.lock()
      defer { lock.unlock() }
      return open > 0
    }
  }

  /// A capped, read-once view of one stream. Output past the cap is dropped from the front and
  /// counted, so a chatty job costs a fixed amount of memory and says what it cost.
  private final class Buffer: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()
    private var dropped = 0
    private var delivered = 0
    private let retain: Int

    init(retain: Int) {
      self.retain = retain
    }

    func append(_ chunk: Data) {
      lock.lock()
      defer { lock.unlock() }
      data.append(chunk)
      if data.count > retain {
        let excess = data.count - retain
        data = Data(data.dropFirst(excess))
        dropped += excess
      }
    }

    var pending: Int {
      lock.lock()
      defer { lock.unlock() }
      return (dropped + data.count) - max(delivered, dropped)
    }

    func take(_ limit: Int) -> (text: String, skipped: Int, remaining: Int) {
      lock.lock()
      defer { lock.unlock() }
      let total = dropped + data.count
      let start = max(delivered, dropped)
      let skipped = max(0, dropped - delivered)
      let end = min(total, start + max(0, limit))
      let slice = data.dropFirst(start - dropped).prefix(end - start)
      delivered = end
      return (String(decoding: slice, as: UTF8.self), skipped, total - end)
    }
  }

#endif
