// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A Linux VM of this Mac's own, booted in this process through Containerization and presented
// as a shell the agent tools already expect. The folder is shared into it; nothing else of
// this Mac is.

import Containerization
import ContainerizationEXT4
import ContainerizationOCI
import Foundation
import IshizukiKit

@available(macOS 27.0, *)
actor ContainerSandbox: ExecTransport {
  let id: String
  let choice: SandboxChoice
  let hostWorkspace: URL
  let artifacts: SandboxArtifacts
  /// Told what the VM is doing, so the line over the composer can say so. A booted VM is a
  /// large thing to have bought without being shown it.
  private let report: @Sendable (SandboxPhase) -> Void

  private var manager: ContainerManager?
  private var container: LinuxContainer?
  private var booting: Task<Void, Error>?
  private var networked = false

  init(
    id: String, choice: SandboxChoice, hostWorkspace: URL, artifacts: SandboxArtifacts,
    report: @escaping @Sendable (SandboxPhase) -> Void
  ) {
    self.id = id
    self.choice = choice
    self.hostWorkspace = hostWorkspace
    self.artifacts = artifacts
    self.report = report
  }

  nonisolated var describes: String { "container \(choice.image)" }

  nonisolated var isAvailable: Bool { true }

  func prepare() async throws {
    if container != nil { return }
    if let booting {
      try await booting.value
      return
    }
    let work = Task { try await boot() }
    booting = work
    do {
      try await work.value
    } catch {
      booting = nil
      report(.failed(error.localizedDescription))
      throw error
    }
  }

  /// Pulls the image, unpacks a root filesystem and boots the VM with the workspace shared in.
  /// Long enough to be worth saying out loud, which is what the report is for.
  private func boot() async throws {
    let platform: SystemPlatform =
      choice.architecture == .amd64 ? .linuxAmd : .linuxArm
    let kernel = try artifacts.kernel(for: platform)

    report(.starting("fetching \(choice.image)"))

    var manager = try await ContainerManager(
      kernel: kernel,
      initfs: try await artifacts.initfs(for: platform),
      network: Self.network(),
      rosetta: choice.architecture == .amd64)
    networked = Self.network() != nil

    let store = manager.imageStore
    let image: Containerization.Image
    if let held = try? await store.get(reference: choice.image) {
      image = held
    } else {
      image = try await store.pull(
        reference: choice.image, platform: platform.ociPlatform())
    }

    report(.starting("unpacking a root filesystem"))
    let root = artifacts.state.appending(path: "rootfs-\(id).ext4")
    try? FileManager.default.createDirectory(
      at: artifacts.state, withIntermediateDirectories: true)
    let rootfs = try await EXT4Unpacker(capacityInBytes: 8 * 1024 * 1024 * 1024)
      .unpack(image, for: platform.ociPlatform(), at: root)

    report(.starting("booting"))
    let guest = SandboxChoice.guestWorkspace.path
    let host = hostWorkspace.path
    let cpus = choice.cpus
    let memory = UInt64(choice.memoryBytes)
    let made = try await manager.create(
      id, image: image, rootfs: rootfs, networking: Self.network() != nil
    ) { config in
      config.cpus = cpus
      config.memoryInBytes = memory
      config.process.arguments = ["/bin/sh", "-c", "sleep 2147483647"]
      config.process.workingDirectory = guest
      config.mounts.append(Containerization.Mount.share(source: host, destination: guest))
    }
    try await made.create()
    try await made.start()

    self.manager = manager
    self.container = made
    report(.running(choice.summary + (networked ? "" : " · no network")))
  }

  /// vmnet needs an entitlement Apple hands out sparingly. Without it the VM still runs and
  /// the folder is still shared; it just cannot reach the network, which is said out loud
  /// rather than discovered halfway through an install.
  private static func network() -> (any Network)? {
    try? VmnetNetwork()
  }

  func exec(_ script: String, stdin: Data?, timeout: Double, byteLimit: Int) async throws
    -> ShellResult
  {
    let captured = try await run(script, stdin: stdin, timeout: timeout)
    func text(_ data: Data) -> (String, Bool) {
      data.count > byteLimit
        ? (String(decoding: data.prefix(byteLimit), as: UTF8.self), true)
        : (String(decoding: data, as: UTF8.self), false)
    }
    let out = text(captured.stdout)
    let err = text(captured.stderr)
    return ShellResult(
      stdout: out.0, stderr: err.0, exitCode: captured.code, truncated: out.1 || err.1)
  }

  func capture(_ script: String, stdin: Data?, timeout: Double) async throws -> (
    data: Data, exitCode: Int32
  ) {
    let captured = try await run(script, stdin: stdin, timeout: timeout)
    return (captured.stdout, captured.code)
  }

  private func run(_ script: String, stdin: Data?, timeout: Double) async throws -> (
    stdout: Data, stderr: Data, code: Int32
  ) {
    try await prepare()
    guard let container else { throw SandboxError.failed("the container", "it is not running") }

    let out = Collector()
    let err = Collector()
    let process = try await container.exec("sh-\(UUID().uuidString.prefix(8))") { config in
      config.arguments = ["/bin/sh", "-c", script]
      config.workingDirectory = SandboxChoice.guestWorkspace.path
      config.stdout = out
      config.stderr = err
      if let stdin { config.stdin = Chunk(data: stdin) }
    }
    try await process.start()
    let status = try await process.wait(timeoutInSeconds: Int64(max(1, timeout.rounded())))
    try? await process.delete()
    return (out.contents, err.contents, status.exitCode)
  }

  /// Stops the VM and takes its root filesystem with it. What was done to the shared folder
  /// stays on the Mac; everything the container installed does not.
  func teardown() async {
    if let container {
      try? await container.stop()
    }
    container = nil
    booting = nil
    if var manager {
      try? manager.delete(id)
      self.manager = nil
    }
    try? FileManager.default.removeItem(
      at: artifacts.state.appending(path: "rootfs-\(id).ext4"))
    report(.off)
  }

  /// One exec's output, gathered as it arrives.
  private final class Collector: Writer, @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()

    var contents: Data {
      lock.lock()
      defer { lock.unlock() }
      return data
    }

    func write(_ more: Data) throws {
      lock.lock()
      data.append(more)
      lock.unlock()
    }

    func close() throws {}
  }

  /// Everything a command is given on stdin, in one go.
  private struct Chunk: ReaderStream {
    let data: Data

    func stream() -> AsyncStream<Data> {
      AsyncStream { continuation in
        continuation.yield(data)
        continuation.finish()
      }
    }
  }
}

enum SandboxPhase: Sendable, Equatable {
  case off
  case starting(String)
  case running(String)
  case failed(String)

  var isUp: Bool { if case .running = self { true } else { false } }
  var isBusy: Bool { if case .starting = self { true } else { false } }

  var detail: String {
    switch self {
    case .off: "off"
    case .starting(let what): what
    case .running(let what): what
    case .failed(let why): why
    }
  }
}
