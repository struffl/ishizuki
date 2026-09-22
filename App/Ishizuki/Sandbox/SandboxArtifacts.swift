// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The two things a VM cannot be booted without: a Linux kernel and an init filesystem. Neither
// is ours to ship, but neither needs apple/container installed either: the init filesystem
// comes from an OCI image (below), and the kernel — if one isn't already sitting on disk from
// an apple/container install or a path someone pointed us at — is fetched once from the same
// Kata Containers release Apple's own Containerization build fetches for its own tests, then
// cached under our own state directory.

import Containerization
import Foundation
import IshizukiKit

struct SandboxArtifacts: Sendable {
  /// Where root filesystems and other per-container state are kept.
  let state: URL
  let kernelPath: String?
  let initfsPath: String?
  let initfsReference: String

  init(kernelPath: String?, initfsPath: String?, initfsReference: String) {
    self.state =
      (FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
      ?? URL(filePath: NSTemporaryDirectory()))
      .appending(path: "studio.ishizuki/sandbox")
    self.kernelPath = kernelPath
    self.initfsPath = initfsPath
    self.initfsReference = initfsReference
  }

  func kernel(
    for platform: SystemPlatform, report: @Sendable (SandboxPhase) -> Void = { _ in }
  ) async throws -> Kernel {
    if let found = kernelPath ?? Self.kernelCandidates.first(where: Self.exists) {
      return Kernel(path: URL(filePath: found), platform: platform)
    }
    let fetched = try await Self.fetchKernel(into: state, report: report)
    return Kernel(path: fetched, platform: platform)
  }

  /// The init filesystem, as a block if someone has one and as an image otherwise. vminitd is
  /// what makes the VM answerable at all, so this is the piece worth a clear refusal.
  func initfs(for platform: SystemPlatform) async throws -> Containerization.Mount {
    if let path = initfsPath ?? Self.initfsCandidates.first(where: Self.exists) {
      return Containerization.Mount.block(
        format: "ext4", source: path, destination: "/", options: ["ro"])
    }
    let store = ImageStore.default
    let image = try await store.getInitImage(reference: initfsReference)
    try? FileManager.default.createDirectory(at: state, withIntermediateDirectories: true)
    return try await image.initBlock(
      at: state.appending(path: "initfs-\(platform.architecture).ext4"), for: platform)
  }

  /// Where an apple/container install leaves its kernel, plus the copy we fetch ourselves.
  /// Checked in the order someone is likeliest to have one.
  static let kernelCandidates = [
    ownedKernelPath,
    home("Library/Application Support/com.apple.container/kernel/vmlinux"),
    home("Library/Application Support/com.apple.containerization/kernel/vmlinux"),
    home(".containerization/kernel/vmlinux"),
    "/opt/kata/share/kata-containers/vmlinux.container",
  ]

  /// The pinned Kata release Containerization's own `make fetch-default-kernel` uses. Apple
  /// silicon only, since that's the only host this app's VMs run on.
  private static let kataRelease = "3.17.0"
  private static var kataDownloadURL: URL {
    URL(
      string:
        "https://github.com/kata-containers/kata-containers/releases/download/\(kataRelease)/kata-static-\(kataRelease)-arm64.tar.xz"
    )!
  }

  private static var ownedKernelPath: String {
    (FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
    ?? URL(filePath: NSTemporaryDirectory()))
      .appending(path: "studio.ishizuki/sandbox/kernel/vmlinux").path
  }

  /// Downloads and unpacks the Kata release tarball, keeping only the one file a VM needs.
  /// `tar` is a system utility present on every Mac, not the apple/container CLI — this is the
  /// same tarball Containerization's own build fetches for its integration tests.
  private static func fetchKernel(
    into state: URL, report: @Sendable (SandboxPhase) -> Void
  ) async throws -> URL {
    let destination = URL(filePath: ownedKernelPath)
    if FileManager.default.fileExists(atPath: destination.path) { return destination }

    report(.starting("fetching a Linux kernel for the VM"))
    let (downloaded, response) = try await URLSession.shared.download(from: kataDownloadURL)
    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
      throw SandboxError.failed(
        "fetching a Linux kernel", "unexpected response from \(kataDownloadURL)")
    }

    let workDir = state.appending(path: "kernel-fetch-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: workDir) }
    let archive = workDir.appending(path: "kata.tar.xz")
    try FileManager.default.moveItem(at: downloaded, to: archive)

    guard let tar = Tooling.locate("tar") else {
      throw SandboxError.missingTool("tar")
    }
    let extraction = Process()
    extraction.executableURL = URL(filePath: tar)
    extraction.arguments = ["-xJf", archive.path, "-C", workDir.path]
    try extraction.run()
    extraction.waitUntilExit()
    guard extraction.terminationStatus == 0 else {
      throw SandboxError.failed(
        "unpacking the kernel", "tar exited \(extraction.terminationStatus)")
    }

    let unpacked = workDir.appending(path: "opt/kata/share/kata-containers/vmlinux.container")
    try FileManager.default.createDirectory(
      at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: unpacked, to: destination)
    return destination
  }

  static let initfsCandidates = [
    home("Library/Application Support/com.apple.container/initfs/initfs.ext4"),
    home("Library/Application Support/com.apple.containerization/initfs/initfs.ext4"),
  ]

  static var isReady: Bool {
    kernelCandidates.contains(where: exists)
  }

  private static func exists(_ path: String) -> Bool {
    FileManager.default.fileExists(atPath: path)
  }

  private static func home(_ path: String) -> String {
    FileManager.default.homeDirectoryForCurrentUser.appending(path: path).path
  }
}
