// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The two things a VM cannot be booted without: a Linux kernel and an init filesystem. Neither
// is ours to ship, so this is where they are found — from an apple/container install, from a
// path someone pointed us at, or not at all, which is said plainly rather than crashed on.

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

  func kernel(for platform: SystemPlatform) throws -> Kernel {
    guard let found = kernelPath ?? Self.kernelCandidates.first(where: Self.exists) else {
      throw SandboxError.missingTool(
        "a Linux kernel for the VM — install apple/container, or point Settings at a vmlinux")
    }
    return Kernel(path: URL(filePath: found), platform: platform)
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

  /// Where an apple/container install leaves its kernel. Checked in the order someone is
  /// likeliest to have one.
  static let kernelCandidates = [
    home("Library/Application Support/com.apple.container/kernel/vmlinux"),
    home("Library/Application Support/com.apple.containerization/kernel/vmlinux"),
    home(".containerization/kernel/vmlinux"),
    "/opt/kata/share/kata-containers/vmlinux.container",
  ]

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
