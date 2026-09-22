// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What a new conversation's sandbox defaults to, the images that have been used before, and
// where the VM's kernel and init filesystem are. Kept between launches; none of it is secret.

import Foundation
import IshizukiKit
import Observation

@MainActor
@Observable
final class SandboxSettings {
  /// One set of these for the app: the panes and the picker are both looking at the same
  /// defaults, and a VM's kernel is not a per-conversation question.
  static let shared = SandboxSettings()

  var defaultChoice: SandboxChoice {
    didSet {
      guard let data = try? JSONEncoder().encode(defaultChoice) else { return }
      defaults.set(data, forKey: "sandbox.default")
    }
  }
  /// Images that have been asked for before, most recent first. Typing a name once is enough.
  var recentImages: [String] { didSet { defaults.set(recentImages, forKey: "sandbox.images") } }
  var kernelPath: String { didSet { defaults.set(kernelPath, forKey: "sandbox.kernel") } }
  var initfsPath: String { didSet { defaults.set(initfsPath, forKey: "sandbox.initfs") } }
  var initfsReference: String {
    didSet { defaults.set(initfsReference, forKey: "sandbox.initfsReference") }
  }

  private let defaults = UserDefaults.standard

  init() {
    let defaults = UserDefaults.standard
    defaultChoice =
      defaults.data(forKey: "sandbox.default")
      .flatMap { try? JSONDecoder().decode(SandboxChoice.self, from: $0) }
      ?? SandboxChoice(
        kind: .native, cpus: SandboxLimits.defaultCPUs,
        memoryBytes: SandboxLimits.defaultMemory)
    recentImages =
      defaults.stringArray(forKey: "sandbox.images") ?? [
        "docker.io/library/debian:bookworm-slim",
        "docker.io/library/ubuntu:24.04",
        "docker.io/library/alpine:3.20",
      ]
    kernelPath = defaults.string(forKey: "sandbox.kernel") ?? ""
    initfsPath = defaults.string(forKey: "sandbox.initfs") ?? ""
    initfsReference =
      defaults.string(forKey: "sandbox.initfsReference")
      ?? "ghcr.io/apple/containerization/vminit:latest"
  }

  func remember(image: String) {
    let trimmed = image.trimmingCharacters(in: .whitespaces)
    guard !trimmed.isEmpty else { return }
    recentImages = [trimmed] + recentImages.filter { $0 != trimmed }.prefix(7)
  }

  var artifacts: SandboxArtifacts {
    SandboxArtifacts(
      kernelPath: kernelPath.isEmpty ? nil : kernelPath,
      initfsPath: initfsPath.isEmpty ? nil : initfsPath,
      initfsReference: initfsReference)
  }
}

/// What this Mac can spare. A VM that takes every core and all the memory would be competing
/// with the pack that is answering, so the ceilings are the machine minus what is already
/// committed here.
enum SandboxLimits {
  static var cores: Int { ProcessInfo.processInfo.processorCount }

  /// Two cores stay with the host: one for the window, one for whatever the pack is doing.
  static var cpuRange: ClosedRange<Int> { 2...max(2, cores - 2) }

  static var defaultCPUs: Int { min(max(2, cores / 2), cpuRange.upperBound) }

  /// Memory a VM may take, in bytes: what is installed, less the model that is resident and
  /// the room the system needs to stay pleasant.
  static func memoryRange(residentBytes: Int) -> ClosedRange<Int> {
    let gigabyte = 1024 * 1024 * 1024
    let reserve = residentBytes + 6 * gigabyte
    let spare = max(2 * gigabyte, Machine.physicalMemory - reserve)
    return (2 * gigabyte)...max(2 * gigabyte, (spare / gigabyte) * gigabyte)
  }

  static var defaultMemory: Int {
    let gigabyte = 1024 * 1024 * 1024
    return min(8 * gigabyte, memoryRange(residentBytes: 0).upperBound)
  }
}
