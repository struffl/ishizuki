// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// How many of a streamed pack's experts this machine holds per layer, and what that costs.

import Foundation

/// The slot budget for a pack that streams its routed experts, decided once, before it opens.
///
/// The slots are locked in memory, so they are as much a part of what the model holds as its
/// shards, and a memory budget that counted only the shards would promise context that is not
/// there. Everything that sizes one asks here, and gets the same answer the store will open with.
public enum StreamedPlan {
  /// A quarter of the bank. Recorded 125B-A6B routes hit 86% of reads there and 91% at 38%, but
  /// decode measured no faster for the extra slots, so what they would pin is left to the system.
  public static let bankShare = 4

  /// Held back beyond the shards and the slots, for the KV cache, activations and everything
  /// else on the machine.
  public static let reserve = 12 * 1_073_741_824

  public struct Layers: Sendable, Equatable {
    public var count: Int
    public var expertCount: Int
    public var stride: Int
  }

  /// The streamed layers a pack ships, or nil when it holds its experts in its shards.
  public static func layers(in directory: URL) -> Layers? {
    let manager = FileManager.default
    let decoder = JSONDecoder()
    guard
      let data = try? Data(contentsOf: directory.appending(path: ExpertRepack.layoutFile)),
      let shared = try? decoder.decode(ExpertLayout.self, from: data)
    else { return nil }
    let folder = directory.appending(path: ExpertRepack.folder)
    let files = ((try? manager.contentsOfDirectory(atPath: folder.path)) ?? [])
      .filter { $0.hasPrefix("layer_") && $0.hasSuffix(".bin") }
    var stride = shared.stride
    for file in files {
      let own = folder.appending(path: file.replacingOccurrences(of: ".bin", with: ".json"))
      if let data = try? Data(contentsOf: own),
        let layout = try? decoder.decode(ExpertLayout.self, from: data)
      {
        stride = max(stride, layout.stride)
      }
    }
    return files.isEmpty ? nil : Layers(count: files.count, expertCount: shared.expertCount, stride: stride)
  }

  /// Slots per layer: `requested` when it is set, otherwise as many as fit beside the shards,
  /// up to a quarter of the bank, and never fewer than a token routes to.
  public static func slots(
    for directory: URL, requested: Int = BonsaiRuntime.expertSlots,
    ceiling: Int = ResidencyManager.gpuCeiling
  ) -> Int? {
    guard let layers = layers(in: directory) else { return nil }
    let topK = (try? BonsaiConfig.load(directory: directory))?.textConfig.numExpertsPerTok ?? 8
    if requested > 0 { return max(requested, topK) }
    let resident = MemoryBudget.weightBytes(in: directory) ?? 0
    let free = max(ceiling - resident - reserve, 0)
    let fit = free / max(layers.count * layers.stride, 1)
    let wanted = min(fit, max(layers.expertCount / bankShare, topK))
    return max(wanted / 8 * 8, topK)
  }

  /// What a pack holds once it is open: its shards, and its slots when it streams.
  public static func residentBytes(in directory: URL) -> Int? {
    guard let weights = MemoryBudget.weightBytes(in: directory) else { return nil }
    guard let layers = layers(in: directory), let slots = slots(for: directory) else {
      return weights
    }
    return weights + layers.count * slots * layers.stride
  }
}
