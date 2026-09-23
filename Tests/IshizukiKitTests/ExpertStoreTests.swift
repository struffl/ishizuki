// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Experts read from disk into a bounded set of slots.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// A layer holds more experts than it can keep in memory, which is the whole point: the store
/// has to put the right ones in slots, evict the right ones, and never read over a tensor the
/// forward pass still needs.
@Suite("Expert store")
struct ExpertStoreTests {
  private let experts = 8
  private let width = 32

  /// One file of `experts` blobs, each carrying a single float32 tensor whose every value is
  /// the expert's own index, so a misplaced read is obvious rather than subtle.
  private func write(to url: URL) throws -> ExpertLayout {
    let layout = ExpertLayout.plan(
      expertCount: experts,
      tensors: [(name: "gate_proj.weight", shape: [width], dtype: .float32)])

    var data = Data(count: experts * layout.stride)
    for expert in 0..<experts {
      let part = layout.parts["gate_proj.weight"]!
      let base = expert * layout.stride + part.offset
      for i in 0..<width {
        let value = Float(expert)
        withUnsafeBytes(of: value) { bytes in
          data.replaceSubrange((base + i * 4)..<(base + i * 4 + 4), with: bytes)
        }
      }
    }
    try data.write(to: url)
    return layout
  }

  private func temporary() -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "experts-\(UUID().uuidString).bin")
  }

  @Test("puts the experts a token asked for into slots, and reads each one once")
  func residency() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }
    let layout = try write(to: url)

    let store = try ExpertStore(url: url, layout: layout, slots: 4)
    let slots = try store.residency(of: [5, 2, 5, 7])
    #expect(slots.count == 4)
    #expect(slots[0] == slots[2], "the same expert twice should land in one slot")
    #expect(store.misses == 3)
    // Asking twice in one breath is deduplication, not a cache hit: the counters measure what
    // survives between tokens, which is what a slot budget has to be judged on.
    #expect(store.hits == 0)

    // Each slot holds the expert it was asked for, not its neighbour.
    let resident = try store.array("gate_proj.weight")
    eval(resident)
    #expect(resident.shape == [4, width])
    for (expert, slot) in zip([5, 2, 5, 7], slots) {
      #expect(resident[slot, 0].item(Float.self) == Float(expert))
      #expect(resident[slot, width - 1].item(Float.self) == Float(expert))
    }
  }

  @Test("a second token reuses what the first one left behind")
  func reuse() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }
    let layout = try write(to: url)

    let store = try ExpertStore(url: url, layout: layout, slots: 4)
    _ = try store.residency(of: [0, 1, 2])
    #expect(store.misses == 3)

    _ = try store.residency(of: [1, 2])
    #expect(store.misses == 3, "nothing new was needed")
    #expect(store.hits == 2)
  }

  @Test("evicts the least recently used, and keeps what this token already claimed")
  func eviction() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }
    let layout = try write(to: url)

    let store = try ExpertStore(url: url, layout: layout, slots: 2)
    // 1 was last asked for before 0, so 1 is the one to go.
    _ = try store.residency(of: [0, 1])
    _ = try store.residency(of: [0])
    let slots = try store.residency(of: [0, 3])
    let resident = try store.array("gate_proj.weight")
    eval(resident)
    #expect(resident[slots[0], 0].item(Float.self) == 0)
    #expect(resident[slots[1], 0].item(Float.self) == 3)

    // Asking for more at once than there are slots cannot be served.
    #expect(throws: BonsaiError.self) { _ = try store.residency(of: [4, 5, 6]) }
  }

  @Test("an expert asked for often but not lately is the one evicted")
  func recencyOverFrequency() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }
    let layout = try write(to: url)

    let store = try ExpertStore(url: url, layout: layout, slots: 2)
    for _ in 0..<3 { _ = try store.residency(of: [0]) }
    _ = try store.residency(of: [1])
    _ = try store.residency(of: [2])
    #expect(store.misses == 3)
    _ = try store.residency(of: [1])
    #expect(store.misses == 3, "1 was more recent than 0, however often 0 was asked for")
    _ = try store.residency(of: [0])
    #expect(store.misses == 4)
  }

  @Test("slots asked to be locked are locked, and read the same")
  func lockedSlots() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }
    let layout = try write(to: url)

    let store = try ExpertStore(url: url, layout: layout, slots: 2)
    #expect(store.isLocked)
    let slots = try store.residency(of: [3, 1])
    let resident = try store.array("gate_proj.weight")
    eval(resident)
    #expect(resident[slots[0], 0].item(Float.self) == 3)
    #expect(resident[slots[1], 0].item(Float.self) == 1)
  }
}
