// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Routed experts kept on disk and read a few at a time.

import Foundation
import MLX

/// Where one expert's three projections sit inside its blob, and how big each is.
///
/// Every expert in a layer has the same shape, so one description covers the file: a blob is a
/// fixed stride and each sub-tensor a fixed offset inside it. That is what lets a miss be one
/// read at a known place rather than a lookup.
public struct ExpertLayout: Codable, Sendable, Equatable {
  public struct Part: Codable, Sendable, Equatable {
    public var offset: Int
    public var byteCount: Int
    public var shape: [Int]
    public var dtype: String

    public init(offset: Int, byteCount: Int, shape: [Int], dtype: String) {
      self.offset = offset
      self.byteCount = byteCount
      self.shape = shape
      self.dtype = dtype
    }
  }

  public var expertCount: Int
  public var stride: Int
  /// Keyed by the name the runtime asks for: `gate_proj.weight`, `gate_proj.scales`, and so on.
  public var parts: [String: Part]

  public init(expertCount: Int, stride: Int, parts: [String: Part]) {
    self.expertCount = expertCount
    self.stride = stride
    self.parts = parts
  }

  /// Packs the given tensors into one blob per expert, each blob page aligned so a read never
  /// straddles a page it did not need.
  public static func plan(
    expertCount: Int, tensors: [(name: String, shape: [Int], dtype: DType)]
  ) -> ExpertLayout {
    var parts: [String: Part] = [:]
    var offset = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
      let bytes = tensor.shape.reduce(1, *) * tensor.dtype.size
      parts[tensor.name] = Part(
        offset: offset, byteCount: bytes, shape: tensor.shape, dtype: "\(tensor.dtype)")
      offset += bytes
    }
    let page = ResidentBuffer.pageSize
    return ExpertLayout(
      expertCount: expertCount, stride: (offset + page - 1) / page * page, parts: parts)
  }
}

/// One layer's experts, read from disk into a bounded set of slots.
///
/// The cache is least-frequently-used with recency breaking ties, which is what TurboFieldfare
/// measured as the better policy for this access pattern: expert choices repeat across tokens
/// but the repeats are not recent, so recency alone evicts the wrong ones.
public final class ExpertStore: @unchecked Sendable {
  public let layout: ExpertLayout
  public let slotCount: Int

  private let descriptor: Int32
  private let buffer: ResidentBuffer
  private let lock = NSLock()

  /// Which expert each slot holds, and how often it has been asked for.
  private var occupant: [Int]
  private var uses: [Int]
  private var lastTouched: [Int]
  private var clock = 0

  public private(set) var hits = 0
  public private(set) var misses = 0

  public init(url: URL, layout: ExpertLayout, slots: Int) throws {
    let opened = open(url.path, O_RDONLY)
    guard opened >= 0 else {
      throw BonsaiError.missingWeight("cannot open \(url.lastPathComponent)")
    }
    self.descriptor = opened
    self.layout = layout
    self.slotCount = min(slots, layout.expertCount)
    self.buffer = try ResidentBuffer(byteCount: self.slotCount * layout.stride)
    self.occupant = Array(repeating: -1, count: self.slotCount)
    self.uses = Array(repeating: 0, count: self.slotCount)
    self.lastTouched = Array(repeating: 0, count: self.slotCount)
  }

  deinit { close(descriptor) }

  /// Brings `experts` into slots and says where each landed, in the order asked for. A repeated
  /// expert is read once.
  @discardableResult
  public func residency(of experts: [Int]) throws -> [Int] {
    lock.lock()
    defer { lock.unlock() }

    var placed: [Int: Int] = [:]
    var result: [Int] = []
    result.reserveCapacity(experts.count)

    for expert in experts {
      if let slot = placed[expert] {
        result.append(slot)
        continue
      }
      guard expert >= 0, expert < layout.expertCount else {
        throw BonsaiError.shapeMismatch("expert \(expert) is not in this layer")
      }
      clock += 1

      if let slot = occupant.firstIndex(of: expert) {
        hits += 1
        uses[slot] += 1
        lastTouched[slot] = clock
        placed[expert] = slot
        result.append(slot)
        continue
      }

      // Nothing this token has already claimed may be evicted, or a read would land under a
      // tensor the forward pass is still going to use.
      let claimed = Set(placed.values)
      guard
        let slot = (0..<slotCount)
          .filter({ !claimed.contains($0) })
          .min(by: {
            (uses[$0], lastTouched[$0]) < (uses[$1], lastTouched[$1])
          })
      else {
        throw BonsaiError.missingComponent(
          "\(experts.count) experts asked for at once, but only \(slotCount) slots")
      }

      misses += 1
      try buffer.read(
        from: descriptor, offset: expert * layout.stride,
        into: (slot * layout.stride)..<(slot * layout.stride + layout.stride))
      occupant[slot] = expert
      uses[slot] = 1
      lastTouched[slot] = clock
      placed[expert] = slot
      result.append(slot)
    }
    return result
  }

  /// One resident part, as an array over the slot it occupies.
  public func array(_ name: String, slot: Int, dtype: DType) throws -> MLXArray {
    guard let part = layout.parts[name] else {
      throw BonsaiError.missingWeight("\(name) is not in this expert layout")
    }
    return buffer.array(
      byteOffset: slot * layout.stride + part.offset, shape: part.shape, dtype: dtype)
  }
}
