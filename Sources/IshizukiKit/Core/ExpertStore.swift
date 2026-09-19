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

    /// The type the bytes are, recovered from the name the plan wrote down. A blob is raw
    /// bytes, so reading one back at the wrong width is silent and wrong; the layout is the
    /// only record of which width is right.
    public var type: DType {
      get throws {
        guard let type = DType.allCases.first(where: { "\($0)" == dtype }) else {
          throw BonsaiError.unsupportedModel("\(dtype) is not a type this runtime reads")
        }
        return type
      }
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
  /// Where each part's slots begin in the buffer. The file is expert major — one blob per
  /// expert — but the slots are part major, so all the slots of one projection are contiguous
  /// and can be handed to a gathered matmul as a single `[slots, ...]` tensor.
  private let base: [String: Int]
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
    var base: [String: Int] = [:]
    var total = 0
    for name in layout.parts.keys.sorted() {
      base[name] = total
      total += self.slotCount * layout.parts[name]!.byteCount
    }
    self.base = base
    self.buffer = try ResidentBuffer(byteCount: total)
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
      // One read per projection: each is contiguous in the expert's blob and contiguous in its
      // own run of slots, so nothing is copied or shuffled after it lands.
      for (name, part) in layout.parts {
        let target = base[name]! + slot * part.byteCount
        try buffer.read(
          from: descriptor, offset: expert * layout.stride + part.offset,
          into: target..<(target + part.byteCount))
      }
      occupant[slot] = expert
      uses[slot] = 1
      lastTouched[slot] = clock
      placed[expert] = slot
      result.append(slot)
    }
    return result
  }

  /// One projection across every slot, shaped for a gathered matmul. The slot numbers that
  /// `residency(of:)` returned index into it.
  ///
  /// The array is a view onto the slot buffer, not a copy: it holds what the slots hold now,
  /// and the next `residency(of:)` rewrites it. Anything computed from it has to be evaluated
  /// before more experts are asked for.
  public func array(_ name: String) throws -> MLXArray {
    guard let part = layout.parts[name], let start = base[name] else {
      throw BonsaiError.missingWeight("\(name) is not in this expert layout")
    }
    return buffer.array(
      byteOffset: start, shape: [slotCount] + part.shape, dtype: try part.type)
  }
}
