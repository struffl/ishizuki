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
/// The cache is least-recently-used. TurboFieldfare measured frequency as the better policy
/// for the model it streamed, but recorded qwen4_exp routes say otherwise: with uses that
/// never decay, an expert hot early in a context holds its slot long after it has gone cold,
/// and LRU hits 8-13 points more of a Whittle 35B-A3B's reads at a third to half of the bank.
/// `Scripts/simulate_expert_cache.py` replays a `RouteProbe` recording against both.
public final class ExpertStore: @unchecked Sendable {
  public let layout: ExpertLayout
  public let slotCount: Int

  private let descriptor: Int32
  /// Where each expert's parts sit when they are not one blob of a repacked file: a release
  /// checkpoint keeps every projection of every expert as a tensor of its own, in shards.
  private let located: Located?
  private let buffer: ResidentBuffer
  /// Where each part's slots begin in the buffer. The file is expert major — one blob per
  /// expert — but the slots are part major, so all the slots of one projection are contiguous
  /// and can be handed to a gathered matmul as a single `[slots, ...]` tensor.
  private let base: [String: Int]
  /// Each part's slots as one array, made once. Metal wraps memory without copying only when it
  /// starts and ends on a page, and MLX copies whatever Metal refuses: a view made per call
  /// over unaligned slots copied every slot of every layer on every token, about 20 GB a token
  /// for the 125B-A6B at 128 slots.
  private var views: [String: MLXArray] = [:]
  private let lock = NSLock()

  /// Which expert each slot holds, and when it was last asked for.
  private var occupant: [Int]
  private var lastTouched: [Int]
  private var clock = 0

  private var hitCount = 0
  private var missCount = 0

  /// Reads and slot refills since the pack was opened, taken together so a reader cannot see
  /// one of them from a half-finished token.
  public struct Traffic: Sendable, Equatable {
    public var hits: Int
    public var misses: Int

    public var reads: Int { hits + misses }
    public var hitRate: Double { reads > 0 ? Double(hits) / Double(reads) : 0 }
  }

  public var traffic: Traffic {
    lock.lock()
    defer { lock.unlock() }
    return Traffic(hits: hitCount, misses: missCount)
  }

  public var hits: Int { traffic.hits }
  public var misses: Int { traffic.misses }

  /// What the slots of this layer occupy, which is the memory the budget buys.
  public var heldBytes: Int { slotCount * layout.stride }

  /// Whether the slots are locked in memory, which `BonsaiRuntime.lockExpertSlots` asks for.
  public var isLocked: Bool { buffer.isLocked }

  /// Every streamed layer's traffic at once: what a readout needs to say whether the slot
  /// budget is buying anything.
  public struct Summary: Sendable, Equatable {
    public var layers: Int
    public var expertCount: Int
    public var slots: Int
    public var hits: Int
    public var misses: Int
    public var heldBytes: Int

    public var reads: Int { hits + misses }
    public var hitRate: Double { reads > 0 ? Double(hits) / Double(reads) : 0 }

    public init?(layers stores: some Collection<ExpertStore>) {
      guard let first = stores.first else { return nil }
      self.layers = stores.count
      self.expertCount = first.layout.expertCount
      self.slots = first.slotCount
      self.hits = 0
      self.misses = 0
      self.heldBytes = 0
      for store in stores {
        let traffic = store.traffic
        hits += traffic.hits
        misses += traffic.misses
        heldBytes += store.heldBytes
      }
    }
  }

  /// Parts read from wherever they already are: `places[expert][part]` is a file in `files` and
  /// the byte offset of that part's tensor in it.
  public struct Placement: Sendable {
    public var files: [URL]
    public var places: [[String: (file: Int, offset: Int)]]

    public init(files: [URL], places: [[String: (file: Int, offset: Int)]]) {
      self.files = files
      self.places = places
    }
  }

  private struct Located {
    let descriptors: [Int32]
    let places: [[String: (file: Int, offset: Int)]]
  }

  public convenience init(url: URL, layout: ExpertLayout, slots: Int) throws {
    let opened = Self.openUncached(url)
    guard opened >= 0 else {
      throw BonsaiError.missingWeight("cannot open \(url.lastPathComponent)")
    }
    try self.init(descriptor: opened, located: nil, layout: layout, slots: slots)
  }

  /// A shard opened for reads that bypass the page cache. The slots are the cache: a copy of
  /// every expert read left in the file cache as well doubles what streaming holds, and on a
  /// machine already near its ceiling that is what gets the rest of it swapped out.
  static func openUncached(_ url: URL) -> Int32 {
    let opened = open(url.path, O_RDONLY)
    if opened >= 0 { _ = fcntl(opened, F_NOCACHE, 1) }
    return opened
  }

  public convenience init(placement: Placement, layout: ExpertLayout, slots: Int) throws {
    guard placement.places.count == layout.expertCount else {
      throw BonsaiError.shapeMismatch(
        "\(placement.places.count) experts placed for a layout of \(layout.expertCount)")
    }
    var descriptors: [Int32] = []
    for url in placement.files {
      let opened = Self.openUncached(url)
      guard opened >= 0 else {
        for previous in descriptors { close(previous) }
        throw BonsaiError.missingWeight("cannot open \(url.lastPathComponent)")
      }
      descriptors.append(opened)
    }
    try self.init(
      descriptor: -1, located: Located(descriptors: descriptors, places: placement.places),
      layout: layout, slots: slots)
  }

  private init(descriptor: Int32, located: Located?, layout: ExpertLayout, slots: Int) throws {
    self.descriptor = descriptor
    self.located = located
    self.layout = layout
    self.slotCount = min(slots, layout.expertCount)
    var base: [String: Int] = [:]
    var total = 0
    for name in layout.parts.keys.sorted() {
      base[name] = total
      total += Self.pageRounded(self.slotCount * layout.parts[name]!.byteCount)
    }
    self.base = base
    self.buffer = try ResidentBuffer(byteCount: total, locked: BonsaiRuntime.lockExpertSlots)
    self.occupant = Array(repeating: -1, count: self.slotCount)
    self.lastTouched = Array(repeating: 0, count: self.slotCount)
  }

  deinit {
    if descriptor >= 0 { close(descriptor) }
    for other in located?.descriptors ?? [] { close(other) }
  }

  private func source(expert: Int, part name: String, _ part: ExpertLayout.Part) -> (Int32, Int) {
    guard let located, let place = located.places[expert][name] else {
      return (descriptor, expert * layout.stride + part.offset)
    }
    return (located.descriptors[place.file], place.offset)
  }

  /// Brings `experts` into slots and says where each landed, in the order asked for. A repeated
  /// expert is read once.
  @discardableResult
  public func residency(of experts: [Int]) throws -> [Int] {
    lock.lock()
    defer { lock.unlock() }

    var placed: [Int: Int] = [:]
    var missed: [(expert: Int, slot: Int)] = []
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
        hitCount += 1
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
          .min(by: { lastTouched[$0] < lastTouched[$1] })
      else {
        throw BonsaiError.missingComponent(
          "\(experts.count) experts asked for at once, but only \(slotCount) slots")
      }

      missCount += 1
      occupant[slot] = expert
      lastTouched[slot] = clock
      placed[expert] = slot
      result.append(slot)
      missed.append((expert: expert, slot: slot))
    }

    // One read per projection per miss: each is contiguous in the expert's blob and in its own
    // run of slots, and every read lands in a disjoint range, so they all go out at once. An
    // SSD answers a token's scattered experts far faster in parallel than one after another.
    let parts = Array(layout.parts)
    let reads = missed.flatMap { miss in parts.map { (miss, $0) } }
    let failure = FirstFailure()
    DispatchQueue.concurrentPerform(iterations: reads.count) { index in
      let (miss, (name, part)) = reads[index]
      let target = base[name]! + miss.slot * part.byteCount
      do {
        let (file, offset) = source(expert: miss.expert, part: name, part)
        try buffer.readUncached(
          from: file, offset: offset, into: target..<(target + part.byteCount))
      } catch {
        failure.record(error)
      }
    }
    if let failure = failure.error {
      for miss in missed { occupant[miss.slot] = -1 }
      throw failure
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
    lock.lock()
    defer { lock.unlock() }
    if let view = views[name] { return view }
    guard let part = layout.parts[name], let start = base[name] else {
      throw BonsaiError.missingWeight("\(name) is not in this expert layout")
    }
    let type = try part.type
    let used = slotCount * part.byteCount
    let flat = buffer.array(
      byteOffset: start, shape: [Self.pageRounded(used) / type.size], dtype: type)
    let view = flat[0..<(used / type.size)].reshaped([slotCount] + part.shape)
    eval(view)
    views[name] = view
    return view
  }

  static func pageRounded(_ bytes: Int) -> Int {
    (bytes + ResidentBuffer.pageSize - 1) / ResidentBuffer.pageSize * ResidentBuffer.pageSize
  }
}

/// The first error any of a batch of concurrent reads hit.
final class FirstFailure: @unchecked Sendable {
  private let lock = NSLock()
  private var first: Error?

  func record(_ error: Error) {
    lock.lock()
    defer { lock.unlock() }
    if first == nil { first = error }
  }

  var error: Error? {
    lock.lock()
    defer { lock.unlock() }
    return first
  }
}
