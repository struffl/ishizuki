// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// V4.1's conditional memory: n-grams hashed into two huge tables and gated into the stream.

import Foundation
import MLX

/// numpy's `default_rng(seed).integers(0, high, size)` for a small non-negative seed.
///
/// DeepSeek draws each engram layer's hash multipliers this way at load, so reading a
/// checkpoint means reproducing the draw exactly: SeedSequence's entropy mixing, PCG64 seeded
/// from four of its words, and Lemire's bounded rejection on top.
struct NumpyGenerator {
  private var state: UInt128
  private let increment: UInt128

  private static let multiplier = (UInt128(2_549_297_995_355_413_924) << 64)
    | UInt128(4_865_540_595_714_422_341)

  init(seed: UInt64) {
    let words = Self.seedSequence(seed, words: 8)
    func pair(_ i: Int) -> UInt64 { UInt64(words[2 * i]) | (UInt64(words[2 * i + 1]) << 32) }
    let initial = (UInt128(pair(0)) << 64) | UInt128(pair(1))
    let sequence = (UInt128(pair(2)) << 64) | UInt128(pair(3))
    increment = (sequence << 1) | 1
    state = 0
    step()
    state = state &+ initial
    step()
  }

  private mutating func step() { state = state &* Self.multiplier &+ increment }

  mutating func next() -> UInt64 {
    step()
    let rotation = UInt64(truncatingIfNeeded: state >> 122)
    let folded = UInt64(truncatingIfNeeded: state >> 64) ^ UInt64(truncatingIfNeeded: state)
    return (folded >> rotation) | (folded << ((64 &- rotation) & 63))
  }

  /// A draw from `0 ..< high`, for a range wider than 32 bits.
  mutating func below(_ high: UInt64) -> UInt64 {
    let range = high - 1
    let exclusive = range &+ 1
    var product = UInt128(next()) &* UInt128(exclusive)
    var leftover = UInt64(truncatingIfNeeded: product)
    if leftover < exclusive {
      let threshold = (UInt64.max - range) % exclusive
      while leftover < threshold {
        product = UInt128(next()) &* UInt128(exclusive)
        leftover = UInt64(truncatingIfNeeded: product)
      }
    }
    return UInt64(truncatingIfNeeded: product >> 64)
  }

  static func seedSequence(_ seed: UInt64, words count: Int) -> [UInt32] {
    let initA: UInt32 = 0x43b0_d7e5
    let multA: UInt32 = 0x931e_8875
    let initB: UInt32 = 0x8b51_f9dd
    let multB: UInt32 = 0x58f3_8ded
    let mixL: UInt32 = 0xca01_f9dd
    let mixR: UInt32 = 0x4973_f715
    var entropy: [UInt32] = []
    var n = seed
    repeat {
      entropy.append(UInt32(truncatingIfNeeded: n))
      n >>= 32
    } while n > 0

    var hashConstant = initA
    func hashmix(_ value: UInt32) -> UInt32 {
      var v = value ^ hashConstant
      hashConstant = hashConstant &* multA
      v = v &* hashConstant
      return v ^ (v >> 16)
    }
    func mix(_ x: UInt32, _ y: UInt32) -> UInt32 {
      let r = mixL &* x &- mixR &* y
      return r ^ (r >> 16)
    }
    var pool = [UInt32](repeating: 0, count: 4)
    for i in 0..<4 { pool[i] = hashmix(i < entropy.count ? entropy[i] : 0) }
    for source in 0..<4 {
      for target in 0..<4 where source != target {
        pool[target] = mix(pool[target], hashmix(pool[source]))
      }
    }
    for source in 4..<max(entropy.count, 4) {
      for target in 0..<4 { pool[target] = mix(pool[target], hashmix(entropy[source])) }
    }

    var constant = initB
    var out: [UInt32] = []
    for i in 0..<count {
      var value = pool[i % 4] ^ constant
      constant = constant &* multB
      value = value &* constant
      out.append(value ^ (value >> 16))
    }
    return out
  }
}

/// Turns tokens into the rows of each engram layer's table.
///
/// Tokens are first folded through a compressed vocabulary, so spellings that normalise alike
/// share their n-grams. Each position then hashes the 2-, 3- and 4-gram ending at it — a running
/// xor of id times per-shift multiplier — into every head's own prime-sized range of the table.
/// A look-back never crosses the start of the sequence or a dead token (an image), and reads the
/// compressed pad id where it would.
public struct DeepSeekNgramHasher: Sendable {
  public let tokenMap: [Int32]
  public let padId: Int64
  public let multipliers: [[Int64]]
  public let primes: [[Int64]]
  public let offsets: [[Int64]]
  public let maxNgram: Int
  public let heads: Int

  public static let dead: Int32 = -1

  public var columns: Int { (maxNgram - 1) * heads }

  public init(tokenMap: [Int32], config: DeepSeekConfig) throws {
    let compressed = Int(tokenMap.max().map { $0 + 1 } ?? 0)
    guard compressed == config.engramCompressedVocab else {
      throw BonsaiError.shapeMismatch(
        "the token map folds into \(compressed) ids, but the checkpoint was hashed over "
          + "\(config.engramCompressedVocab)")
    }
    self.tokenMap = tokenMap
    self.padId = Int64(tokenMap[config.engramPadTokenId])
    self.maxNgram = config.engramMaxNgram
    self.heads = config.engramHeads
    self.multipliers = Self.multipliers(
      layers: config.engramLayers, ngram: config.engramMaxNgram, vocab: compressed)
    let perLayer = (config.engramMaxNgram - 1) * config.engramHeads
    let all = Self.primes(from: config.engramBucketBase, count: perLayer * config.engramLayers.count)
    self.primes = (0..<config.engramLayers.count).map { Array(all[($0 * perLayer)..<(($0 + 1) * perLayer)]) }
    self.offsets = primes.map { row in
      var running: Int64 = 0
      return row.map { size in
        defer { running += size }
        return running
      }
    }
    for (layer, rows) in zip(primes, config.engramRows) where layer.reduce(0, +) != Int64(rows) {
      throw BonsaiError.shapeMismatch(
        "an engram table has \(rows) rows, but its heads' primes sum to \(layer.reduce(0, +))")
    }
  }

  /// One odd multiplier per (layer, look-back), bounded so a token times it cannot overflow.
  static func multipliers(layers: [Int], ngram: Int, vocab: Int) -> [[Int64]] {
    let bound = max(1, (Int64.max / Int64(vocab)) / 2)
    return layers.map { layer in
      var generator = NumpyGenerator(seed: UInt64(10007 * layer))
      return (0..<ngram).map { _ in Int64(generator.below(UInt64(bound))) * 2 + 1 }
    }
  }

  /// The first `count` primes at or above `base`, in order.
  static func primes(from base: Int, count: Int) -> [Int64] {
    func isPrime(_ n: Int) -> Bool {
      guard n >= 2 else { return false }
      if n % 2 == 0 { return n == 2 }
      var d = 3
      while d * d <= n {
        if n % d == 0 { return false }
        d += 2
      }
      return true
    }
    var found: [Int64] = []
    var n = base
    while found.count < count {
      if isPrime(n) { found.append(Int64(n)) }
      n += 1
    }
    return found
  }

  /// The compressed ids of `tokens`, with dead positions marked.
  public func compress(_ tokens: [Int32], dead: [Bool]? = nil) -> [Int32] {
    tokens.enumerated().map { index, token in
      dead?[index] == true ? Self.dead : tokenMap[Int(token)]
    }
  }

  /// Every engram layer's row indices for `compressed`, which follows `history`: one row of
  /// `columns` per position, order-major, as the table's offsets lay the heads out.
  public func hashes(_ compressed: [Int32], history: [Int32]) -> [[Int64]] {
    let context = history + compressed
    let start = history.count
    var out = [[Int64]](repeating: [], count: multipliers.count)
    for layer in 0..<multipliers.count { out[layer].reserveCapacity(compressed.count * columns) }
    var tokens = [Int64](repeating: 0, count: maxNgram)
    for position in start..<context.count {
      var blocked = false
      for shift in 0..<maxNgram {
        let source = position - shift
        blocked = blocked || source < 0 || context[source] == Self.dead
        tokens[shift] = blocked ? padId : Int64(context[source])
      }
      for (layer, factors) in multipliers.enumerated() {
        var rolling = tokens[0] &* factors[0]
        for order in 1..<maxNgram {
          rolling ^= tokens[order] &* factors[order]
          for head in 0..<heads {
            let column = (order - 1) * heads + head
            out[layer].append(rolling % primes[layer][column] + offsets[layer][column])
          }
        }
      }
    }
    return out
  }

  /// What the next chunk's look-back needs: the last few compressed ids.
  public func history(after context: [Int32]) -> [Int32] {
    Array(context.suffix(maxNgram - 1))
  }
}

/// One engram layer's table: fp8 rows, a power-of-two scale per 32 of them.
public protocol DeepSeekEngramTable: Sendable {
  /// The rows at `indices`, widened, `[indices.count, headDim]`.
  func rows(_ indices: [Int64]) throws -> MLXArray
}

/// A table held in memory, as a small checkpoint's is.
public struct ResidentEngramTable: DeepSeekEngramTable, @unchecked Sendable {
  let codes: MLXArray
  let scales: MLXArray

  public init(codes: MLXArray, scales: MLXArray) {
    self.codes = codes
    self.scales = scales
  }

  public func rows(_ indices: [Int64]) throws -> MLXArray {
    let at = MLXArray(indices.map { Int32($0) })
    let picked = take(codes, at, axis: 0)
    return dequantized(
      picked.view(dtype: .uint32), scales: take(scales, at, axis: 0), biases: nil,
      groupSize: 32, bits: 8, mode: .mxfp8, dtype: .float32)
  }
}

/// The table read from the checkpoint's own shard, a row at a time.
///
/// The release keeps each table as one 100 GB tensor and its scales as another, so a row is two
/// reads at computed offsets and nothing has to be repacked first.
public final class CheckpointEngramTable: DeepSeekEngramTable, @unchecked Sendable {
  let codes: DeepSeekCheckpoint.Entry
  let scales: DeepSeekCheckpoint.Entry
  private let codeFile: Int32
  private let scaleFile: Int32

  public init(checkpoint: DeepSeekCheckpoint, prefix: String) throws {
    self.codes = try checkpoint.entry(prefix + ".weight")
    self.scales = try checkpoint.entry(prefix + ".scale")
    self.codeFile = open(codes.file.path, O_RDONLY)
    self.scaleFile = open(scales.file.path, O_RDONLY)
    guard codeFile >= 0, scaleFile >= 0 else {
      throw BonsaiError.missingWeight("cannot open the engram table \(prefix)")
    }
  }

  deinit {
    close(codeFile)
    close(scaleFile)
  }

  public func rows(_ indices: [Int64]) throws -> MLXArray {
    let codeWidth = codes.rowBytes
    let scaleWidth = scales.rowBytes
    let count = indices.count
    let codeBuffer = try ResidentBuffer(byteCount: count * codeWidth)
    let scaleBuffer = try ResidentBuffer(byteCount: count * scaleWidth)
    let failure = ReadFailure()
    DispatchQueue.concurrentPerform(iterations: count) { slot in
      let row = Int(indices[slot])
      do {
        try codeBuffer.read(
          from: codeFile, offset: codes.offset + row * codeWidth,
          into: (slot * codeWidth)..<((slot + 1) * codeWidth))
        try scaleBuffer.read(
          from: scaleFile, offset: scales.offset + row * scaleWidth,
          into: (slot * scaleWidth)..<((slot + 1) * scaleWidth))
      } catch {
        failure.record(error)
      }
    }
    if let error = failure.error { throw error }
    let words = codeBuffer.array(shape: [count, codeWidth / 4], dtype: .uint32)
    let rowScales = scaleBuffer.array(shape: [count, scaleWidth], dtype: .uint8)
    let out = dequantized(
      words, scales: rowScales, biases: nil, groupSize: 32, bits: 8, mode: .mxfp8,
      dtype: .float32)
    eval(out)
    return out
  }
}

private final class ReadFailure: @unchecked Sendable {
  private let lock = NSLock()
  private(set) var error: Error?

  func record(_ error: Error) {
    lock.lock()
    defer { lock.unlock() }
    if self.error == nil { self.error = error }
  }
}

/// Writes an engram layer's lookup into the residual copies, each copy gated by how well the
/// fetched key matches it.
struct DeepSeekEngramBlock: @unchecked Sendable {
  let index: Int
  let projection: any Projection
  let weight: MLXArray
  let copies: Int
  let dim: Int
  let eps: Float

  init(index: Int, prefix: String, config: DeepSeekConfig, weights: DeepSeekWeights) throws {
    self.index = index
    self.projection = try weights.linear(prefix + ".wkv")
    self.weight = try weights.float32(prefix + ".q_weight") * weights.float32(prefix + ".k_weight")
    self.copies = config.hcMult
    self.dim = config.dim
    self.eps = config.normEps
  }

  /// `streams` `[b, s, copies, dim]`, `rows` `[b, s, columns * headDim]`; `live` is false where a
  /// position takes no part, and those pass through untouched.
  func callAsFunction(_ streams: MLXArray, rows: MLXArray, live: MLXArray? = nil) -> MLXArray {
    let b = streams.dim(0)
    let s = streams.dim(1)
    let kv = projection(rows).asType(.float32)
    let key = kv[.ellipsis, ..<(copies * dim)].reshaped([b, s, copies, dim])
    let value = kv[.ellipsis, (copies * dim)...]
    let h = streams.asType(.float32)
    let rstd =
      rsqrt(h.square().mean(axis: -1) + eps) * rsqrt(key.square().mean(axis: -1) + eps)
    let dot = (h * weight * key).sum(axis: -1) * rstd * Float(pow(Double(dim), -0.5))
    let root = sqrt(maximum(abs(dot), MLXArray(Float(1e-6))))
    var gate = sigmoid(MLX.where(dot .< 0, -root, root))
    if let live { gate = gate * live.asType(.float32).expandedDimensions(axis: -1) }
    return h + gate.expandedDimensions(axis: -1) * value.expandedDimensions(axis: -2)
  }
}
