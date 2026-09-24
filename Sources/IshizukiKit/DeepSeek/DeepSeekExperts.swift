// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// V4.1's routed experts read out of the release shards a few at a time.

import Foundation
import MLX

/// A layer's experts streamed into slots straight from the checkpoint.
///
/// Every expert in the release is six tensors in its layer's shard — three fp4 matrices and
/// their scales — which is already exactly what mxfp4 multiplies. So nothing is repacked: a
/// miss is six reads at the offsets the shard's header gives, into slots laid out for a gathered
/// matmul. A decode step's handful of rows runs in groups that fit the slots; a prefill chunk,
/// which wants nearly the whole bank, runs expert by expert so each is read once per chunk.
public struct StreamedExpertBank: DeepSeekExpertBank, @unchecked Sendable {
  public let store: ExpertStore

  public var count: Int { store.layout.expertCount }

  static let projections = ["w1", "w2", "w3"]

  public init(store: ExpertStore) {
    self.store = store
  }

  public init(checkpoint: DeepSeekCheckpoint, prefix: String, count: Int, slots: Int) throws {
    var tensors: [(name: String, shape: [Int], dtype: DType)] = []
    var files: [URL] = []
    var fileIndex: [URL: Int] = [:]
    var places = [[String: (file: Int, offset: Int)]](repeating: [:], count: count)
    for expert in 0..<count {
      for projection in Self.projections {
        for (suffix, part) in [(".weight", ".weight"), (".scale", ".scales")] {
          let entry = try checkpoint.entry("\(prefix).\(expert).\(projection)\(suffix)")
          if expert == 0 {
            let shape =
              suffix == ".weight"
              ? [entry.shape[0], entry.byteCount / entry.shape[0] / 4] : entry.shape
            tensors.append(
              (name: projection + part, shape: shape, dtype: suffix == ".weight" ? .uint32 : .uint8))
          }
          let file = fileIndex[entry.file] ?? {
            files.append(entry.file)
            fileIndex[entry.file] = files.count - 1
            return files.count - 1
          }()
          places[expert][projection + part] = (file: file, offset: entry.offset)
        }
      }
    }
    let layout = ExpertLayout.plan(expertCount: count, tensors: tensors)
    self.store = try ExpertStore(
      placement: ExpertStore.Placement(files: files, places: places), layout: layout, slots: slots)
  }

  private func project(
    _ name: String, _ input: MLXArray, slots: MLXArray, sorted: Bool = false
  ) throws -> MLXArray {
    gatherQuantizedMM(
      input, try store.array(name + ".weight"), scales: try store.array(name + ".scales"),
      biases: nil, rhsIndices: slots, transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
      sortedIndices: sorted)
  }

  public func run(_ x: MLXArray, chosen: MLXArray, weights: MLXArray, limit: Float) throws
    -> MLXArray
  {
    eval(chosen, weights)
    let rows = x.dim(0)
    let topK = chosen.dim(-1)
    let asked = chosen.asArray(Int32.self).map(Int.init)
    if rows > Self.tokenMajorRows {
      return try byExpert(x, asked: asked, weights: weights, topK: topK, limit: limit)
    }

    var pieces: [MLXArray] = []
    var start = 0
    while start < rows {
      var wanted: Set<Int> = []
      var end = start
      while end < rows {
        let next = wanted.union(asked[(end * topK)..<((end + 1) * topK)])
        if next.count > store.slotCount, end > start { break }
        wanted = next
        end += 1
      }
      let slots = try store.residency(of: Array(asked[(start * topK)..<(end * topK)]))
      let placed = MLXArray(slots.map { Int32($0) }, [end - start, topK])
      let batched = x[start..<end].expandedDimensions(axes: [-2, -3])
      let hidden = clampedSwiGLU(
        gate: try project("w1", batched, slots: placed).asType(.float32),
        up: try project("w3", batched, slots: placed).asType(.float32), limit: limit)
      let scaled = hidden * weights[start..<end].expandedDimensions(axes: [-1, -2])
      let piece = try project("w2", scaled.asType(x.dtype), slots: placed).squeezed(axis: -2)
      eval(piece)
      pieces.append(piece)
      start = end
    }
    return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
  }

  /// Past this many rows a chunk runs expert by expert rather than token by token.
  static let tokenMajorRows = 8

  private func byExpert(
    _ x: MLXArray, asked: [Int], weights: MLXArray, topK: Int, limit: Float
  ) throws -> MLXArray {
    let flatWeights = weights.reshaped([-1])
    let byChoice = (0..<asked.count).sorted { (asked[$0], $0) < (asked[$1], $1) }
    var done: [Int] = []
    done.reserveCapacity(asked.count)
    var outputs: [MLXArray] = []
    var start = 0
    while start < byChoice.count {
      var experts: [Int] = []
      var end = start
      while end < byChoice.count {
        let expert = asked[byChoice[end]]
        if experts.last != expert {
          if experts.count == store.slotCount { break }
          experts.append(expert)
        }
        end += 1
      }
      let slotOf = Dictionary(uniqueKeysWithValues: zip(experts, try store.residency(of: experts)))
      let pairs = byChoice[start..<end].sorted {
        (slotOf[asked[$0]]!, $0) < (slotOf[asked[$1]]!, $1)
      }
      let tokens = MLXArray(pairs.map { Int32($0 / topK) })
      let placed = MLXArray(pairs.map { Int32(slotOf[asked[$0]]!) })
      let share = flatWeights[MLXArray(pairs.map { Int32($0) })]
      let input = x[tokens].expandedDimensions(axis: -2)
      let hidden = clampedSwiGLU(
        gate: try project("w1", input, slots: placed, sorted: true).asType(.float32),
        up: try project("w3", input, slots: placed, sorted: true).asType(.float32), limit: limit)
      let scaled = hidden * share.reshaped([-1, 1, 1])
      let out = try project("w2", scaled.asType(x.dtype), slots: placed, sorted: true)
        .squeezed(axis: -2)
      eval(out)
      outputs.append(out)
      done += pairs
      start = end
    }
    var position = [Int32](repeating: 0, count: done.count)
    for (index, pair) in done.enumerated() { position[pair] = Int32(index) }
    return concatenated(outputs, axis: 0)[MLXArray(position)]
      .reshaped([x.dim(0), topK, x.dim(-1)])
  }
}
