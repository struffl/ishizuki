// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// Experts read from slots answer exactly as experts held in memory.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// Streaming is only worth anything if it changes nothing. The same weights, reached two ways —
/// stacked in memory and read into slots — have to give the same numbers, because the whole
/// claim of the repacker is that it moves bytes without touching values.
@Suite("Streamed experts")
struct StreamedExpertsTests {
  private let experts = 6
  private let hidden = 64
  private let intermediate = 96
  private let groupSize = 32
  private let bits = 4

  private struct Bank {
    var resident: ResidentExperts
    var layout: ExpertLayout
    var parts: [String: MLXArray]
  }

  /// One layer's worth of quantized experts, plus the flat tensors a blob is cut from.
  private func bank() -> Bank {
    var parts: [String: MLXArray] = [:]
    var stacked: [String: StackedExperts] = [:]

    for (name, out, into) in [
      ("gate_proj", intermediate, hidden), ("up_proj", intermediate, hidden),
      ("down_proj", hidden, intermediate),
    ] {
      let dense = MLXRandom.normal([experts, out, into], dtype: .float32) * 0.1
      let (weight, scales, biases) = quantized(
        dense, groupSize: groupSize, bits: bits, mode: .affine)
      let half = scales.asType(.float16)
      let halfBiases = biases!.asType(.float16)
      parts["\(name).weight"] = weight
      parts["\(name).scales"] = half
      parts["\(name).biases"] = halfBiases
      stacked[name] = StackedExperts(
        weight: weight, scales: half, biases: halfBiases, groupSize: groupSize, bits: bits)
    }

    let layout = ExpertLayout.plan(
      expertCount: experts,
      tensors: parts.map {
        (name: $0.key, shape: Array($0.value.shape.dropFirst()), dtype: $0.value.dtype)
      })
    return Bank(
      resident: ResidentExperts(
        gate: stacked["gate_proj"]!, up: stacked["up_proj"]!, down: stacked["down_proj"]!),
      layout: layout, parts: parts)
  }

  /// Cuts the stacked tensors into one blob per expert, the way the repacker does.
  private func write(_ bank: Bank, to url: URL) throws {
    var blob = Data(count: experts * bank.layout.stride)
    for (name, part) in bank.layout.parts {
      let bytes = bank.parts[name]!.asData().data
      for expert in 0..<experts {
        let from = bytes.startIndex + expert * part.byteCount
        let to = expert * bank.layout.stride + part.offset
        blob.replaceSubrange(
          to..<(to + part.byteCount), with: bytes[from..<(from + part.byteCount)])
      }
    }
    try blob.write(to: url)
  }

  private func temporary() -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "streamed-\(UUID().uuidString).bin")
  }

  private func compare(slots: Int, routing: [[Int32]]) throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }

    let bank = bank()
    try write(bank, to: url)
    let streamed = StreamedExperts(
      store: try ExpertStore(url: url, layout: bank.layout, slots: slots))

    for chosen in routing {
      let x = MLXRandom.normal([chosen.count / 2, hidden], dtype: .float32)
      let indices = MLXArray(chosen, [chosen.count / 2, 2])

      let want = bank.resident.swiglu(x, chosen: indices, groupSize: groupSize, bits: bits)
      let got = try streamed.swiglu(x, chosen: indices, groupSize: groupSize, bits: bits)
      eval(want, got)

      #expect(got.shape == want.shape)
      let error = (got - want).abs().max().item(Float.self)
      #expect(error == 0, "slots must not change a single value, but moved one by \(error)")
    }
  }

  @Test("a layer read into slots answers exactly as one held whole")
  func matchesResident() throws {
    try compare(slots: experts, routing: [[0, 1, 2, 3], [4, 5, 0, 2]])
  }

  /// The case the whole design exists for: fewer slots than experts, so one token evicts what
  /// the last one left and a projection is rebuilt from a different corner of the buffer.
  @Test("and still does when the slots are fewer than the experts")
  func matchesUnderEviction() throws {
    try compare(
      slots: 3, routing: [[0, 1], [2, 3], [4, 5], [0, 1], [3, 0], [5, 5], [1, 4]])
  }

  /// A prefill chunk routes to every expert a layer has, which no slot budget worth having can
  /// hold at once. It has to come out right anyway, in groups, and match the resident answer
  /// token for token.
  @Test("runs a batch wider than its slots by splitting it")
  func matchesAcrossAWideBatch() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }

    let bank = bank()
    try write(bank, to: url)
    let streamed = StreamedExperts(
      store: try ExpertStore(url: url, layout: bank.layout, slots: 2))

    let tokens = 32
    let x = MLXRandom.normal([tokens, hidden], dtype: .float32)
    var routing: [Int32] = []
    for index in 0..<(tokens * 2) {
      routing.append(Int32((index * 7 + index / 3) % experts))
    }
    let chosen = MLXArray(routing, [tokens, 2])

    let want = bank.resident.swiglu(x, chosen: chosen, groupSize: groupSize, bits: bits)
    let got = try streamed.swiglu(x, chosen: chosen, groupSize: groupSize, bits: bits)
    eval(want, got)

    #expect(got.shape == want.shape)
    #expect((got - want).abs().max().item(Float.self) == 0)
    #expect(streamed.store.misses > streamed.store.slotCount, "the batch should have evicted")
  }

  /// The one floor left under the slot count: a single token's experts have to fit, because
  /// there is nothing smaller to split into.
  @Test("refuses a token that routes wider than it can hold")
  func refusesAnOversubscribedToken() throws {
    let url = temporary()
    defer { try? FileManager.default.removeItem(at: url) }

    let bank = bank()
    try write(bank, to: url)
    let streamed = StreamedExperts(
      store: try ExpertStore(url: url, layout: bank.layout, slots: 2))

    let x = MLXRandom.normal([1, hidden], dtype: .float32)
    let chosen = MLXArray([Int32](arrayLiteral: 0, 1, 2, 3), [1, 4])
    #expect(throws: BonsaiError.self) {
      _ = try streamed.swiglu(x, chosen: chosen, groupSize: groupSize, bits: bits)
    }
  }
}
