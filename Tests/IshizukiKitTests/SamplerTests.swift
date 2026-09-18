// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

private let useCPU: Void = Device.setDefault(device: .cpu)

private func survivors(_ logits: [Float], _ options: SamplingOptions) -> Set<Int> {
  _ = useCPU
  let scores = Sampler(options: options)
    .truncatedScores(MLXArray(logits).reshaped([1, logits.count]))
  eval(scores)
  let values = scores.asType(.float32).asArray(Float.self)
  return Set(values.indices.filter { values[$0] > -Float.greatestFiniteMagnitude })
}

private let logits: [Float] = [10, 9, 7, 4, 1]

@Suite("Sampler warper chain")
struct SamplerTests {
  @Test("min-p survivors are invariant to temperature")
  func minPInvariantToTemperature() {
    let reference = survivors(logits, SamplingOptions(minP: 0.05))
    for temperature in [Float(0.1), 0.7, 1.0, 2.0, 5.0, 10.0] {
      let got = survivors(
        logits, SamplingOptions(temperature: temperature, minP: 0.05))
      #expect(got == reference, "temperature \(temperature) changed the min-p survivors")
    }
    #expect(reference.count < logits.count)
    #expect(reference == [0, 1])
  }

  @Test("top-p survivors are invariant to temperature")
  func topPInvariantToTemperature() {
    let reference = survivors(logits, SamplingOptions(topP: 0.9))
    for temperature in [Float(0.1), 1.0, 5.0] {
      let got = survivors(
        logits, SamplingOptions(temperature: temperature, topP: 0.9))
      #expect(got == reference)
    }
    #expect(reference.count < logits.count)
  }

  @Test("min-p in log space matches the softmax formulation")
  func minPMatchesSoftmaxFormulation() {
    for minP in [Float(0.01), 0.05, 0.2, 0.5, 0.9] {
      let got = survivors(logits, SamplingOptions(minP: minP))

      let peak = logits.map { expf($0 - logits.max()!) }.max()!
      let want = Set(
        logits.indices.filter { expf(logits[$0] - logits.max()!) >= minP * peak })
      #expect(got == want, "min-p \(minP)")
    }
  }

  @Test("min-p is clamped to 0...1 so the argmax always survives")
  func minPClamped() {
    #expect(SamplingOptions(minP: 5.0).minP == 1.0)
    #expect(SamplingOptions(minP: -1.0).minP == 0.0)
    #expect(survivors(logits, SamplingOptions(minP: 5.0)) == [0])
  }

  @Test("warpers compose: each one only ever narrows the set")
  func warpersCompose() {
    let topKOnly = survivors(logits, SamplingOptions(topK: 3))
    let both = survivors(logits, SamplingOptions(topK: 3, minP: 0.05))
    #expect(both.isSubset(of: topKOnly))
    #expect(topKOnly == [0, 1, 2])
  }

  @Test("defaults disable every truncation warper")
  func defaultsAreTemperatureOnly() {
    let options = SamplingOptions()
    #expect(options.temperature == 0.7)
    #expect(options.topP == 1.0)
    #expect(options.topK == 0)
    #expect(options.minP == 0.0)
    #expect(options.repetitionPenalty == 1.0)
    #expect(survivors(logits, options).count == logits.count)
  }

  @Test("greedy sampling ignores the warpers and takes the argmax")
  func greedyTakesArgmax() {
    _ = useCPU
    let sampler = Sampler(options: .greedy)
    let row = MLXArray(logits).reshaped([1, logits.count])
    #expect(sampler(row) == 0)
  }
}
