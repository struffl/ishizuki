// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

// Probe, not a test: splits a greedy decode step into GPU time and host time.
@Suite("HostGapProbe", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_PACK"] != nil))
struct HostGapProbe {
  @Test("probe")
  func probe() throws {
    let path = ProcessInfo.processInfo.environment["ISHIZUKI_PACK"]!
    let steps = Int(ProcessInfo.processInfo.environment["ISHIZUKI_STEPS"] ?? "") ?? 64
    let model = try BonsaiModel(directory: URL(filePath: path))
    let prompt = model.tokenizer.encode("Write a short story about a lighthouse keeper.")

    func prefilled() -> (ModelCache, MLXArray) {
      let cache = model.text.makeCache()
      let ids = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
      let logits = model.text(ids, cache: cache)
      eval(logits)
      return (cache, logits[0..., -1, 0...].argMax(axis: -1))
    }

    func sync() -> ([Int], Double) {
      var (cache, next) = prefilled()
      var tokens: [Int] = []
      let start = Date()
      for _ in 0..<steps {
        let token = next.item(Int.self)
        tokens.append(token)
        let step = model.text(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
        eval(step)
        next = step[0..., -1, 0...].argMax(axis: -1)
      }
      return (tokens, -start.timeIntervalSinceNow)
    }

    func pipelined() -> ([Int], Double) {
      var (cache, next) = prefilled()
      var tokens: [Int] = []
      let start = Date()
      for _ in 0..<steps {
        let step = model.text(next.reshaped([1, 1]), cache: cache)
        let following = step[0..., -1, 0...].argMax(axis: -1)
        asyncEval(following)
        tokens.append(next.item(Int.self))
        next = following
      }
      return (tokens, -start.timeIntervalSinceNow)
    }

    func graphOnly() -> Double {
      let (cache, next) = prefilled()
      var built: [MLXArray] = []
      let start = Date()
      var input = next
      for _ in 0..<min(steps, 8) {
        let step = model.text(input.reshaped([1, 1]), cache: cache)
        input = step[0..., -1, 0...].argMax(axis: -1)
        built.append(input)
      }
      let seconds = -start.timeIntervalSinceNow / Double(min(steps, 8))
      eval(built)
      return seconds
    }

    func projections() -> [PackedLinear] {
      var found: [ObjectIdentifier: PackedLinear] = [:]
      var seen = Set<ObjectIdentifier>()
      func walk(_ value: Any, depth: Int) {
        guard depth < 12 else { return }
        if let linear = value as? PackedLinear {
          found[ObjectIdentifier(linear)] = linear
          return
        }
        if type(of: value) is AnyClass {
          let id = ObjectIdentifier(value as AnyObject)
          if seen.contains(id) { return }
          seen.insert(id)
        }
        for child in Mirror(reflecting: value).children { walk(child.value, depth: depth + 1) }
      }
      walk(model.text, depth: 0)
      return Array(found.values)
    }

    func projectionsOnly(width: Int = 1) -> (Int, Double) {
      let all = projections().filter { $0 !== model.text.lmHead }
      let xs = all.map { MLXRandom.normal([1, width, $0.inputDim]).asType(.float16) }
      eval(xs)
      func once() -> Double {
        let start = Date()
        var outs: [MLXArray] = []
        for _ in 0..<4 { outs += zip(all, xs).map { $0($1) } }
        eval(outs)
        return -start.timeIntervalSinceNow / 4
      }
      _ = once()
      return (all.count, min(once(), once()))
    }

    func generator(pipelined: Bool) -> ([Int], Double) {
      BonsaiRuntime.pipelineDecode = pipelined
      defer { BonsaiRuntime.pipelineDecode = true }
      let result = Generator(model: model, politeness: .normal).generate(
        promptTokens: prompt, options: .greedy, maxTokens: steps)
      return (result.tokens, result.stats.generationTokensPerSecond)
    }
    _ = generator(pipelined: true)
    var rates: [Bool: Double] = [:]
    var outputs: [Bool: [Int]] = [:]
    for round in 0..<4 {
      let pipelined = round % 2 == 1
      let (tokens, rate) = generator(pipelined: pipelined)
      rates[pipelined] = max(rates[pipelined] ?? 0, rate)
      outputs[pipelined] = tokens
    }
    print(String(format: "Generator serial   : %6.2f tok/s", rates[false]!))
    print(String(format: "Generator pipelined: %6.2f tok/s", rates[true]!))
    print("Generator tokens identical: \(outputs[false]! == outputs[true]!)")

    _ = sync()
    let (a, syncSeconds) = sync()
    let (b, pipeSeconds) = pipelined()
    let (_, syncAgain) = sync()
    let (_, pipeAgain) = pipelined()
    let build = graphOnly()

    let syncMs = min(syncSeconds, syncAgain) * 1000 / Double(steps)
    let pipeMs = min(pipeSeconds, pipeAgain) * 1000 / Double(steps)
    print(String(format: "sync      : %6.2f ms/step  %6.2f tok/s", syncMs, 1000 / syncMs))
    print(String(format: "pipelined : %6.2f ms/step  %6.2f tok/s", pipeMs, 1000 / pipeMs))
    print(String(format: "graph build (host only): %6.2f ms/step", build * 1000))
    let (count, projectionSeconds) = projectionsOnly()
    print(String(format: "projections only (%d, no lm head): %6.2f ms/step", count, projectionSeconds * 1000))
    for width in [2, 4, 8, 16] {
      print(String(format: "projections only, %2d rows: %6.2f ms", width, projectionsOnly(width: width).1 * 1000))
    }
    for width in [1, 2, 3, 4, 6, 8] {
      let block = MLXArray((0..<width).map { Int32(prompt[$0 % prompt.count]) }).reshaped([1, width])
      var line = String(format: "verify %2d tokens:", width)
      var picks: [[Int32]] = []
      for enabled in [false, true] {
        BonsaiRuntime.useVerifyMatmul = enabled
        let (cache, _) = prefilled()
        let first = model.text(block, cache: cache)
        eval(first)
        picks.append(first.argMax(axis: -1).reshaped([-1]).asArray(Int32.self))
        var best = Double.infinity
        for _ in 0..<5 {
          let start = Date()
          eval(model.text(block, cache: cache))
          best = min(best, -start.timeIntervalSinceNow)
        }
        line += String(format: "  %@ %6.2f ms", enabled ? "new" : "mlx", best * 1000)
      }
      print(line + "  argmax agree: \(picks[0] == picks[1])")
    }
    BonsaiRuntime.useVerifyMatmul = true
    print("tokens identical: \(a == b)")
  }
}
