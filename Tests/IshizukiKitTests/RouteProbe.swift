// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

// Probe, not a test: records which experts every sparse layer routes to, for a cache simulation.
@Suite(
  "RouteProbe",
  .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_ROUTE_PACK"] != nil))
struct RouteProbe {
  static let prompts = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a hash map handles collisions, with a small example in Swift.",
    "What were the main causes of the French Revolution? Answer in a few paragraphs.",
    "Solve step by step: a train leaves at 3pm going 80 km/h, another at 4pm going 120 km/h. When does the second catch up?",
    "Écris un poème court sur la mer en hiver.",
    "Write a Python function that parses a CSV file and returns the average of each numeric column.",
  ]

  @Test("record")
  func record() throws {
    let env = ProcessInfo.processInfo.environment
    if let slots = env["ISHIZUKI_EXPERT_SLOTS"].flatMap(Int.init) {
      BonsaiRuntime.expertSlots = slots
    }
    BonsaiRuntime.lockExpertSlots = env["ISHIZUKI_LOCK_SLOTS"] != "0"
    let model = try BonsaiModel(path: URL(filePath: env["ISHIZUKI_ROUTE_PACK"]!))
    let prompts = Array(Self.prompts.prefix(Int(env["ISHIZUKI_PROMPTS"] ?? "") ?? Self.prompts.count))
    let output = URL(filePath: env["ISHIZUKI_ROUTE_OUT"] ?? NSTemporaryDirectory() + "routes.json")
    let steps = Int(env["ISHIZUKI_STEPS"] ?? "") ?? 256
    let text: TextModel = model.text
    MLXRandom.seed(7)

    var pending: [(Int, MLXArray)] = []
    BonsaiRuntime.onRoute = { layer, chosen in pending.append((layer, chosen)) }
    defer { BonsaiRuntime.onRoute = nil }

    func drain() -> [String: [[Int32]]] {
      eval(pending.map(\.1))
      var byLayer: [String: [[Int32]]] = [:]
      for (layer, chosen) in pending {
        let k = chosen.dim(-1)
        let flat = chosen.reshaped([-1]).asArray(Int32.self)
        for row in stride(from: 0, to: flat.count, by: k) {
          byLayer[String(layer), default: []].append(Array(flat[row..<(row + k)]))
        }
      }
      pending.removeAll()
      return byLayer
    }

    var sessions: [[String: Any]] = []
    let started = Date()
    for prompt in prompts {
      let ids = model.tokenizer.encode(prompt)
      let cache = text.makeCache()
      var logits = text(MLXArray(ids.map(Int32.init)).reshaped([1, ids.count]), cache: cache)
      let prefill = drain()
      var decode: [String: [[Int32]]] = [:]
      var emitted: [Int] = []
      let decodeStart = Date()
      for _ in 0..<steps {
        let last = logits[0..., -1, 0...].asType(.float32) / 0.7
        let token = categorical(last).item(Int32.self)
        emitted.append(Int(token))
        logits = text(MLXArray([token]).reshaped([1, 1]), cache: cache)
        for (layer, rows) in drain() { decode[layer, default: []] += rows }
      }
      let rate = Double(steps) / -decodeStart.timeIntervalSinceNow
      let traffic = model.store.expertTraffic
      print(
        String(format: "--- %.2f tok/s, expert hit rate so far %.1f%% --- ", rate,
          100 * (traffic?.hitRate ?? 0))
          + "\(prompt.prefix(40))… → "
          + model.tokenizer.decode(emitted).prefix(160).replacingOccurrences(of: "\n", with: "⏎"))
      sessions.append(["prompt": prompt, "prefill": prefill, "decode": decode])
    }
    let config = text.config
    let payload: [String: Any] = [
      "experts": config.numExperts ?? 0, "topK": config.numExpertsPerTok ?? 0,
      "sessions": sessions,
    ]
    try JSONSerialization.data(withJSONObject: payload).write(to: output)
    print(String(format: "routes for %d prompts x %d tokens in %.0f s → %@",
      prompts.count, steps, -started.timeIntervalSinceNow, output.path))
  }
}
