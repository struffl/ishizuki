// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public protocol Drafter: AnyObject {
  func propose(context: [Int], count: Int) -> [Int]

  func commit(tokens: [Int])

  func reset()
}

public final class NgramDrafter: Drafter {
  public let maxPatternLength: Int
  public let minPatternLength: Int

  public init(maxPatternLength: Int = 8, minPatternLength: Int = 2) {
    self.maxPatternLength = maxPatternLength
    self.minPatternLength = minPatternLength
  }

  public func propose(context: [Int], count: Int) -> [Int] {
    guard count > 0, context.count > minPatternLength else { return [] }

    var length = min(maxPatternLength, context.count - 1)
    while length >= minPatternLength {
      let pattern = Array(context.suffix(length))
      var start = context.count - length - 1
      while start >= 0 {
        if Array(context[start..<(start + length)]) == pattern {
          let from = start + length
          let to = min(from + count, context.count)
          if to > from { return Array(context[from..<to]) }
        }
        start -= 1
      }
      length -= 1
    }
    return []
  }

  public func commit(tokens: [Int]) {}
  public func reset() {}
}

public final class MTPDrafter: Drafter {
  public init(model: BonsaiModel) throws {
    guard model.hasMTP else {
      throw BonsaiError.missingComponent(
        "this pack ships no MTP head (components.mtp = false, mtp_num_hidden_layers = 0, "
          + "no mtp.* tensors). Use NgramDrafter, or supply a separate draft model.")
    }
    throw BonsaiError.missingComponent("MTP head loading is not implemented")
  }

  public func propose(context: [Int], count: Int) -> [Int] { [] }
  public func commit(tokens: [Int]) {}
  public func reset() {}
}

public struct SpeculativeStats: Sendable {
  public var proposed = 0
  public var accepted = 0
  public var rounds = 0
  public var rollbacks = 0

  public var acceptanceRate: Double {
    proposed > 0 ? Double(accepted) / Double(proposed) : 0
  }
  public var tokensPerRound: Double {
    rounds > 0 ? Double(accepted + rounds) / Double(rounds) : 0
  }
}

public final class SpeculativeDecoder: @unchecked Sendable {
  public let model: BonsaiModel
  public let drafter: Drafter
  public let draftLength: Int

  public init(model: BonsaiModel, drafter: Drafter, draftLength: Int = 4) {
    self.model = model
    self.drafter = drafter
    self.draftLength = draftLength
  }

  public func generate(
    promptTokens: [Int],
    maxTokens: Int = 256,
    onToken: ((String) -> Bool)? = nil
  ) -> (result: GenerationResult, speculative: SpeculativeStats) {
    let cache = model.text.makeCache()
    var detokenizer = StreamingDetokenizer(tokenizer: model.tokenizer)
    var stats = SpeculativeStats()

    let promptStart = Date()
    let promptIds = MLXArray(promptTokens.map { Int32($0) })
      .reshaped([1, promptTokens.count])
    var logits = model.text.lastLogits(inputs: promptIds, cache: cache)
    eval(logits)
    let promptSeconds = -promptStart.timeIntervalSinceNow

    var context = promptTokens
    var generated: [Int] = []
    var text = ""
    var stoppedOnEOS = false
    let generationStart = Date()

    outer: while generated.count < maxTokens {
      let confirmed = logits[0, -1].argMax().item(Int.self)

      var emitted = [confirmed]
      let draft = drafter.propose(
        context: context + [confirmed], count: draftLength)
      stats.rounds += 1

      if draft.isEmpty {
        if !append(
          &emitted, to: &context, &generated, &text, &detokenizer,
          onToken: onToken, stopped: &stoppedOnEOS, limit: maxTokens)
        {
          break outer
        }
        let input = MLXArray([Int32(confirmed)]).reshaped([1, 1])
        logits = model.text(input, cache: cache)
        eval(logits)
        continue
      }

      stats.proposed += draft.count
      let snapshot = cache.snapshot()

      let block = [confirmed] + draft
      let blockIds = MLXArray(block.map { Int32($0) }).reshaped([1, block.count])
      let blockLogits = model.text(blockIds, cache: cache)
      eval(blockLogits)

      let predictions = blockLogits[0].argMax(axis: -1).asArray(Int32.self)
      var acceptedCount = 0
      while acceptedCount < draft.count,
        Int(predictions[acceptedCount]) == draft[acceptedCount]
      {
        acceptedCount += 1
      }
      stats.accepted += acceptedCount
      emitted.append(contentsOf: draft.prefix(acceptedCount))

      if acceptedCount == draft.count {
        logits = blockLogits[0..., -1..<blockLogits.dim(1), 0...]
      } else {
        stats.rollbacks += 1
        cache.restore(snapshot)
        let replay = [confirmed] + Array(draft.prefix(acceptedCount))
        let replayIds = MLXArray(replay.map { Int32($0) }).reshaped([1, replay.count])
        logits = model.text(replayIds, cache: cache)
        eval(logits)
      }

      if !append(
        &emitted, to: &context, &generated, &text, &detokenizer,
        onToken: onToken, stopped: &stoppedOnEOS, limit: maxTokens)
      {
        break outer
      }
      drafter.commit(tokens: emitted)
    }

    let generationSeconds = -generationStart.timeIntervalSinceNow
    return (
      GenerationResult(
        tokens: generated, text: text,
        stats: GenerationStats(
          promptTokens: promptTokens.count, generatedTokens: generated.count,
          promptSeconds: promptSeconds, generationSeconds: generationSeconds),
        stoppedOnEOS: stoppedOnEOS),
      stats
    )
  }

  private func append(
    _ tokens: inout [Int], to context: inout [Int], _ generated: inout [Int],
    _ text: inout String, _ detokenizer: inout StreamingDetokenizer,
    onToken: ((String) -> Bool)?, stopped: inout Bool, limit: Int
  ) -> Bool {
    for token in tokens {
      if model.tokenizer.eosTokenIds.contains(token) {
        stopped = true
        return false
      }
      context.append(token)
      generated.append(token)
      let fragment = detokenizer.append(token)
      if !fragment.isEmpty {
        text += fragment
        if let onToken, !onToken(fragment) { return false }
      }
      if generated.count >= limit { return false }
    }
    return true
  }
}
