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

/// A drafter that reads the backbone's own activation rather than the token history. The
/// decoder hands it every forward pass it makes, and takes its proposal from that instead of
/// from the context.
public protocol HiddenStateDrafter: Drafter {
  /// `hidden` is the pre-norm activation of the span just run; `nextTokens` is what each of
  /// those positions is followed by, so the draft head advances in lockstep with the backbone.
  func observe(hidden: MLXArray, nextTokens: [Int])
}

/// Drafting with the multi-token-prediction head a pack ships beside the backbone.
///
/// The head runs over the same span the backbone just did, one position offset, so its cache
/// tracks the backbone's without any separate bookkeeping. It proposes one token per round —
/// the head predicts t+2 from t, and chaining it further would need the draft's own hidden
/// state fed back, which is left alone rather than guessed at.
public final class MTPDrafter: HiddenStateDrafter {
  private let model: BonsaiModel
  private let head: MTPHead
  private let cache: ModelCache
  private var pending: [Int] = []

  public var depth: Int { 1 }

  public init(model: BonsaiModel, kvConfig: KVCacheConfig = KVCacheConfig()) throws {
    guard let head = model.mtp else {
      throw BonsaiError.missingComponent(
        "this pack ships no MTP head (components.mtp = false, mtp_num_hidden_layers = 0, "
          + "no mtp.* tensors). Use NgramDrafter, or supply a separate draft model.")
    }
    self.model = model
    self.head = head
    self.cache = head.makeCache(kvConfig: kvConfig)
  }

  public func observe(hidden: MLXArray, nextTokens: [Int]) {
    guard hidden.dim(1) == nextTokens.count, !nextTokens.isEmpty else {
      pending = []
      return
    }
    let ids = MLXArray(nextTokens.map { Int32($0) }).reshaped([1, nextTokens.count])
    let drafted = head(
      hidden: hidden, embeddings: model.backbone.embed(ids), cache: cache)
    let logits = model.backbone.lastLogits(drafted)
    pending = [logits[0, -1].argMax().item(Int.self)]
  }

  public func propose(context: [Int], count: Int) -> [Int] {
    guard count > 0 else { return [] }
    return Array(pending.prefix(count))
  }

  public func commit(tokens: [Int]) {}

  public func reset() {
    cache.reset()
    pending = []
  }
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
    let cache = model.backbone.makeCache()
    var detokenizer = StreamingDetokenizer(tokenizer: model.tokenizer)
    var stats = SpeculativeStats()

    // A head-based drafter needs the activation behind each forward, not just its tokens, so
    // every pass keeps its hidden states until the token that follows them is known.
    let hiddenDrafter = drafter as? HiddenStateDrafter
    var observed: (hidden: MLXArray, following: [Int])?

    // Only a draft block needs logits at every position, to check the proposals against. The
    // vocabulary projection is the widest matmul in the model, so everything else takes it once.
    func forward(_ tokens: [Int], allPositions: Bool = false) -> MLXArray {
      let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
      let h = model.backbone.trunk(inputs: ids, cache: cache)
      if hiddenDrafter != nil { observed = (h, Array(tokens.dropFirst())) }
      let normed = model.backbone.normed(h)
      return allPositions ? model.backbone.logits(normed) : model.backbone.lastLogits(normed)
    }

    // The span just run is handed over once its trailing token is known, which keeps the draft
    // head's cache advancing over exactly the positions the backbone kept.
    func handOver(confirmed: Int) {
      guard let hiddenDrafter, let observed else { return }
      hiddenDrafter.observe(
        hidden: observed.hidden, nextTokens: observed.following + [confirmed])
    }

    let promptStart = Date()
    var logits = forward(promptTokens)
    eval(logits)
    let promptSeconds = -promptStart.timeIntervalSinceNow

    var context = promptTokens
    var generated: [Int] = []
    var text = ""
    var stoppedOnEOS = false
    let generationStart = Date()

    outer: while generated.count < maxTokens {
      let confirmed = logits[0, -1].argMax().item(Int.self)
      handOver(confirmed: confirmed)

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
        logits = forward([confirmed])
        eval(logits)
        continue
      }

      stats.proposed += draft.count
      let snapshot = cache.snapshot()

      let block = [confirmed] + draft
      let blockLogits = forward(block, allPositions: true)
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
        // Only the accepted prefix is replayed, so what the draft head is handed next round is
        // the span the backbone actually kept.
        logits = forward([confirmed] + Array(draft.prefix(acceptedCount)))
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
