// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Drafted decoding for a served turn: prompt lookup when the reply repeats what came before, the
// pack's own MTP head otherwise, checked a block at a time and exact for greedy and sampled picks.

import Foundation
import MLX
import MLXRandom

extension Generator {
  struct Drafts {
    let lookup: Drafter
    let mtp: MTPDrafter?
  }

  struct DraftedDecode {
    var generated: [Int] = []
    var text = ""
    var stoppedOnEOS = false
    var cancelled = false
    var stats = SpeculativeStats()
  }

  func drafts(
    options: SamplingOptions, constraint: OutputConstraint?, promptEmbeddings: MLXArray?,
    positions: MLXArray?
  ) -> Drafts? {
    guard BonsaiRuntime.speculativeDecode, constraint == nil, promptEmbeddings == nil,
      positions == nil, options.repetitionPenalty == 1, options.presencePenalty == 0
    else { return nil }
    let mtp = model.mtp == nil ? nil : try? MTPDrafter(model: model, kvConfig: kvConfig)
    return Drafts(lookup: lookup?() ?? NgramDrafter(minPatternLength: 3), mtp: mtp)
  }

  func speculate(
    _ drafts: Drafts, logits first: MLXArray, observed start: (MLXArray, [Int])?,
    cache: ModelCache, sampler: Sampler, promptTokens: [Int], maxTokens: Int,
    detokenizer: inout StreamingDetokenizer,
    isCancelled: (@Sendable () -> Bool)?,
    onProgress: ((GenerationProgress) -> Void)?,
    onToken: ((String) -> Bool)?
  ) -> DraftedDecode {
    var out = DraftedDecode()
    var logits = first
    var observed = start
    var forced: Int?
    var context = promptTokens
    var ngramIdle = 0
    let temperature = sampler.options.temperature

    func forward(_ tokens: [Int], allPositions: Bool) -> (MLXArray, MLXArray) {
      let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
      let h = model.text.trunk(inputs: ids, cache: cache)
      let normed = model.text.normed(h)
      return (h, allPositions ? model.text.lmHead(normed) : model.text.lastLogits(normed))
    }

    func emit(_ tokens: [Int]) -> Bool {
      for token in tokens {
        if model.tokenizer.eosTokenIds.contains(token) {
          out.stoppedOnEOS = true
          return false
        }
        context.append(token)
        out.generated.append(token)
        onProgress?(.decode(count: out.generated.count))
        let fragment = detokenizer.append(token)
        if !fragment.isEmpty {
          out.text += fragment
          if let onToken, !onToken(fragment) { return false }
        }
        if out.generated.count >= maxTokens { return false }
      }
      return true
    }

    func scores(_ row: MLXArray) -> MLXArray {
      sampler.truncatedScores(row.reshaped([1, -1]))
    }

    var carry: [Int] = []

    while out.generated.count < maxTokens {
      if isCancelled?() == true {
        out.cancelled = true
        break
      }
      let confirmed = forced ?? sampler.token(logits[0..., -1, 0...]).item(Int.self)
      forced = nil
      if let mtp = drafts.mtp, let observed {
        mtp.observe(hidden: observed.0, nextTokens: observed.1 + [confirmed])
      }
      observed = nil

      var draft: [Int] = []
      let room = BonsaiRuntime.draftLength
      if ngramIdle == 0 {
        draft = drafts.lookup.propose(context: context + [confirmed], count: room)
      } else {
        ngramIdle -= 1
      }
      let fromNgram = !draft.isEmpty
      if draft.isEmpty, let mtp = drafts.mtp {
        draft = mtp.propose(context: context, count: 1)
      }
      if carry.count + 1 + draft.count > 16 { draft = [] }
      out.stats.rounds += 1

      let offset = carry.count
      let block = carry + [confirmed] + draft
      carry = []

      if draft.isEmpty {
        guard emit([confirmed]) else { break }
        let (h, next) = forward(block, allPositions: false)
        eval(next)
        logits = next
        observed = (offset == 0 ? h : h[0..., offset..., 0...], [])
        continue
      }

      out.stats.proposed += draft.count
      let snapshot = cache.snapshot()
      let (h, blockLogits) = forward(block, allPositions: true)
      eval(blockLogits)

      var accepted = 0
      if temperature > 0 {
        let rows = blockLogits[0, offset..., 0...]
        let scaled = sampler.truncatedScores(rows) / temperature
        let ids = MLXArray(draft.map { Int32($0) })
        let chance = takeAlong(
          softmax(scaled[..<draft.count], axis: -1), ids.reshaped([-1, 1]), axis: -1
        ).reshaped([-1])
        let rolls = MLXRandom.uniform(0 ..< 1, [draft.count])
        let spoiled = scaled[..<draft.count]
        spoiled[MLXArray(Int32(0)..<Int32(draft.count)), ids] = MLXArray(-Float.infinity)
        let replacements = MLXRandom.categorical(spoiled, axis: -1)
        let bonus = MLXRandom.categorical(scaled[draft.count...], axis: -1)
        eval(chance, rolls, replacements, bonus)
        let chances = chance.asArray(Float.self)
        let draws = rolls.asArray(Float.self)
        while accepted < draft.count, draws[accepted] < chances[accepted] { accepted += 1 }
        forced =
          accepted < draft.count
          ? Int(replacements.asArray(Int32.self)[accepted]) : bonus.item(Int.self)
      } else {
        let predictions = blockLogits[0, offset...].argMax(axis: -1).asArray(Int32.self)
        while accepted < draft.count, Int(predictions[accepted]) == draft[accepted] {
          accepted += 1
        }
        forced = Int(predictions[accepted])
      }
      out.stats.accepted += accepted
      if fromNgram, accepted == 0 { ngramIdle = 4 }

      let kept = [confirmed] + draft.prefix(accepted)
      if accepted == draft.count {
        logits = blockLogits[0..., (blockLogits.dim(1) - 1)..., 0...]
        observed = (offset == 0 ? h : h[0..., offset..., 0...], Array(block[(offset + 1)...]))
      } else {
        out.stats.rollbacks += 1
        cache.restore(snapshot)
        carry = Array(block[..<offset]) + kept
        if let mtp = drafts.mtp, let forced = forced {
          mtp.observe(
            hidden: h[0..., offset..<(offset + kept.count), 0...],
            nextTokens: Array(kept.dropFirst()) + [forced])
        }
      }

      guard emit(kept) else {
        if accepted == draft.count { cache.restore(snapshot) }
        break
      }

      let delay = Politeness.throttleDelay(for: politeness)
      if delay > 0 { Thread.sleep(forTimeInterval: delay) }
    }
    return out
  }
}
