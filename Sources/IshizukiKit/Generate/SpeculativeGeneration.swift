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
      if ngramIdle == 0 {
        draft = drafts.lookup.propose(
          context: context + [confirmed], count: BonsaiRuntime.draftLength)
      } else {
        ngramIdle -= 1
      }
      let fromNgram = !draft.isEmpty
      if draft.isEmpty, let mtp = drafts.mtp {
        draft = mtp.propose(context: context, count: 1)
      }
      out.stats.rounds += 1

      if draft.isEmpty {
        guard emit([confirmed]) else { break }
        let (h, next) = forward([confirmed], allPositions: false)
        eval(next)
        logits = next
        observed = (h, [])
        continue
      }

      out.stats.proposed += draft.count
      let snapshot = cache.snapshot()
      let block = [confirmed] + draft
      let (h, blockLogits) = forward(block, allPositions: true)
      eval(blockLogits)

      var accepted = 0
      if temperature > 0 {
        while accepted < draft.count {
          let row = scores(blockLogits[0, accepted]) / temperature
          let probability = softmax(row, axis: -1)[0, draft[accepted]].item(Float.self)
          if MLXRandom.uniform(0 ..< 1, [1]).item(Float.self) < probability {
            accepted += 1
            continue
          }
          let mask = MLXArray.zeros([row.dim(-1)], dtype: .float32)
          mask[draft[accepted]] = MLXArray(-Float.infinity)
          forced = MLXRandom.categorical(row + mask, axis: -1).item(Int.self)
          break
        }
      } else {
        let predictions = blockLogits[0].argMax(axis: -1).asArray(Int32.self)
        while accepted < draft.count, Int(predictions[accepted]) == draft[accepted] {
          accepted += 1
        }
      }
      out.stats.accepted += accepted
      if fromNgram, accepted == 0 { ngramIdle = 4 }

      let kept = [confirmed] + draft.prefix(accepted)
      if accepted == draft.count {
        logits = blockLogits[0..., (blockLogits.dim(1) - 1)..., 0...]
        observed = (h, Array(block.dropFirst()))
      } else {
        out.stats.rollbacks += 1
        cache.restore(snapshot)
        let (replayed, next) = forward(kept, allPositions: false)
        eval(next)
        logits = next
        observed = (replayed, Array(kept.dropFirst()))
      }

      guard emit(kept) else {
        cache.restore(snapshot)
        break
      }

      let delay = Politeness.throttleDelay(for: politeness)
      if delay > 0 { Thread.sleep(forTimeInterval: delay) }
    }
    return out
  }
}
