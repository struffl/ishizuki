// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A DFlash draft beside a backbone, and the drafted loop that verifies a block per round.

import Foundation
import MLX
import MLXRandom

/// DFlash drafting for a served turn.
///
/// The draft reads the backbone's taps at every position the backbone keeps, each once and in
/// order: a verify's taps are handed over only for the positions it kept. What it has not read
/// yet waits here until the next proposal, which is a block after the token just confirmed.
final class DFlashDrafter {
  let draft: DFlashDraft
  let backbone: TextModel
  private let caches: [DFlashDraft.LayerCache]
  private var pending: MLXArray?
  private var pendingStart = 0
  private(set) var observed = 0

  init(draft: DFlashDraft, backbone: TextModel) {
    self.draft = draft
    self.backbone = backbone
    self.caches = draft.makeCaches()
  }

  var taps: [Int] { draft.config.targetLayers }

  /// Taps for positions `start ..< start + n`. Only as many as the draft's window reaches are
  /// kept, which bounds what a long prompt leaves waiting.
  func observe(_ taps: MLXArray, at start: Int) {
    if pending == nil { pendingStart = start }
    var rows = pending.map { concatenated([$0, taps], axis: 1) } ?? taps
    if let window = draft.config.window, rows.dim(1) > window - 1 {
      let drop = rows.dim(1) - (window - 1)
      rows = rows[0..., drop...]
      pendingStart += drop
    }
    pending = rows
    observed = start + taps.dim(1)
  }

  /// The drafts after `token`, which sits at `position`, the first the backbone has not run,
  /// left on the GPU so the verify can be queued behind them before they are read.
  func propose(after token: Int, at position: Int) -> MLXArray? {
    guard let pending, observed == position else { return nil }
    if caches.first?.keys == nil {
      for cache in caches { cache.offset = pendingStart }
    }
    let size = draft.config.blockSize
    let block = MLXArray(
      [Int32(token)] + Array(repeating: Int32(draft.config.maskToken), count: size - 1)
    ).reshaped([1, size])
    let embedded = backbone.embedTokens(block)
    let hidden = draft.hidden(block: embedded, taps: pending, caches: caches)
    self.pending = nil
    let logits = backbone.lmHead(hidden.asType(embedded.dtype))
    return draft.drafts(hidden: hidden, logits: logits, anchor: block[0..., 0])
      .asType(.int32).reshaped([-1])
  }
}

extension Generator {
  /// Decoding with a DFlash draft: a block of drafts a round, verified in one forward and kept
  /// up to the first the backbone would not have picked. A sampled round keeps a draft with the
  /// chance the backbone gives it and, at the first it turns down, samples again with that token
  /// struck out, which leaves the distribution exactly the backbone's. A partial accept keeps
  /// what agreed in the caches rather than running it again, so a verify is one block wide.
  ///
  /// Drafting only pays while the draft is right often enough, which prose often is not. The
  /// loop keeps the rate of its drafted rounds against plain steps, which feed the draft's
  /// context all the same, and runs whichever is ahead, looking at the other now and then.
  func speculateBlocks(
    _ drafter: DFlashDrafter, logits first: MLXArray, cache: ModelCache, sampler: Sampler,
    maxTokens: Int, detokenizer: inout StreamingDetokenizer,
    isCancelled: (@Sendable () -> Bool)?,
    onProgress: ((GenerationProgress) -> Void)?,
    onToken: ((String) -> Bool)?
  ) -> DraftedDecode {
    var out = DraftedDecode()
    let text = drafter.backbone
    let temperature = sampler.options.temperature
    var next = sampler.token(first[0..., -1, 0...]).item(Int.self)
    cache.recordsSteps = true
    defer { cache.recordsSteps = false }

    func emit(_ tokens: [Int]) -> Bool {
      for token in tokens {
        if model.tokenizer.eosTokenIds.contains(token) {
          out.stoppedOnEOS = true
          return false
        }
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

    var drafting = true
    var draftedRate: Double?
    var plainRate: Double?
    var streak = 0
    while out.generated.count < maxTokens {
      if isCancelled?() == true {
        out.cancelled = true
        break
      }
      let started = Date()
      let start = cache.offset
      let current = next
      let proposal = drafting ? drafter.propose(after: current, at: start) : nil
      if let proposal { asyncEval(proposal) }
      let anchor = MLXArray([Int32(current)])
      let input = proposal.map { concatenated([anchor, $0]) } ?? anchor
      let snapshot = cache.snapshot()
      let step = text.trunk(inputs: input.reshaped([1, -1]), cache: cache, taps: drafter.taps)
      let logits = text.lmHead(text.normed(step.trunk))[0]

      var accepted = 0
      var drafts: [Int] = []
      if let proposal, temperature > 0 {
        let n = proposal.dim(0)
        let scaled = sampler.truncatedScores(logits) / temperature
        let chance = takeAlong(
          softmax(scaled[..<n], axis: -1), proposal.reshaped([-1, 1]), axis: -1
        ).reshaped([-1])
        let rolls = MLXRandom.uniform(0 ..< 1, [n])
        let spoiled = scaled[..<n]
        spoiled[MLXArray(Int32(0)..<Int32(n)), proposal] = MLXArray(-Float.infinity)
        let replacements = MLXRandom.categorical(spoiled, axis: -1)
        let bonus = MLXRandom.categorical(scaled[n...], axis: -1)
        eval(chance, rolls, replacements, bonus, proposal)
        drafts = proposal.asArray(Int32.self).map(Int.init)
        let chances = chance.asArray(Float.self)
        let draws = rolls.asArray(Float.self)
        while accepted < n, draws[accepted] < chances[accepted] { accepted += 1 }
        next =
          accepted < n
          ? Int(replacements.asArray(Int32.self)[accepted]) : bonus.item(Int.self)
      } else if let proposal {
        let picks = argMax(logits, axis: -1)
        eval(picks, proposal)
        let chosen = picks.asArray(Int32.self).map(Int.init)
        drafts = proposal.asArray(Int32.self).map(Int.init)
        while accepted < drafts.count, chosen[accepted] == drafts[accepted] { accepted += 1 }
        next = chosen[accepted]
      } else {
        next = sampler.token(logits).item(Int.self)
      }
      if proposal != nil {
        out.stats.rounds += 1
        out.stats.proposed += drafts.count
        out.stats.accepted += accepted
      }
      let kept = [current] + drafts.prefix(accepted)
      if accepted < drafts.count {
        out.stats.rollbacks += 1
        cache.keep(kept.count, since: snapshot)
      }
      if let taps = step.taps { drafter.observe(taps[0..., ..<kept.count], at: start) }

      guard emit(kept) else {
        cache.restore(snapshot)
        break
      }

      let rate = Double(kept.count) / max(-started.timeIntervalSinceNow, 1e-6)
      if proposal != nil {
        draftedRate = draftedRate.map { 0.7 * $0 + 0.3 * rate } ?? rate
      } else {
        plainRate = plainRate.map { 0.7 * $0 + 0.3 * rate } ?? rate
      }
      streak += 1
      if drafting {
        let losing = plainRate.map { draftedRate! < $0 } ?? false
        if losing || streak >= (plainRate == nil ? 4 : 32) {
          drafting = false
          streak = 0
        }
      } else if streak >= 8 || (plainRate ?? 0) < (draftedRate ?? .infinity) {
        drafting = true
        streak = 0
      }

      let delay = Politeness.throttleDelay(for: politeness)
      if delay > 0 { Thread.sleep(forTimeInterval: delay) }
    }
    return out
  }
}
