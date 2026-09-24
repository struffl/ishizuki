// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public struct GenerationStats: Sendable {
  public var promptTokens: Int
  public var generatedTokens: Int
  public var promptSeconds: Double
  public var generationSeconds: Double

  public var promptTokensPerSecond: Double {
    promptSeconds > 0 ? Double(promptTokens) / promptSeconds : 0
  }
  public var generationTokensPerSecond: Double {
    generationSeconds > 0 ? Double(generatedTokens) / generationSeconds : 0
  }
}

public enum GenerationProgress: Sendable {
  case prefill(done: Int, total: Int)
  case decode(count: Int)
}

public struct GenerationResult: Sendable {
  public var tokens: [Int]
  public var text: String
  public var stats: GenerationStats
  public var stoppedOnEOS: Bool
  public var cancelled: Bool = false
  public var speculative: SpeculativeStats?
}

public final class Generator: @unchecked Sendable {
  public let model: BonsaiModel
  public var kvConfig: KVCacheConfig
  public var prefillChunkSize: Int
  public var politeness: Politeness.Level = .normal
  var lookup: (() -> Drafter)?
  var speculativeDecode: Bool?
  var pipelineDecode: Bool?

  public init(
    model: BonsaiModel, prefillChunkSize: Int? = nil,
    kvConfig: KVCacheConfig = KVCacheConfig(),
    politeness: Politeness.Level = .normal
  ) {
    self.model = model
    self.politeness = politeness
    // The slices are fixed-shape, so a chunk that is not their size has nothing to hand over.
    if let bank = BonsaiRuntime.aneBank {
      self.prefillChunkSize = bank.rows
    } else {
      self.prefillChunkSize =
        prefillChunkSize
        ?? Politeness.prefillChunk(
          for: politeness, default: model.streamsExperts ? BonsaiRuntime.streamedPrefillChunk : 512)
    }
    self.kvConfig = kvConfig
  }

  public func generate(
    promptTokens: [Int],
    options: SamplingOptions = SamplingOptions(),
    maxTokens: Int = 512,
    cache: ModelCache? = nil,
    promptEmbeddings: MLXArray? = nil,
    positions: MLXArray? = nil,
    cachedPrefixLength: Int = 0,
    constraint: OutputConstraint? = nil,
    /// A token index to stop the prefill on exactly, so a caller can take a rewind point
    /// somewhere other than a chunk boundary. The chunk is cut short to land on it.
    checkpointAt: Int? = nil,
    isCancelled: (@Sendable () -> Bool)? = nil,
    onCheckpoint: (() -> Void)? = nil,
    onPrefilled: (() -> Void)? = nil,
    onProgress: ((GenerationProgress) -> Void)? = nil,
    onToken: ((String) -> Bool)? = nil
  ) -> GenerationResult {
    let cache = cache ?? model.backbone.makeCache(kvConfig: kvConfig)
    let sampler = Sampler(options: options)
    var detokenizer = StreamingDetokenizer(tokenizer: model.tokenizer)

    let promptStart = Date()
    var logits: MLXArray
    var prefilled = 0
    let drafting = drafts(
      options: options, constraint: constraint, promptEmbeddings: promptEmbeddings,
      positions: positions)
    var observed: (MLXArray, [Int])?

    let prefillTotal = max(0, promptTokens.count - cachedPrefixLength)
    onProgress?(.prefill(done: 0, total: prefillTotal))

    func abandoned() -> GenerationResult {
      GenerationResult(
        tokens: [], text: "",
        stats: GenerationStats(
          promptTokens: prefilled, generatedTokens: 0,
          promptSeconds: -promptStart.timeIntervalSinceNow, generationSeconds: 0),
        stoppedOnEOS: false, cancelled: true)
    }

    if isCancelled?() == true { return abandoned() }

    let replay = model.backbone.replayTail.map {
      max(cachedPrefixLength, promptTokens.count - $0)
    } ?? cachedPrefixLength

    if let promptEmbeddings {
      let ids = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptTokens.count])
      if replay > 0 {
        model.backbone.encode(
          inputs: ids[0..., ..<replay], inputEmbeddings: promptEmbeddings[0..., ..<replay],
          cache: cache)
      }
      logits = model.backbone.lastLogits(
        inputs: ids[0..., replay...], inputEmbeddings: promptEmbeddings[0..., replay...],
        cache: cache, positions: positions?[.ellipsis, replay...])
      eval(logits)
      prefilled = prefillTotal
      onProgress?(.prefill(done: prefillTotal, total: prefillTotal))
    } else {
      precondition(!promptTokens.isEmpty, "generate requires a non-empty prompt")
      precondition(
        cachedPrefixLength < promptTokens.count,
        "at least one prompt token must be processed to produce logits")
      var index = cachedPrefixLength
      var last: MLXArray?
      while index < promptTokens.count {
        if isCancelled?() == true { return abandoned() }
        var end = min(index + prefillChunkSize, promptTokens.count)
        if index < replay { end = min(end, replay) }
        if let stop = checkpointAt, index < stop, stop < end { end = stop }
        let chunk = MLXArray(promptTokens[index..<end].map { Int32($0) })
          .reshaped([1, end - index])
        if index < replay {
          model.backbone.encode(inputs: chunk, inputEmbeddings: nil, cache: cache)
          index = end
          prefilled = index - cachedPrefixLength
          onProgress?(.prefill(done: prefilled, total: prefillTotal))
          if index == checkpointAt { onCheckpoint?() }
          continue
        }
        let trunk: MLXArray
        if let dspark = drafting?.dspark {
          let step = dspark.forward(chunk, cache: cache)
          dspark.commit(step.hidden, start: step.start, count: end - index)
          trunk = step.trunk
        } else if let dflash = drafting?.dflash {
          let step = dflash.backbone.trunk(inputs: chunk, cache: cache, taps: dflash.taps)
          if let taps = step.taps { dflash.observe(taps, at: index) }
          trunk = step.trunk
        } else {
          trunk = model.backbone.trunk(inputs: chunk, cache: cache)
        }
        last = model.backbone.normed(trunk)
        eval(last!)
        if let mtp = drafting?.mtp {
          if end < promptTokens.count {
            mtp.observe(hidden: trunk, nextTokens: Array(promptTokens[(index + 1)...end]))
          } else {
            observed = (trunk, Array(promptTokens[(index + 1)..<end]))
          }
        }
        index = end
        prefilled = index - cachedPrefixLength
        onProgress?(.prefill(done: prefilled, total: prefillTotal))
        if index == checkpointAt { onCheckpoint?() }
      }
      logits = model.backbone.lastLogits(last!)
      eval(logits)
    }
    let promptSeconds = -promptStart.timeIntervalSinceNow
    onPrefilled?()

    guard maxTokens > 0 else {
      return GenerationResult(
        tokens: [], text: "",
        stats: GenerationStats(
          promptTokens: prefilled, generatedTokens: 0, promptSeconds: promptSeconds,
          generationSeconds: 0),
        stoppedOnEOS: false)
    }

    var decodePosition: Int? = positions.map { $0.max().item(Int.self) + 1 }

    let generationStart = Date()
    var generated: [Int] = []
    var text = ""
    var stoppedOnEOS = false
    var cancelled = false

    var nextLogits = logits[0..., -1, 0...]
    onProgress?(.decode(count: 0))

    if let dflash = drafting?.dflash {
      let decoded = speculateBlocks(
        dflash, logits: logits, cache: cache, sampler: sampler, maxTokens: maxTokens,
        detokenizer: &detokenizer,
        isCancelled: isCancelled, onProgress: onProgress, onToken: onToken)
      return GenerationResult(
        tokens: decoded.generated, text: decoded.text,
        stats: GenerationStats(
          promptTokens: promptTokens.count - cachedPrefixLength,
          generatedTokens: decoded.generated.count, promptSeconds: promptSeconds,
          generationSeconds: -generationStart.timeIntervalSinceNow),
        stoppedOnEOS: decoded.stoppedOnEOS, cancelled: decoded.cancelled,
        speculative: decoded.stats)
    }

    if let drafting, drafting.hasHead {
      let decoded = speculate(
        drafting, logits: logits, observed: observed, cache: cache, sampler: sampler,
        promptTokens: promptTokens, maxTokens: maxTokens, detokenizer: &detokenizer,
        isCancelled: isCancelled, onProgress: onProgress, onToken: onToken)
      return GenerationResult(
        tokens: decoded.generated, text: decoded.text,
        stats: GenerationStats(
          promptTokens: promptTokens.count - cachedPrefixLength,
          generatedTokens: decoded.generated.count, promptSeconds: promptSeconds,
          generationSeconds: -generationStart.timeIntervalSinceNow),
        stoppedOnEOS: decoded.stoppedOnEOS, cancelled: decoded.cancelled,
        speculative: decoded.stats)
    }

    func stepPositions() -> MLXArray? {
      decodePosition.map { MLXArray([Int32($0), Int32($0), Int32($0)]).reshaped([3, 1]) }
    }

    let lookup = drafting?.lookup
    var drafted: SpeculativeStats?
    if constraint == nil, pipelineDecode ?? BonsaiRuntime.pipelineDecode {
      var stats = SpeculativeStats()
      var carry: [Int] = []
      var idle = 0
      var backoff = 4
      var pending = sampler.token(nextLogits, recentTokens: promptTokens)
      asyncEval(pending)
      decoding: while generated.count < maxTokens {
        if isCancelled?() == true {
          cancelled = true
          break
        }
        let snapshot = cache.snapshot()
        let input =
          carry.isEmpty
          ? pending.reshaped([1, 1])
          : concatenated([MLXArray(carry.map { Int32($0) }), pending.reshaped([-1]).asType(.int32)])
            .reshaped([1, -1])
        carry = []
        let step = model.backbone(input, cache: cache, positions: stepPositions())
        if decodePosition != nil { decodePosition! += 1 }
        let following = sampler.token(
          step[0..., -1, 0...], recentTokens: promptTokens + generated, pending: pending)
        asyncEval(following)

        let token = pending.item(Int.self)
        if model.tokenizer.eosTokenIds.contains(token) {
          cache.restore(snapshot)
          stoppedOnEOS = true
          break
        }
        generated.append(token)
        onProgress?(.decode(count: generated.count))

        let fragment = detokenizer.append(token)
        if !fragment.isEmpty {
          text += fragment
          if let onToken, !onToken(fragment) {
            cache.restore(snapshot)
            break
          }
        }

        let delay = Politeness.throttleDelay(for: politeness)
        if delay > 0 { Thread.sleep(forTimeInterval: delay) }
        pending = following

        guard let lookup, generated.count < maxTokens else { continue }
        stats.rounds += 1
        if idle > 0 {
          idle -= 1
          continue
        }
        let draft = lookup.propose(
          context: promptTokens + generated, count: BonsaiRuntime.draftLength)
        guard !draft.isEmpty else { continue }

        stats.proposed += draft.count
        let settled = cache.snapshot()
        let block = model.backbone(
          MLXArray(draft.map { Int32($0) }).reshaped([1, draft.count]), cache: cache)
        let picks = sampler.token(block[0])
        eval(following, picks)
        let candidates = [following.item(Int.self)] + picks.asArray(Int32.self).map(Int.init)
        var accepted = 0
        while accepted < draft.count, candidates[accepted] == draft[accepted] { accepted += 1 }
        stats.accepted += accepted
        if accepted == 0 {
          idle = backoff
          backoff = min(backoff * 2, 32)
        } else {
          backoff = 4
        }
        if accepted < draft.count {
          stats.rollbacks += 1
          cache.restore(settled)
          carry = Array(draft[..<accepted])
        }
        if accepted > 0 { pending = picks[(accepted - 1)..<accepted] }

        for kept in draft[..<accepted] {
          if model.tokenizer.eosTokenIds.contains(kept) {
            cache.restore(settled)
            stoppedOnEOS = true
            break decoding
          }
          generated.append(kept)
          onProgress?(.decode(count: generated.count))
          let fragment = detokenizer.append(kept)
          if !fragment.isEmpty {
            text += fragment
            if let onToken, !onToken(fragment) {
              cache.restore(settled)
              break decoding
            }
          }
          if generated.count >= maxTokens {
            cache.restore(settled)
            break decoding
          }
        }
      }
      if lookup != nil { drafted = stats }
    } else {
      for _ in 0..<maxTokens {
        if isCancelled?() == true {
          cancelled = true
          break
        }
        let token: Int
        if let constraint {
          // A complete document may stop here; an exhausted one must.
          guard
            let picked = sampler(
              nextLogits, allowed: constraint.allowedTokens(tokenizer: model.tokenizer))
          else {
            stoppedOnEOS = constraint.isComplete
            break
          }
          constraint.accept(model.tokenizer.tokenBytes(picked))
          token = picked
        } else {
          token = sampler(nextLogits, recentTokens: promptTokens + generated)
        }

        if model.tokenizer.eosTokenIds.contains(token) {
          stoppedOnEOS = true
          break
        }
        generated.append(token)
        onProgress?(.decode(count: generated.count))

        let fragment = detokenizer.append(token)
        if !fragment.isEmpty {
          text += fragment
          if let onToken, !onToken(fragment) { break }
        }

        let delay = Politeness.throttleDelay(for: politeness)
        if delay > 0 { Thread.sleep(forTimeInterval: delay) }

        let input = MLXArray([Int32(token)]).reshaped([1, 1])
        let step = model.backbone(input, cache: cache, positions: stepPositions())
        eval(step)
        if decodePosition != nil { decodePosition! += 1 }
        nextLogits = step[0..., -1, 0...]
      }
    }
    let generationSeconds = -generationStart.timeIntervalSinceNow

    return GenerationResult(
      tokens: generated,
      text: text,
      stats: GenerationStats(
        promptTokens: promptTokens.count - cachedPrefixLength,
        generatedTokens: generated.count,
        promptSeconds: promptSeconds,
        generationSeconds: generationSeconds),
      stoppedOnEOS: stoppedOnEOS,
      cancelled: cancelled,
      speculative: drafted)
  }
}
