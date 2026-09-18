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
}

public final class Generator: @unchecked Sendable {
  public let model: BonsaiModel
  public var kvConfig: KVCacheConfig
  public var prefillChunkSize: Int
  public var politeness: Politeness.Level = .adaptive

  public init(
    model: BonsaiModel, prefillChunkSize: Int? = nil,
    kvConfig: KVCacheConfig = KVCacheConfig(),
    politeness: Politeness.Level = .adaptive
  ) {
    self.model = model
    self.politeness = politeness
    self.prefillChunkSize = prefillChunkSize ?? Politeness.prefillChunk(for: politeness)
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
    onProgress: ((GenerationProgress) -> Void)? = nil,
    onToken: ((String) -> Bool)? = nil
  ) -> GenerationResult {
    let cache = cache ?? model.text.makeCache(kvConfig: kvConfig)
    let sampler = Sampler(options: options)
    var detokenizer = StreamingDetokenizer(tokenizer: model.tokenizer)

    let promptStart = Date()
    var logits: MLXArray

    let prefillTotal = max(0, promptTokens.count - cachedPrefixLength)
    onProgress?(.prefill(done: 0, total: prefillTotal))

    if let promptEmbeddings {
      logits = model.text(
        nil, inputEmbeddings: promptEmbeddings, cache: cache, positions: positions)
      eval(logits)
      onProgress?(.prefill(done: prefillTotal, total: prefillTotal))
    } else {
      precondition(!promptTokens.isEmpty, "generate requires a non-empty prompt")
      precondition(
        cachedPrefixLength < promptTokens.count,
        "at least one prompt token must be processed to produce logits")
      var index = cachedPrefixLength
      var last: MLXArray?
      while index < promptTokens.count {
        let end = min(index + prefillChunkSize, promptTokens.count)
        let chunk = MLXArray(promptTokens[index..<end].map { Int32($0) })
          .reshaped([1, end - index])
        last = model.text(chunk, cache: cache)
        eval(last!)
        index = end
        onProgress?(.prefill(done: index - cachedPrefixLength, total: prefillTotal))
      }
      logits = last!
    }
    let promptSeconds = -promptStart.timeIntervalSinceNow

    var decodePosition: Int? = positions.map { $0.max().item(Int.self) + 1 }

    let generationStart = Date()
    var generated: [Int] = []
    var text = ""
    var stoppedOnEOS = false

    var nextLogits = logits[0..., -1, 0...]
    onProgress?(.decode(count: 0))

    for _ in 0..<maxTokens {
      let token: Int
      if let constraint {
        // A complete document may stop here; an exhausted one must.
        guard let picked = sampler(nextLogits, allowed: constraint.allowedTokens(tokenizer: model.tokenizer))
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
      let stepPositions = decodePosition.map {
        MLXArray([Int32($0), Int32($0), Int32($0)]).reshaped([3, 1])
      }
      let step = model.text(input, cache: cache, positions: stepPositions)
      eval(step)
      if decodePosition != nil { decodePosition! += 1 }
      nextLogits = step[0..., -1, 0...]
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
      stoppedOnEOS: stoppedOnEOS)
  }
}
