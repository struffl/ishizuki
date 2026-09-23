// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom

public struct SamplingOptions: Sendable {
  public var temperature: Float
  public var topP: Float
  public var topK: Int
  public var minP: Float
  public var repetitionPenalty: Float
  public var repetitionContext: Int
  /// A flat penalty subtracted from any token that has already appeared in the window,
  /// regardless of how many times — unlike `repetitionPenalty`, which scales with recurrence.
  public var presencePenalty: Float
  public var seed: UInt64?

  public init(
    temperature: Float = 0.7, topP: Float = 1.0, topK: Int = 0, minP: Float = 0.0,
    repetitionPenalty: Float = 1.0, repetitionContext: Int = 64, presencePenalty: Float = 0.0,
    seed: UInt64? = nil
  ) {
    self.temperature = temperature
    self.topP = topP
    self.topK = topK
    self.minP = min(max(minP, 0), 1)
    self.repetitionPenalty = repetitionPenalty
    self.repetitionContext = repetitionContext
    self.presencePenalty = presencePenalty
    self.seed = seed
  }

  public static let greedy = SamplingOptions(temperature: 0, topP: 1, topK: 0)
}

public struct Sampler {
  public let options: SamplingOptions

  public init(options: SamplingOptions) {
    self.options = options
    if let seed = options.seed { MLXRandom.seed(seed) }
  }

  public func callAsFunction(_ logits: MLXArray, recentTokens: [Int] = []) -> Int {
    token(logits, recentTokens: recentTokens).item(Int.self)
  }

  /// The pick left on the GPU, so the next step can be queued before it is read. `pending` is a
  /// token already sampled but not yet read back, counted as the newest entry of the window.
  public func token(
    _ logits: MLXArray, recentTokens: [Int] = [], pending: MLXArray? = nil
  ) -> MLXArray {
    let scores = truncatedScores(logits, recentTokens: recentTokens, pending: pending)

    guard options.temperature > 0 else {
      return scores.argMax(axis: -1)
    }
    return MLXRandom.categorical(scores / options.temperature, axis: -1)
  }

  /// Sample restricted to `allowed`, by gathering just those logits. The allowed set is small,
  /// so this is cheaper than masking the full vocabulary and keeps the warper chain intact.
  public func callAsFunction(_ logits: MLXArray, allowed: [Int]) -> Int? {
    guard !allowed.isEmpty else { return nil }
    let indices = MLXArray(allowed.map { Int32($0) })
    let gathered = logits.reshaped([-1])[indices].reshaped([1, allowed.count])
    let scores = truncatedScores(gathered)

    guard options.temperature > 0 else {
      return allowed[scores.argMax(axis: -1).item(Int.self)]
    }
    let choice = MLXRandom.categorical(scores / options.temperature, axis: -1).item(Int.self)
    return allowed[choice]
  }

  public func truncatedScores(
    _ logits: MLXArray, recentTokens: [Int] = [], pending: MLXArray? = nil
  ) -> MLXArray {
    var scores = logits.asType(.float32)

    if let window = window(recentTokens, pending: pending) {
      if options.repetitionPenalty != 1.0 {
        scores = applyRepetitionPenalty(scores, window: window)
      }
      if options.presencePenalty != 0 {
        scores = applyPresencePenalty(scores, window: window)
      }
    }

    guard options.temperature > 0 else { return scores }

    if options.topK > 0 { scores = applyTopK(scores, k: options.topK) }
    if options.topP > 0 && options.topP < 1 { scores = applyTopP(scores, topP: options.topP) }
    if options.minP > 0 { scores = applyMinP(scores, minP: options.minP) }

    return scores
  }

  private func window(_ tokens: [Int], pending: MLXArray?) -> MLXArray? {
    let context = options.repetitionContext
    guard context > 0 else { return nil }
    let known = tokens.suffix(pending == nil ? context : context - 1).map { Int32($0) }
    let recent = known.isEmpty ? nil : MLXArray(known)
    switch (recent, pending) {
    case (let recent?, let pending?):
      return concatenated([recent, pending.reshaped([-1]).asType(.int32)])
    case (let recent?, nil): return recent
    case (nil, let pending?): return pending.reshaped([-1]).asType(.int32)
    case (nil, nil): return nil
    }
  }

  private func applyRepetitionPenalty(_ scores: MLXArray, window: MLXArray) -> MLXArray {
    let selected = scores[0..., window]
    let penalized = MLX.where(
      selected .> 0, selected / options.repetitionPenalty,
      selected * options.repetitionPenalty)
    let updated = scores
    updated[0..., window] = penalized
    return updated
  }

  private func applyPresencePenalty(_ scores: MLXArray, window: MLXArray) -> MLXArray {
    let seen = MLXArray.zeros([scores.dim(-1)], dtype: scores.dtype)
    seen[window] = MLXArray(Float(1))
    return scores - options.presencePenalty * seen
  }

  private func applyTopK(_ scores: MLXArray, k: Int) -> MLXArray {
    let vocab = scores.dim(-1)
    guard k < vocab else { return scores }
    let threshold = MLX.sorted(scores, axis: -1)[.ellipsis, (vocab - k)...][.ellipsis, 0]
    return MLX.where(
      scores .< threshold.expandedDimensions(axis: -1),
      MLXArray(-Float.infinity), scores)
  }

  private func applyMinP(_ scores: MLXArray, minP: Float) -> MLXArray {
    let threshold = scores.max(axis: -1, keepDims: true) + logf(minP)
    return MLX.where(scores .< threshold, MLXArray(-Float.infinity), scores)
  }

  private func applyTopP(_ scores: MLXArray, topP: Float) -> MLXArray {
    let probabilities = softmax(scores, axis: -1)
    let order = MLX.argSort(probabilities, axis: -1)
    let ascending = MLX.takeAlong(probabilities, order, axis: -1)
    let cumulative = MLX.cumsum(ascending, axis: -1)
    let keep = cumulative .> (1.0 - topP)
    let maskedAscending = MLX.where(keep, ascending, MLXArray(Float(0)))

    let restore = MLX.argSort(order, axis: -1)
    let restored = MLX.takeAlong(maskedAscending, restore, axis: -1)
    return MLX.where(restored .> 0, scores, MLXArray(-Float.infinity))
  }
}
