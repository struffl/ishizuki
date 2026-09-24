// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DeepSeek-V4.1's own draft head, proposing for the generator's drafted loop.

import Foundation
import MLX

/// DSpark drafting for a served turn.
///
/// The draft head keeps a window of the backbone's own positions. A forward hands back what the
/// window needs, and only positions the verify loop keeps are committed to it — each once, in
/// order, however often the loop runs them again — so a rejected draft never reaches the window
/// and nothing has to be taken back out. A proposal is the block after the token just confirmed,
/// cut where the confidence head stops giving the drafts together better than even odds.
///
/// Only worth it with the experts in memory: a verify of six tokens reads the union of six
/// tokens' experts, which off a disk costs more than the tokens it saves.
final class DSparkDrafter {
  let model: DeepSeekModel
  let draft: DeepSeekDraft
  private let caches: [DeepSeekLayerCache]
  private(set) var observed = 0

  init?(model: DeepSeekModel) {
    guard let draft = model.draft else { return nil }
    self.model = model
    self.draft = draft
    self.caches = draft.makeCaches()
  }

  /// The backbone's forward over `tokens`, and what the draft windows read at each position.
  func forward(_ tokens: MLXArray, cache: ModelCache) -> (
    trunk: MLXArray, hidden: MLXArray?, start: Int
  ) {
    let start = cache.offset
    let step = model.forward(tokens, cache: cache)
    return (step.trunk, step.draftHidden, start)
  }

  /// Hands the first `count` positions of a forward to the windows, skipping any already there.
  func commit(_ hidden: MLXArray?, start: Int, count: Int) {
    guard let hidden else { return }
    let skip = max(0, observed - start)
    guard skip < count else { return }
    draft.observe(hidden[0..., skip..<count], caches: caches, start: start + skip)
    observed = start + count
  }

  /// Drafts after `token`, which sits at the first position the backbone has not run.
  func propose(after token: Int, at position: Int, limit: Int) -> [Int] {
    guard observed == position, limit > 0 else { return [] }
    let proposal = draft.propose(after: token, at: position, caches: caches, model: model) {
      argMax($0, axis: -1).item(Int.self)
    }
    let odds = sigmoid(proposal.confidence[0]).asArray(Float.self)
    var survival: Float = 1
    var kept: [Int] = []
    for (index, draft) in proposal.tokens.dropFirst().enumerated() where kept.count < limit {
      survival *= odds[index]
      guard survival >= 0.5 else { break }
      kept.append(draft)
    }
    return kept
  }
}
