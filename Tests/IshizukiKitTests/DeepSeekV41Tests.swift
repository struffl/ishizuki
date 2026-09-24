// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A whole DeepSeek-V4.1 model, against DeepSeek's own reference code.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// The fixture is a tiny random V4.1 written in the release's own formats — fp8 in 32x32 blocks,
/// fp4 experts, fp8 engram rows — by `Scripts/make_deepseek_v41_fixture.py`, which ran it through
/// DeepSeek's `inference/model.py` with its GPU kernels replaced by plain torch. It keeps every
/// structural feature of the real config at a smaller size: window-only layers, ratio-2 encoder
/// sources and reusers, a ratio-1 decoder source that is also the candidate source, a reindexer,
/// two engram layers, four residual copies.
@Suite("DeepSeek-V4.1")
struct DeepSeekV41Tests {
  private var fixture: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appending(path: "Fixtures/deepseek-v41")
  }

  private func reference() throws -> [String: MLXArray] {
    try loadArrays(url: fixture.appending(path: "reference.safetensors"))
  }

  private func model(
    dense: Bool = true, fakeQuant: Bool = true, slots: Int? = nil
  ) throws -> DeepSeekModel {
    let weights = DeepSeekWeights(
      checkpoint: try DeepSeekCheckpoint(directory: fixture), dense: dense, compute: .float32,
      expertSlots: slots)
    let map = try #require(try reference()["token_map"]).asArray(Int32.self)
    let model = try DeepSeekModel(weights: weights, tokenMap: map)
    model.fakeQuant = fakeQuant
    return model
  }

  private func tokens() throws -> MLXArray {
    try #require(try reference()["tokens"]).asType(.int32)
  }

  private func drift(_ got: MLXArray, _ want: MLXArray) -> Float {
    let worst = (got.asType(.float32) - want).abs().max().item(Float.self)
    return worst / max(want.abs().max().item(Float.self), 1e-12)
  }

  @Test("reads the architecture out of the release config")
  func readsTheConfig() throws {
    let config = try DeepSeekConfig.load(directory: fixture)
    #expect(config.layers == 10)
    #expect(config.compressRatio(layer: 1) == 0)
    #expect(config.compressRatio(layer: 3) == 2)
    #expect(config.kvSource(of: 3) == 2)
    #expect(config.kvSource(of: 9) == 6)
    #expect(config.indexSource(of: 9) == 8)
    #expect(config.experts(layer: 11) == (routed: 4, activated: 2))
  }

  @Test("draws numpy's multipliers and DeepSeek's primes")
  func hashesLikeTheReference() throws {
    let reference = try reference()
    let model = try model()
    let hasher = try #require(model.hasher)
    let multipliers = try #require(reference["multipliers"]).asArray(Int64.self)
    #expect(hasher.multipliers.flatMap { $0 } == multipliers)
    let primes = try #require(reference["primes"]).asArray(Int64.self)
    #expect(hasher.primes.flatMap { $0 } == primes)
    let offsets = try #require(reference["offsets"]).asArray(Int64.self)
    #expect(hasher.offsets.flatMap { $0 } == offsets)

    let ids = try tokens()[0, ..<20].asArray(Int32.self)
    let hashes = hasher.hashes(hasher.compress(ids), history: [])
    let want = try #require(reference["hashes"])
    for layer in 0..<2 {
      let expected = want[0, 0..., layer, 0...].asType(.int64).asArray(Int64.self)
      #expect(hashes[layer] == expected, "engram layer \(layer) hashed differently")
    }

    var history: [Int32] = []
    var stepped: [[Int64]] = [[], []]
    for id in ids {
      let compressed = hasher.compress([id])
      let rows = hasher.hashes(compressed, history: history)
      history = hasher.history(after: history + compressed)
      for layer in 0..<2 { stepped[layer] += rows[layer] }
    }
    #expect(stepped == hashes, "hashing a token at a time reached different rows")
  }

  /// numpy's draw for the release's own engram layers, over its 99,092 compressed ids, and the
  /// primes whose sums are the two table sizes its config declares.
  @Test("draws the release's multipliers and table primes")
  func matchesTheRelease() {
    let multipliers = DeepSeekNgramHasher.multipliers(layers: [1, 14], ngram: 4, vocab: 99_092)
    #expect(
      multipliers == [
        [76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391],
        [67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591],
      ])
    let primes = DeepSeekNgramHasher.primes(from: 16_000_000, count: 48)
    #expect(Array(primes[0..<3]) == [16_000_057, 16_000_079, 16_000_081])
    #expect(primes[..<24].reduce(0, +) == 384_006_168)
    #expect(primes[24...].reduce(0, +) == 384_016_682)
  }

  @Test("carries every layer's streams where the reference carries them")
  func matchesEveryLayer() throws {
    let reference = try reference()
    let model = try model()
    let prompt = try tokens()[0..., ..<20]
    var worst: [(Int, Float)] = []
    let hidden = model.hidden(prompt, cache: model.makeCache()) { layer, streams in
      if let want = reference["stage_\(layer)"] { worst.append((layer, drift(streams, want))) }
    }
    eval(hidden)
    for (layer, value) in worst {
      #expect(value < 1e-4, "layer \(layer) drifted by \(value)")
    }
    let logits = model.logits(hidden)
    let want = try #require(reference["prefill_logits"])
    #expect(drift(logits, want) < 1e-4, "prefill logits drifted by \(drift(logits, want))")
  }

  @Test("decodes each token to where the reference's decode step puts it")
  func matchesDecode() throws {
    let reference = try reference()
    let model = try model()
    let all = try tokens()
    let cache = model.makeCache()
    eval(model(all[0..., ..<20], cache: cache))
    let want = try #require(reference["decode_logits"])
    for t in 20..<32 {
      let step = model(all[0..., t..<(t + 1)], cache: cache)
      let value = drift(step[0, 0], want[t - 20])
      #expect(value < 1e-4, "decode step at \(t) drifted by \(value)")
    }
  }

  @Test("reaches the same logits in one prefill, in chunks, and a token at a time")
  func chunksAgree() throws {
    let reference = try reference()
    let model = try model()
    let all = try tokens()
    let want = try #require(reference["whole_logits"])

    let whole = model(all, cache: model.makeCache())
    #expect(drift(whole, want) < 1e-4, "one prefill drifted by \(drift(whole, want))")

    let cache = model.makeCache()
    var pieces: [MLXArray] = []
    var start = 0
    for length in [7, 5, 1, 8, 11] {
      pieces.append(model(all[0..., start..<(start + length)], cache: cache))
      start += length
    }
    let chunked = concatenated(pieces, axis: 1)
    #expect(drift(chunked, want) < 1e-4, "chunked prefill drifted by \(drift(chunked, want))")
  }

  /// A prefill chunk is attended and indexed a few queries at a time; five at once, against a
  /// window of eight, cuts every block boundary through somebody's window.
  @Test("attends a chunk in blocks of queries to the same answer as all at once")
  func blocksQueries() throws {
    let reference = try reference()
    let model = try model()
    let all = try tokens()
    let saved = DeepSeekAttention.queryBlock
    DeepSeekAttention.queryBlock = 5
    defer { DeepSeekAttention.queryBlock = saved }
    let whole = model(all, cache: model.makeCache())
    let want = try #require(reference["whole_logits"])
    #expect(drift(whole, want) < 1e-4, "blocked prefill drifted by \(drift(whole, want))")
  }

  @Test("matches the continuous arithmetic when the cache rounding is off")
  func matchesContinuous() throws {
    let reference = try reference()
    let model = try model(fakeQuant: false)
    let all = try tokens()
    let whole = model(all, cache: model.makeCache())
    let want = try #require(reference["continuous_whole_logits"])
    #expect(drift(whole, want) < 1e-4, "continuous logits drifted by \(drift(whole, want))")
    // The rounding is worth several logits here, so matching both runs pins it exactly.
    let rounded = try #require(reference["whole_logits"])
    #expect(drift(rounded, want) > 0.1)
  }

  @Test("multiplies the release's fp8 and fp4 where they lie as the dense weights would")
  func nativeFormatsMatchDense() throws {
    let all = try tokens()
    let dense = try model()
    let native = try model(dense: false)
    let a = dense(all, cache: dense.makeCache())
    let b = native(all, cache: native.makeCache())
    #expect(drift(b, a) < 1e-4, "the mx formats drifted from dense by \(drift(b, a))")
  }

  /// Three slots against eight experts routed two at a time: the prefill chunk runs expert by
  /// expert through them, and every decode step evicts.
  @Test("streams the experts out of the shards to the same answer as holding them")
  func streamsExperts() throws {
    let all = try tokens()
    let held = try model(dense: false)
    let streamed = try model(dense: false, slots: 3)
    let want = held(all, cache: held.makeCache())
    let cache = streamed.makeCache()
    let prefill = streamed(all[0..., ..<20], cache: cache)
    #expect(drift(prefill, want[0..., ..<20]) < 1e-5, "streamed prefill differs")
    for t in 20..<32 {
      let step = streamed(all[0..., t..<(t + 1)], cache: cache)
      #expect(drift(step[0, 0], want[0, t]) < 1e-5, "streamed decode at \(t) differs")
    }
  }

  /// The draft head reads the backbone's last three layers' inputs, keeps a window of its own
  /// made from them, and proposes five tokens after the one the backbone chose.
  /// With the cache rounding on, one window key the two implementations round a hair apart is
  /// visible in the first decode step's drafts and gone once it leaves the window; with it off
  /// the arithmetic agrees everywhere.
  /// A release's n-gram tables are far too large to hold, so they are read a row at a time out
  /// of the shard; the fixture's are small enough to hold, which is the answer to check against.
  @Test("reads n-gram rows out of the shard as holding the tables would")
  func readsEngramRows() throws {
    let held = try model()
    let checkpoint = try DeepSeekCheckpoint(directory: fixture)
    let tables = try held.config.engramLayers.map {
      try CheckpointEngramTable(checkpoint: checkpoint, prefix: "layers.\($0).engram.embed")
    }
    let read = try DeepSeekModel(
      weights: DeepSeekWeights(checkpoint: checkpoint, dense: true, compute: .float32),
      tokenMap: try #require(try reference()["token_map"]).asArray(Int32.self), tables: tables)
    let all = try tokens()
    let want = held(all[0..., ..<20], cache: held.makeCache())
    let got = read(all[0..., ..<20], cache: read.makeCache())
    #expect(drift(got, want) < 1e-6, "rows read from the shard differ from the held table's")
  }

  @Test("drafts the tokens, logits and odds DSpark does", arguments: [true, false])
  func matchesTheDraftHead(rounded: Bool) throws {
    let reference = try reference()
    let model = try model(fakeQuant: rounded)
    let label = rounded ? "" : "continuous_"
    let tolerance: Float = rounded ? 2e-3 : 1e-4
    let draft = try #require(model.draft)
    let all = try tokens()
    let cache = model.makeCache()
    let caches = draft.makeCaches()
    let prefill = model.forward(all[0..., ..<20], cache: cache)
    draft.observe(try #require(prefill.draftHidden), caches: caches, start: 0)

    let ids = try #require(reference[label + "dspark_ids"])
    let logits = try #require(reference[label + "dspark_logits"])
    let odds = try #require(reference[label + "dspark_confidence"])
    for t in 20..<32 {
      let step = model.forward(all[0..., t..<(t + 1)], cache: cache)
      let next = argMax(model.lastLogits(model.normed(step.trunk))[0, -1]).item(Int.self)
      draft.observe(try #require(step.draftHidden), caches: caches, start: t)
      let proposal = draft.propose(after: next, at: t + 1, caches: caches, model: model) {
        argMax($0, axis: -1).item(Int.self)
      }
      #expect(proposal.tokens == ids[t - 20].asArray(Int32.self).map(Int.init), "drafts at \(t)")
      #expect(drift(proposal.logits[0], logits[t - 20]) < tolerance, "draft logits at \(t)")
      #expect(drift(proposal.confidence[0], odds[t - 20]) < tolerance, "draft odds at \(t)")
    }
  }

  /// What the prefix store writes to disk and reads back: the windows, the owners' latents and
  /// index keys, a ratio-2 owner's half-finished group, and the n-gram look-back.
  @Test("carries a prefix through export and load")
  func archives() throws {
    let model = try model()
    let all = try tokens()
    let cache = model.makeCache()
    eval(model(all[0..., ..<21], cache: cache))
    let fresh = model.makeCache()
    for (target, source) in zip(fresh.layers, cache.layers) {
      let exported = try #require(source.export())
      #expect(target.load(exported, offset: source.offset))
    }
    let kept = model(all[0..., 21..<25], cache: cache)
    let restored = model(all[0..., 21..<25], cache: fresh)
    #expect(drift(restored, kept) < 1e-6, "a reloaded prefix decodes differently")
  }

  /// A 6x9 patch picture becomes a 2x3 grid, and a ten-position span: start, three images and a
  /// newline twice, end. Its positions route with the image bias and take no part in any n-gram.
  @Test("reads a picture the way DeepSeek-ViT does, and answers around it")
  func matchesVision() throws {
    let reference = try reference()
    let weights = DeepSeekWeights(
      checkpoint: try DeepSeekCheckpoint(directory: fixture), dense: true, compute: .float32)
    let vision = try DeepSeekVision(weights: weights)
    let image = DeepSeekVision.patchify(try #require(reference["vision_pixels"]), patchSize: 14)
    #expect(image.spanLength == 10)
    let features = vision.encode(image, dtype: .float32)
    let want = try #require(reference["vision_features"])
    #expect(drift(features, want) < 1e-4, "aligner rows drifted by \(drift(features, want))")

    let model = try model()
    let ids = try #require(reference["vision_ids"]).asType(.int32)
    let types = try #require(reference["vision_types"]).asArray(Int32.self)
    let start = try #require(types.firstIndex { $0 >= 0 })
    var embeddings = model.embed(ids)
    embeddings[0..., start..<(start + image.spanLength), 0...] =
      vision.span(image, dtype: .float32).expandedDimensions(axis: 0)
    let hidden = model.normed(
      model.forward(ids, cache: model.makeCache(), embeddings: embeddings).trunk)
    let logits = model.logits(hidden)
    let wanted = try #require(reference["vision_logits"])
    #expect(drift(logits, wanted) < 1e-4, "the picture's prompt drifted by \(drift(logits, wanted))")

    let hasher = try #require(model.hasher)
    let dead = types.map { $0 >= 0 }
    let hashes = hasher.hashes(hasher.compress(ids.asArray(Int32.self), dead: dead), history: [])
    let expected = try #require(reference["vision_hashes"])
    for layer in 0..<2 {
      #expect(hashes[layer] == expected[0, 0..., layer, 0...].asType(.int64).asArray(Int64.self))
    }
  }

  /// `plan_image_grid` from the release, at its own settings: 544x544 minimum, 1,024 tokens.
  @Test("sizes a picture the way the release's image processor does")
  func plansPictures() {
    let cases: [((Int, Int), (Int, Int))] = [
      ((1024, 768), (1036, 770)), ((3000, 200), (3010, 210)), ((100, 100), (546, 546)),
      ((1344, 1344), (1302, 1302)), ((4032, 3024), (1512, 1134)), ((640, 480), (644, 490)),
      ((200, 3000), (210, 3010)), ((1920, 1080), (1708, 966)), ((50, 2000), (98, 3444)),
    ]
    for ((width, height), want) in cases {
      let got = DeepSeekVision.plan(
        width: width, height: height, patchSize: 14, ratio: 3, minPixels: 295_936,
        maxTokens: 1024)
      #expect(got == want, "\(width)x\(height) planned as \(got)")
    }
  }

  /// The generator's drafted loop with DSpark proposing: verification makes it exact, so greedy
  /// decoding has to come out token for token what it is without drafts.
  @Test("drafts with DSpark to exactly what plain greedy decoding says")
  func draftsExactly() throws {
    let compute = BonsaiRuntime.deepseekCompute
    let streams = BonsaiRuntime.deepseekStreamsExperts
    BonsaiRuntime.deepseekCompute = .float32
    BonsaiRuntime.deepseekStreamsExperts = false
    defer {
      BonsaiRuntime.deepseekCompute = compute
      BonsaiRuntime.deepseekStreamsExperts = streams
    }
    let model = try BonsaiModel(directory: fixture)
    let prompt = try tokens()[0, ..<20].asArray(Int32.self).map(Int.init)
    let plain = Generator(model: model)
    plain.speculativeDecode = false
    let drafted = Generator(model: model)
    drafted.speculativeDecode = true
    let greedy = SamplingOptions(temperature: 0)
    let a = plain.generate(promptTokens: prompt, options: greedy, maxTokens: 24)
    let b = drafted.generate(promptTokens: prompt, options: greedy, maxTokens: 24)
    #expect(a.tokens == b.tokens, "drafting changed what greedy decoding says")
    #expect((b.speculative?.proposed ?? 0) > 0, "DSpark never proposed")
  }

  /// Bounded replay built the long way round: the whole stack over the head of the prompt, the
  /// windows above the last global KV source dropped, then the tail. Encoding the head instead
  /// has to leave every layer holding the same, and the tail reading the same logits — the
  /// layers it skips owe the tail nothing but those windows. The picture puts its span in the
  /// encoded head, where its positions still have to be told apart from the words.
  @Test(
    "replays a prompt's tail as dropping the decoder's windows would",
    arguments: [false, true])
  func replaysTheTail(picture: Bool) throws {
    let model = try model()
    var all = try tokens()
    var embeddings: MLXArray?
    if picture {
      let reference = try reference()
      let vision = try DeepSeekVision(
        weights: DeepSeekWeights(
          checkpoint: try DeepSeekCheckpoint(directory: fixture), dense: true, compute: .float32))
      let image = DeepSeekVision.patchify(try #require(reference["vision_pixels"]), patchSize: 14)
      all = try #require(reference["vision_ids"]).asType(.int32)
      let types = try #require(reference["vision_types"]).asArray(Int32.self)
      let start = try #require(types.firstIndex { $0 >= 0 })
      let rows = model.embed(all)
      rows[0..., start..<(start + image.spanLength), 0...] =
        vision.span(image, dtype: .float32).expandedDimensions(axis: 0)
      embeddings = rows
    }
    let n = all.dim(1)
    let decoder = try #require(model.decoderStart)
    let split = n - model.config.window
    func rows(_ range: Range<Int>) -> MLXArray? { embeddings?[0..., range, 0...] }
    func logits(_ range: Range<Int>, _ cache: ModelCache) -> MLXArray {
      model.logits(
        model.normed(model.forward(all[0..., range], cache: cache, embeddings: rows(range)).trunk))
    }

    let long = model.makeCache()
    eval(logits(0..<split, long))
    for layer in decoder..<model.config.layers {
      (long.layers[layer] as! DeepSeekLayerCache).window = nil
    }
    let want = logits(split..<n, long)

    let short = model.makeCache()
    model.encode(all[0..., ..<11], cache: short, embeddings: rows(0..<11))
    model.encode(all[0..., 11..<split], cache: short, embeddings: rows(11..<split))
    let got = logits(split..<n, short)
    #expect(drift(got, want) < 1e-6, "the replayed tail reads different logits")

    for layer in 0..<model.config.layers {
      let a = short.layers[layer] as! DeepSeekLayerCache
      let b = long.layers[layer] as! DeepSeekLayerCache
      #expect(a.offset == b.offset && a.compressedCount == b.compressedCount, "layer \(layer)")
      for (x, y) in [(a.latents, b.latents), (a.keys, b.keys), (a.window, b.window)] {
        #expect((x == nil) == (y == nil), "layer \(layer) holds different state")
        if let x, let y { #expect(drift(x, y) < 1e-6, "layer \(layer) holds different state") }
      }
    }
  }

  /// The generator encodes all but a prompt's last window and runs the whole stack over that,
  /// and a prompt no longer than the window it runs exactly.
  @Test("prefills a long prompt by replaying its tail")
  func generatorReplays() throws {
    let bonsai = try BonsaiModel(directory: fixture)
    let model = try #require(bonsai.deepseek)
    let all = try tokens()
    let window = model.config.window
    let greedy = SamplingOptions(temperature: 0)

    func byHand(_ prompt: Int, replay: Bool) -> [Int] {
      let cache = model.makeCache()
      let split = replay ? max(prompt - window, 0) : 0
      if split > 0 { model.encode(all[0..., ..<split], cache: cache) }
      var logits = model.lastLogits(
        model.normed(model.trunk(all[0..., split..<prompt], cache: cache)))
      var out: [Int] = []
      for _ in 0..<6 {
        let next = argMax(logits[0, -1], axis: -1).item(Int.self)
        out.append(next)
        logits = model(MLXArray([Int32(next)]).reshaped([1, 1]), cache: cache)
      }
      return out
    }

    for (prompt, replay) in [(20, true), (20, false), (window, true)] {
      model.boundedReplay = replay
      let generator = Generator(model: bonsai)
      generator.speculativeDecode = false
      let tokens = all[0, ..<prompt].asArray(Int32.self).map(Int.init)
      let result = generator.generate(promptTokens: tokens, options: greedy, maxTokens: 6)
      #expect(
        result.tokens == byHand(prompt, replay: replay),
        "\(prompt) tokens, replay \(replay): the generator prefilled differently")
    }
    model.boundedReplay = true
    #expect(byHand(window, replay: true) == byHand(window, replay: false))
  }

  /// A served turn goes through the chat format, the session cache and its rewind point, none of
  /// which the model's own tests reach; the next turn has to pick the first one's cache back up.
  @Test("serves a turn, and the next one picks its cache back up")
  func serves() throws {
    let server = try APIServer(directory: fixture, preload: false)
    func ask(_ messages: [ChatMessage]) throws -> (text: String, tokens: Int, reused: Int) {
      let request = APIServer.Request(
        messages: messages, tools: nil, maxTokens: 6, temperature: 0, stream: false,
        thinking: false, images: [], responseSchema: nil, model: nil, effort: nil)
      let out = try server.complete(request)
      return (out.parsed.content, out.completionTokens, out.reused)
    }
    let first = try ask([.user("hello there")])
    #expect(first.tokens > 0)
    let second = try ask([.user("hello there"), .assistant(first.text), .user("and again")])
    #expect(second.tokens > 0)
    #expect(second.reused > 0, "the second turn prefilled from scratch")
  }

  @Test("rewinds to a snapshot and decodes the same again")
  func rewinds() throws {
    let model = try model()
    let all = try tokens()
    let cache = model.makeCache()
    eval(model(all[0..., ..<21], cache: cache))
    let saved = cache.snapshot()
    var first: [MLXArray] = []
    for t in 21..<25 { first.append(model(all[0..., t..<(t + 1)], cache: cache)) }
    cache.restore(saved)
    for (index, t) in (21..<25).enumerated() {
      let again = model(all[0..., t..<(t + 1)], cache: cache)
      #expect(drift(again, first[index]) < 1e-6, "step \(t) after a rewind differs")
    }
  }
}
