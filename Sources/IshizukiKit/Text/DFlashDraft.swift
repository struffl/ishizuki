// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DFlash, z-lab's block-diffusion drafter: a whole block of drafts from the backbone's own taps.

import Foundation
import MLX
import MLXFast
import MLXNN

/// A drafter that proposes a block at once rather than a token at a time.
///
/// It reads the backbone's residual stream after a handful of its layers, projected into one
/// width, as the context its attention layers look back over. The block is the token just
/// confirmed and mask tokens after it, attended both ways, so every draft is made in the same
/// forward pass. DFlash 2 adds a two-tap convolution along the block around each sub-layer, and
/// keeps each position's best candidates for a small selector that walks one path through them,
/// each pick conditioning the next. The embedding and the head are the backbone's own.
public final class DFlashDraft: @unchecked Sendable {
  public struct Config: Sendable {
    public let hidden: Int
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let eps: Float
    public let ropeTheta: Float
    public let window: Int?
    public let blockSize: Int
    public let maskToken: Int
    public let targetLayers: [Int]
    public let convKernel: Int
    public let convGroup: Int
    public let selectorTopK: Int

    init(json: [String: Any]) throws {
      let dflash = json["dflash_config"] as? [String: Any] ?? [:]
      func int(_ value: Any?) -> Int? { (value as? NSNumber)?.intValue }
      guard let hidden = int(json["hidden_size"]), let heads = int(json["num_attention_heads"]),
        let kvHeads = int(json["num_key_value_heads"]), let headDim = int(json["head_dim"]),
        let layers = dflash["target_layer_ids"] as? [NSNumber],
        let mask = int(dflash["mask_token_id"])
      else {
        throw BonsaiError.unsupportedModel("not a DFlash draft config")
      }
      let types = json["layer_types"] as? [String] ?? []
      self.hidden = hidden
      self.heads = heads
      self.kvHeads = kvHeads
      self.headDim = headDim
      self.eps = (json["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
      let rope = json["rope_parameters"] as? [String: Any] ?? json["rope_scaling"] as? [String: Any]
      self.ropeTheta =
        (rope?["rope_theta"] as? NSNumber)?.floatValue
        ?? (json["rope_theta"] as? NSNumber)?.floatValue ?? 10000
      self.window =
        !types.isEmpty && types.allSatisfy({ $0 == "sliding_attention" })
        ? int(json["sliding_window"]) : nil
      self.blockSize = int(dflash["block_size"]) ?? int(json["block_size"]) ?? 16
      self.maskToken = mask
      self.targetLayers = layers.map(\.intValue)
      self.convKernel = int(dflash["conv_kernel_size"]) ?? 0
      self.convGroup = int(dflash["conv_group_size"]) ?? 0
      self.selectorTopK = int(dflash["selector_top_k"]) ?? 0
    }
  }

  /// One attention layer's context: every projected tap it has been handed, up to its window.
  public final class LayerCache: @unchecked Sendable {
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0

    public init() {}
  }

  /// A projection held quantized, applied to float32 rows: a verify-width block goes through
  /// the few-row kernel, which decodes each weight once for every row, and a narrower one
  /// through MLX's own. Nothing is multiplied in bf16, which an M1 does not have.
  struct Matrix {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let groupSize: Int
    let bits: Int

    init(_ dense: MLXArray, groupSize: Int, bits: Int) {
      let (weight, scales, biases) = quantized(
        dense.asType(.float32), groupSize: groupSize, bits: bits)
      self.weight = weight
      self.scales = scales
      self.biases = biases ?? MLXArray.zeros(like: scales)
      self.groupSize = groupSize
      self.bits = bits
      eval(self.weight, self.scales, self.biases)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
      let width = x.dim(-1)
      let rows = x.size / width
      if VerifyMatmul.supportedRows.contains(rows),
        let y = VerifyMatmul.apply(
          x.reshaped([rows, width]), weight, scales: scales, biases: biases,
          groupSize: groupSize, bits: bits)
      {
        return y.reshaped(Array(x.shape.dropLast()) + [y.dim(-1)])
      }
      return quantizedMM(
        x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize,
        bits: bits, mode: .affine)
    }
  }

  struct Attention {
    let q: Matrix
    let k: Matrix
    let v: Matrix
    let o: Matrix
    let qNorm: MLXArray
    let kNorm: MLXArray
    let config: Config

    func callAsFunction(_ x: MLXArray, context: MLXArray, cache: LayerCache) -> MLXArray {
      let b = x.dim(0)
      let l = x.dim(1)
      let heads = config.heads
      let kvHeads = config.kvHeads
      let d = config.headDim
      var context = context
      if let window = config.window, context.dim(1) > window - 1 {
        let skip = context.dim(1) - (window - 1)
        context = context[0..., skip...]
        cache.offset += skip
      }
      let s = context.dim(1)
      let start = cache.offset

      func split(_ y: MLXArray, _ count: Int, norm: MLXArray?) -> MLXArray {
        var shaped = y.reshaped([b, y.dim(1), count, d])
        if let norm { shaped = MLXFast.rmsNorm(shaped, weight: norm, eps: config.eps) }
        return shaped.transposed(0, 2, 1, 3)
      }
      func rope(_ y: MLXArray, _ offset: Int) -> MLXArray {
        MLXFast.RoPE(
          y, dimensions: d, traditional: false, base: config.ropeTheta, scale: 1, offset: offset)
      }

      let queries = rope(split(q(x), heads, norm: qNorm), start + s)
      let blockKeys = rope(split(k(x), kvHeads, norm: kNorm), start + s)
      let blockValues = split(v(x), kvHeads, norm: nil)
      if s > 0 {
        let contextKeys = rope(split(k(context), kvHeads, norm: kNorm), start)
        let contextValues = split(v(context), kvHeads, norm: nil)
        cache.keys = cache.keys.map { concatenated([$0, contextKeys], axis: 2) } ?? contextKeys
        cache.values =
          cache.values.map { concatenated([$0, contextValues], axis: 2) } ?? contextValues
        if let window = config.window, let held = cache.keys, held.dim(2) > window - 1 {
          let from = held.dim(2) - (window - 1)
          cache.keys = held[0..., 0..., from..., 0...]
          cache.values = cache.values![0..., 0..., from..., 0...]
        }
        cache.offset += s
      }
      let held = cache.keys?.dim(2) ?? 0
      let keys = cache.keys.map { concatenated([$0, blockKeys], axis: 2) } ?? blockKeys
      let values = cache.values.map { concatenated([$0, blockValues], axis: 2) } ?? blockValues

      var mask: MLXArray?
      if let window = config.window, held + l > window {
        let query = MLXArray(Int32(held)..<Int32(held + l)).reshaped([l, 1])
        let key = MLXArray(Int32(0)..<Int32(held + l)).reshaped([1, held + l])
        mask = (key .>= MLXArray(Int32(held))) .|| ((query - key) .< MLXArray(Int32(window)))
      }
      let out = MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: 1 / Float(d).squareRoot(),
        mask: mask)
      return o(out.transposed(0, 2, 1, 3).reshaped([b, l, heads * d]))
    }
  }

  /// DFlash 2's convolution along the block: two taps, each a fixed kernel plus one the
  /// position predicts for itself, per group of channels. It never reaches past the block's
  /// first position into the context.
  struct Conv {
    let base: MLXArray
    let projection: Matrix
    let kernel: Int
    let group: Int

    func prepare(_ x: MLXArray) -> (MLXArray, MLXArray) {
      let groups = x.dim(-1) / group
      let dynamic = projection(x).reshaped([x.dim(0), x.dim(1), 2, kernel, groups])
      return (
        convolve(x, dynamic[0..., 0..., 0], base: base[0]), dynamic[0..., 0..., 1]
      )
    }

    func finish(_ x: MLXArray, _ dynamic: MLXArray) -> MLXArray {
      convolve(x, dynamic, base: base[1])
    }

    private func convolve(_ x: MLXArray, _ dynamic: MLXArray, base: MLXArray) -> MLXArray {
      let b = x.dim(0)
      let l = x.dim(1)
      let groups = x.dim(-1) / group
      let blocks = x.reshaped([b, l, groups, group])
      let weights = dynamic.reshaped([b, l, kernel, groups, 1])
      var out = MLXArray.zeros(like: blocks)
      for offset in 0..<kernel {
        let shifted =
          offset == 0
          ? blocks
          : concatenated(
            [MLXArray.zeros([b, offset, groups, group], dtype: blocks.dtype), blocks[0..., ..<(l - offset)]],
            axis: 1)
        let fixed = base[offset].reshaped([1, 1, groups, group]).asType(x.dtype)
        out = out + fixed * shifted + weights[0..., 0..., offset] * shifted
      }
      return out.reshaped(x.shape)
    }
  }

  struct Layer {
    let attention: Attention
    let gate: Matrix
    let up: Matrix
    let down: Matrix
    let inputNorm: MLXArray
    let postNorm: MLXArray
    let attentionConv: Conv?
    let feedConv: Conv?
    let eps: Float

    func callAsFunction(_ x: MLXArray, context: MLXArray, cache: LayerCache) -> MLXArray {
      func feed(_ y: MLXArray) -> MLXArray { down(silu(gate(y)) * up(y)) }
      let normed = MLXFast.rmsNorm(x, weight: inputNorm, eps: eps)
      let attended: MLXArray
      if let attentionConv {
        let (mixed, kernel) = attentionConv.prepare(normed)
        attended = attentionConv.finish(attention(mixed, context: context, cache: cache), kernel)
      } else {
        attended = attention(normed, context: context, cache: cache)
      }
      let h = x + attended
      let second = MLXFast.rmsNorm(h, weight: postNorm, eps: eps)
      let fed: MLXArray
      if let feedConv {
        let (mixed, kernel) = feedConv.prepare(second)
        fed = feedConv.finish(feed(mixed), kernel)
      } else {
        fed = feed(second)
      }
      return h + fed
    }
  }

  /// DFlash 2's selector: a low-rank edge score between each candidate and the pick before it,
  /// added to the candidate's own logit.
  struct Selector {
    let hiddenProjection: Matrix
    let predecessors: MLXArray
    let successors: MLXArray
    let topK: Int

    /// The path through each position's candidates, greedily, starting after `anchor`.
    func select(_ hidden: MLXArray, logits: MLXArray, anchor: MLXArray) -> MLXArray {
      let vocab = logits.dim(-1)
      let candidates = argPartition(logits, kth: vocab - topK, axis: -1)[.ellipsis, (vocab - topK)...]
      let unary = takeAlong(logits, candidates, axis: -1).asType(.float32)
      let projected = hiddenProjection(hidden)
      var before = anchor
      var path: [MLXArray] = []
      for position in 0..<hidden.dim(1) {
        let from = take(predecessors, before, axis: 0).asType(.float32)
        let to = take(successors, candidates[0..., position], axis: 0).asType(.float32)
        let edges = ((from * projected[0..., position]).expandedDimensions(axis: 1) * to).sum(
          axis: -1)
        let chosen = argMax(unary[0..., position] + edges, axis: -1)
        before = takeAlong(
          candidates[0..., position], chosen.expandedDimensions(axis: -1), axis: -1
        ).squeezed(axis: -1)
        path.append(before)
      }
      return stacked(path, axis: 1)
    }
  }

  public let config: Config
  let fc: Matrix
  let hiddenNorm: MLXArray
  let layers: [Layer]
  let norm: MLXArray
  let selector: Selector?

  /// Reads a drafter as z-lab ships it: `config.json` and its bf16 safetensors. The projections
  /// are held at `bits`, eight by default, which is as close to bf16 as the draft needs to be;
  /// everything runs in float32 around them. Float16 would not do: the residual grows past what
  /// it holds, to over a hundred thousand on Qwen3.8's taps.
  public init(directory: URL, bits: Int = 8, groupSize: Int = 128) throws {
    let data = try Data(contentsOf: directory.appending(path: "config.json"))
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw BonsaiError.unsupportedModel("unreadable DFlash config")
    }
    let config = try Config(json: json)
    self.config = config
    var weights: [String: MLXArray] = [:]
    let files = try FileManager.default.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "safetensors" }
    for file in files {
      for (name, array) in try loadArrays(url: file) { weights[name] = array }
    }
    func weight(_ name: String) throws -> MLXArray {
      guard let array = weights.removeValue(forKey: name) else {
        throw BonsaiError.missingWeight("\(name) is not in the DFlash draft")
      }
      let held = array.asType(name.hasSuffix("codebook") ? .float16 : .float32)
      eval(held)
      return held
    }
    func matrix(_ name: String) throws -> Matrix {
      guard let array = weights.removeValue(forKey: name) else {
        throw BonsaiError.missingWeight("\(name) is not in the DFlash draft")
      }
      return Matrix(array, groupSize: groupSize, bits: bits)
    }
    func conv(_ prefix: String) throws -> Conv? {
      guard config.convKernel > 0 else { return nil }
      return Conv(
        base: try weight(prefix + ".base_kernel"),
        projection: try matrix(prefix + ".kernel_projection.weight"), kernel: config.convKernel,
        group: config.convGroup)
    }

    self.fc = try matrix("fc.weight")
    self.hiddenNorm = try weight("hidden_norm.weight")
    self.norm = try weight("norm.weight")
    let count = (json["num_hidden_layers"] as? NSNumber)?.intValue ?? 0
    self.layers = try (0..<count).map { index in
      let prefix = "layers.\(index)"
      return Layer(
        attention: Attention(
          q: try matrix(prefix + ".self_attn.q_proj.weight"),
          k: try matrix(prefix + ".self_attn.k_proj.weight"),
          v: try matrix(prefix + ".self_attn.v_proj.weight"),
          o: try matrix(prefix + ".self_attn.o_proj.weight"),
          qNorm: try weight(prefix + ".self_attn.q_norm.weight"),
          kNorm: try weight(prefix + ".self_attn.k_norm.weight"), config: config),
        gate: try matrix(prefix + ".mlp.gate_proj.weight"),
        up: try matrix(prefix + ".mlp.up_proj.weight"),
        down: try matrix(prefix + ".mlp.down_proj.weight"),
        inputNorm: try weight(prefix + ".input_layernorm.weight"),
        postNorm: try weight(prefix + ".post_attention_layernorm.weight"),
        attentionConv: try conv(prefix + ".attention_conv"),
        feedConv: try conv(prefix + ".mlp_conv"), eps: config.eps)
    }
    if config.selectorTopK > 0 {
      self.selector = Selector(
        hiddenProjection: try matrix("candidate_selector.hidden_projection.weight"),
        predecessors: try weight("candidate_selector.predecessor_codebook"),
        successors: try weight("candidate_selector.successor_codebook"),
        topK: config.selectorTopK)
    } else {
      self.selector = nil
    }
  }

  public func makeCaches() -> [LayerCache] { layers.map { _ in LayerCache() } }

  /// A drafter for `text` wherever one is kept: in `dflash` inside the pack, beside the pack in
  /// its library, or in the Hugging Face cache. A drafter names its base model only in its card,
  /// so it is matched by shape — the backbone's depth, width and vocabulary — and DFlash 2 is
  /// preferred to the first DFlash.
  public static func find(for text: TextModel, beside pack: URL) -> URL? {
    find(
      layers: text.layers.count, hidden: text.config.hiddenSize, vocab: text.config.vocabSize,
      beside: pack)
  }

  static func find(
    layers: Int, hidden: Int, vocab: Int, beside pack: URL, hub: URL? = nil
  ) -> URL? {
    let fm = FileManager.default
    var candidates = [pack.appending(path: "dflash")]
    let library = pack.deletingLastPathComponent()
    candidates += (try? fm.contentsOfDirectory(at: library, includingPropertiesForKeys: nil)) ?? []
    let environment = ProcessInfo.processInfo.environment
    let hub =
      hub ?? environment["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
      ?? environment["HF_HOME"].map { URL(fileURLWithPath: $0).appending(path: "hub") }
      ?? fm.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub")
    for repo in (try? fm.contentsOfDirectory(at: hub, includingPropertiesForKeys: nil)) ?? []
    where repo.lastPathComponent.localizedCaseInsensitiveContains("dflash") {
      candidates += (try? fm.contentsOfDirectory(
        at: repo.appending(path: "snapshots"), includingPropertiesForKeys: nil)) ?? []
    }
    var found: (url: URL, second: Bool)?
    for directory in candidates {
      guard let data = try? Data(contentsOf: directory.appending(path: "config.json")),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let kinds = json["architectures"] as? [String],
        kinds.contains(where: { $0.hasPrefix("DFlash") }),
        (json["num_target_layers"] as? NSNumber)?.intValue == layers,
        (json["hidden_size"] as? NSNumber)?.intValue == hidden,
        (json["vocab_size"] as? NSNumber)?.intValue == vocab
      else { continue }
      let second = kinds.contains("DFlash2DraftModel")
      if found == nil || (second && found?.second == false) { found = (directory, second) }
    }
    return found?.url
  }

  /// The block's normalised hidden states after its first position, whose token is already
  /// known. `taps` is the backbone's layers side by side at every position the caches have not
  /// been handed yet; they are kept, and the block itself is not.
  public func hidden(
    block embedded: MLXArray, taps: MLXArray, caches: [LayerCache]
  ) -> MLXArray {
    var h = embedded.asType(.float32)
    let context = MLXFast.rmsNorm(fc(taps.asType(.float32)), weight: hiddenNorm, eps: config.eps)
    for (layer, cache) in zip(layers, caches) {
      h = layer(h, context: context, cache: cache)
    }
    return MLXFast.rmsNorm(h[0..., 1...], weight: norm, eps: config.eps)
  }

  /// The drafts for a block whose logits the backbone's head has made from `hidden`: the
  /// selector's path when the drafter has one, each position's best token otherwise.
  public func drafts(hidden: MLXArray, logits: MLXArray, anchor: MLXArray) -> MLXArray {
    guard let selector else { return argMax(logits, axis: -1) }
    return selector.select(hidden, logits: logits, anchor: anchor)
  }
}
