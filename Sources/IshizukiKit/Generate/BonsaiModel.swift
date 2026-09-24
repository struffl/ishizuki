// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public final class BonsaiModel: @unchecked Sendable {
  public let config: BonsaiConfig
  public let store: WeightStore
  /// The Qwen-shaped stack, for every pack but a DeepSeek-V4.1 one.
  public let text: TextModel!
  public let deepseek: DeepSeekModel?
  public let mtp: MTPHead?
  public let tokenizer: BonsaiTokenizer
  public let directory: URL

  public let tensorPrefix: String

  /// What the generator runs, whichever stack this is.
  public var backbone: any LanguageBackbone { deepseek ?? text }

  /// Whether the routed experts are read from disk into slots rather than held.
  public var streamsExperts: Bool { store.expertTraffic != nil || deepseek?.streamsExperts == true }

  private static let visionPrefix = "vision_tower."
  private static let visionProbe = visionPrefix + "patch_embed.proj.weight"

  private let visionLock = NSLock()
  private var tower: VisionTower?
  private let deepseekWeights: DeepSeekWeights?
  private var deepseekTower: DeepSeekVision?

  public convenience init(
    directory: URL, ropeScaling: RopeScaling = .none, hot: Bool = false
  ) throws {
    if let raw = try? Data(contentsOf: directory.appending(path: "config.json")),
      let object = try? JSONSerialization.jsonObject(with: raw) as? [String: Any],
      DeepSeekConfig.describes(object)
    {
      try self.init(deepseek: directory)
      return
    }
    let config = try BonsaiConfig.load(directory: directory)
    try config.validate()
    var store = try WeightStore(directory: directory)
    if config.profile == .exl3 {
      store = store.canonical(
        zeroCentredNorms: TensorNaming.zeroCentredNormModelTypes.contains(config.modelType))
    }
    // A repacked sparse model keeps its routed experts beside the shards; opening them here
    // is what makes the layers stream rather than load.
    store = try store
      .openingExperts(at: directory, slots: StreamedPlan.slots(for: directory) ?? 16)
      .openingEngrams(at: directory, capacity: BonsaiRuntime.engramRows)
    if let centred = config.centredNorms {
      store = try store.foldingCentredNorms(expected: centred)
    }
    try self.init(
      config: config,
      store: store,
      tokenizer: try BonsaiTokenizer(directory: directory, config: config),
      directory: directory, ropeScaling: ropeScaling, hot: hot)
  }

  /// A pack directory or a `.gguf`, whichever the caller was handed.
  ///
  /// A GGUF's tower travels as a separate `mmproj-*.gguf`, so one sitting beside the model is
  /// taken to belong to it — that is the layout `pull` writes and the one every publisher of
  /// these files uses.
  public convenience init(
    path: URL, ropeScaling: RopeScaling = .none, hot: Bool = false
  ) throws {
    guard path.pathExtension.lowercased() == "gguf" else {
      try self.init(directory: path, ropeScaling: ropeScaling, hot: hot)
      return
    }
    try self.init(
      gguf: path, mmproj: Self.projector(beside: path), ropeScaling: ropeScaling, hot: hot)
  }

  static func projector(beside file: URL) -> URL? {
    let directory = file.deletingLastPathComponent()
    let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
    return
      names
      .filter { $0.lowercased().hasPrefix("mmproj") && $0.hasSuffix(".gguf") }
      .sorted()
      .first
      .map { directory.appending(path: $0) }
  }

  /// One file instead of a directory, and a second one for the tower.
  ///
  /// llama.cpp's converter splits a multimodal checkpoint in two, so a picture needs the
  /// `mmproj-*.gguf` that came out of the same run. Without it the model loads and generates;
  /// it just has no eyes, which `hasVision` reports as usual.
  public convenience init(
    gguf url: URL, mmproj: URL? = nil, ropeScaling: RopeScaling = .none, hot: Bool = false
  ) throws {
    let file = try GGUFFile(url: url)
    let architecture = try GGUFArchitecture(file: file)

    var vision: BonsaiConfig.VisionConfig?
    var store = try GGUFWeights.load(file: file)
    if let mmproj {
      let projector = try GGUFFile(url: mmproj)
      vision = try GGUFVision(file: projector).config
      store = try GGUFWeights.loadVision(file: projector, into: store)
    }

    try self.init(
      config: architecture.config(vision: vision), store: store,
      tokenizer: try BonsaiTokenizer(gguf: file), directory: url.deletingLastPathComponent(),
      ropeScaling: ropeScaling, hot: hot)
  }

  private init(
    config: BonsaiConfig, store: WeightStore, tokenizer: BonsaiTokenizer, directory: URL,
    ropeScaling: RopeScaling, hot: Bool
  ) throws {
    self.directory = directory
    self.config = config
    self.store = store
    self.tokenizer = tokenizer

    self.tensorPrefix = store.languageModelPrefix

    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: tensorPrefix)
    var scaling = ropeScaling
    let rope = config.textConfig.ropeParameters
    if !scaling.isActive, rope.ropeType == "yarn", let factor = rope.factor, factor > 1 {
      scaling = RopeScaling(
        method: .yarn, factor: factor,
        originalContext: rope.originalMaxPositionEmbeddings
          ?? config.textConfig.maxPositionEmbeddings)
    } else if scaling.isActive {
      scaling.originalContext = config.textConfig.maxPositionEmbeddings
    }
    self.text = try TextModel(
      config: config, factory: factory, store: store, ropeScaling: scaling)
    self.deepseek = nil
    self.deepseekWeights = nil

    // The head is optional twice over: the config has to declare it and the pack has to ship
    // the tensors. A pack that declares it and omits them still loads, without drafting, and so
    // does one whose head is not the single-`fc` kind `MTPHead` reads: qwen4_exp's is split
    // into `fc_embedding` and `fc_hidden` behind a mixer of its own.
    let head = tensorPrefix + "mtp.fc"
    if config.components?.mtp == true,
      store.has(head + ".weight") || store.has(head + ".trellis")
    {
      self.mtp = try MTPHead(
        config: config.textConfig, factory: factory, store: store, rope: text.rope)
    } else {
      self.mtp = nil
    }

    if hot { try vision() }
  }

  /// A DeepSeek-V4.1 release, read from its own shards: the fp8 and fp4 weights multiplied as
  /// they are, the routed experts streamed into slots, the n-gram tables read a row at a time.
  private init(deepseek directory: URL) throws {
    let checkpoint = try DeepSeekCheckpoint(directory: directory)
    let config = checkpoint.config
    let tokenizer = try BonsaiTokenizer(
      directory: directory, eosTokenIds: [config.eosTokenId].compactMap { $0 })
    let streams = BonsaiRuntime.deepseekStreamsExperts
    let weights = DeepSeekWeights(
      checkpoint: checkpoint, compute: BonsaiRuntime.deepseekCompute,
      expertSlots: streams ? DeepSeekModel.expertSlots(for: checkpoint) : nil)
    let map =
      DeepSeekTokenMap.saved(in: directory, count: config.vocabSize)
      ?? DeepSeekTokenMap.build(tokenizer: tokenizer, count: config.vocabSize)
    self.deepseek = try DeepSeekModel(weights: weights, tokenMap: map)
    self.deepseekWeights = weights
    self.text = nil
    self.mtp = nil
    self.directory = directory
    self.store = WeightStore(arrays: [:])
    self.tokenizer = tokenizer
    self.tensorPrefix = ""
    self.config = try BonsaiConfig.flat([
      "model_type": "deepseek_v41", "hidden_size": config.dim,
      "num_hidden_layers": config.layers, "num_attention_heads": config.heads,
      "num_key_value_heads": 1, "head_dim": config.headDim, "vocab_size": config.vocabSize,
      "max_position_embeddings": 1_048_576, "rms_norm_eps": Double(config.normEps),
      "eos_token_id": config.eosTokenId ?? 1, "bos_token_id": config.bosTokenId ?? 0,
      "num_experts": config.routedExperts, "num_experts_per_tok": config.activatedExperts,
      "moe_intermediate_size": config.moeInterDim,
    ])
  }

  /// A pack has a tower when its config declares one and the shards actually carry it. Asking
  /// reads no tensors, so the question stands on its own before anything has been built.
  public var hasVision: Bool {
    if let deepseekWeights { return deepseekWeights.has("vision.patch_embed.proj.weight") }
    return config.visionConfig != nil && store.has(Self.visionProbe)
  }

  public var isVisionLoaded: Bool {
    visionLock.lock()
    defer { visionLock.unlock() }
    return tower != nil || deepseekTower != nil
  }

  /// DeepSeek-V4.1's tower, built the first time a picture needs it.
  public func deepseekVision() throws -> DeepSeekVision? {
    visionLock.lock()
    defer { visionLock.unlock() }
    if let deepseekTower { return deepseekTower }
    guard let deepseekWeights, hasVision else { return nil }
    let built = try DeepSeekVision(weights: deepseekWeights)
    deepseekTower = built
    return built
  }

  /// A picture off disk, prepared the way this model's tower reads pictures.
  public func processImage(contentsOf url: URL) throws -> ProcessedImage {
    if deepseek != nil {
      guard let tower = try deepseekVision() else {
        throw BonsaiError.missingComponent("this release has no vision tower")
      }
      let image = try tower.prepare(contentsOf: url)
      return ProcessedImage(
        patches: image.patches, grid: (t: 1, h: image.patchRows, w: image.patchColumns),
        deepseek: image)
    }
    guard let visionConfig = config.visionConfig, hasVision else {
      throw BonsaiError.missingComponent("this pack has no vision tower")
    }
    return try ImageProcessor(config: visionConfig).process(contentsOf: url)
  }

  /// The tower, built and read off disk the first time an image needs it and held afterwards.
  /// A text-only session never pays for the couple of gigabytes it weighs; `--hot` is how a
  /// server asks for that cost at startup instead of on the first picture.
  @discardableResult
  public func vision() throws -> VisionTower? {
    if deepseek != nil {
      _ = try deepseekVision()
      return nil
    }
    visionLock.lock()
    defer { visionLock.unlock() }
    if let tower { return tower }
    guard let visionConfig = config.visionConfig, store.has(Self.visionProbe) else {
      return nil
    }
    let built = try VisionTower(
      config: visionConfig, store: store, quantization: config.quantization)
    store.warm(prefix: Self.visionPrefix)
    tower = built
    return built
  }

  public var hasMTP: Bool { mtp != nil }
}
