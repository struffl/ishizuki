// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

public final class BonsaiModel: @unchecked Sendable {
  public let config: BonsaiConfig
  public let store: WeightStore
  public let text: TextModel
  public let mtp: MTPHead?
  public let tokenizer: BonsaiTokenizer
  public let directory: URL

  public let tensorPrefix: String

  private static let visionPrefix = "vision_tower."
  private static let visionProbe = visionPrefix + "patch_embed.proj.weight"

  private let visionLock = NSLock()
  private var tower: VisionTower?

  public convenience init(
    directory: URL, ropeScaling: RopeScaling = .none, hot: Bool = false
  ) throws {
    let config = try BonsaiConfig.load(directory: directory)
    try config.validate()
    try self.init(
      config: config, store: try WeightStore(directory: directory),
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

    let nested = store.has("language_model.model.norm.weight")
    self.tensorPrefix = nested ? "language_model." : ""

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

    // The head is optional twice over: the config has to declare it and the pack has to ship
    // the tensors. A pack that declares it and omits them still loads, without drafting.
    if config.components?.mtp == true, !store.names(prefix: tensorPrefix + "mtp").isEmpty {
      self.mtp = try MTPHead(
        config: config.textConfig, factory: factory, store: store, rope: text.rope)
    } else {
      self.mtp = nil
    }

    if hot { try vision() }
  }

  /// A pack has a tower when its config declares one and the shards actually carry it. Asking
  /// reads no tensors, so the question stands on its own before anything has been built.
  public var hasVision: Bool {
    config.visionConfig != nil && store.has(Self.visionProbe)
  }

  public var isVisionLoaded: Bool {
    visionLock.lock()
    defer { visionLock.unlock() }
    return tower != nil
  }

  /// The tower, built and read off disk the first time an image needs it and held afterwards.
  /// A text-only session never pays for the couple of gigabytes it weighs; `--hot` is how a
  /// server asks for that cost at startup instead of on the first picture.
  @discardableResult
  public func vision() throws -> VisionTower? {
    visionLock.lock()
    defer { visionLock.unlock() }
    if let tower { return tower }
    guard let visionConfig = config.visionConfig, store.has(Self.visionProbe) else {
      return nil
    }
    let built = try VisionTower(config: visionConfig, store: store)
    store.warm(prefix: Self.visionPrefix)
    tower = built
    return built
  }

  public var hasMTP: Bool { mtp != nil }
}
