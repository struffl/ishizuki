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

  public init(
    directory: URL, ropeScaling: RopeScaling = .none, hot: Bool = false
  ) throws {
    self.directory = directory

    let config = try BonsaiConfig.load(directory: directory)
    try config.validate()
    self.config = config

    let store = try WeightStore(directory: directory)
    self.store = store

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

    self.tokenizer = try BonsaiTokenizer(directory: directory, config: config)

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
