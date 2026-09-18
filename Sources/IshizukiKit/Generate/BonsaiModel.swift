// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

public final class BonsaiModel: @unchecked Sendable {
  public let config: BonsaiConfig
  public let store: WeightStore
  public let text: TextModel
  public let vision: VisionTower?
  public let tokenizer: BonsaiTokenizer
  public let directory: URL

  public let tensorPrefix: String

  public init(
    directory: URL, loadVision: Bool = true, ropeScaling: RopeScaling = .none
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
    if scaling.isActive {
      scaling.originalContext = config.textConfig.maxPositionEmbeddings
    }
    self.text = try TextModel(
      config: config, factory: factory, store: store, ropeScaling: scaling)

    if loadVision, let visionConfig = config.visionConfig,
      store.has("vision_tower.patch_embed.proj.weight")
    {
      self.vision = try VisionTower(config: visionConfig, store: store)
    } else {
      self.vision = nil
    }

    self.tokenizer = try BonsaiTokenizer(directory: directory, config: config)
  }

  public var hasVision: Bool { vision != nil }

  public var hasMTP: Bool {
    config.components?.mtp == true && !store.names(prefix: "mtp").isEmpty
  }
}
