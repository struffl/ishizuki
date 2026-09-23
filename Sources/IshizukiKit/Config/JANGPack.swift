// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

/// A JANG pack is MLX affine shards with a `jang_config.json` beside them saying what the
/// plain config does not: whether the weights sit in a Hadamard-rotated basis, and whether the
/// norms were left zero-centred for the runtime to shift. Both read as an ordinary affine pack
/// if ignored, and both then generate fluent nonsense, so a manifest this cannot honour refuses
/// to load instead.
public enum JANGPack {
  public static let manifestFile = "jang_config.json"
  public static let rotationContract = "prism.hadamard.v1"
  public static let centredNormLayout = "zero-centered-runtime-plus-one"
  static let languagePrefix = "language_model."

  public static func manifest(in directory: URL) throws -> [String: Any]? {
    let url = directory.appending(path: manifestFile)
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    guard
      let object = try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
        as? [String: Any]
    else {
      throw BonsaiError.unsupportedModel("\(manifestFile) is not a JSON object")
    }
    return object
  }

  static func apply(
    to config: inout BonsaiConfig, raw: [String: Any], manifest: [String: Any]?
  ) throws {
    if let manifest { try checkFormat(manifest) }

    let layout = manifest?["layout"] as? [String: Any]
    if let norms = layout?["language_norms"] as? String {
      guard norms == centredNormLayout else {
        throw BonsaiError.unsupportedModel("JANG norm layout '\(norms)' is not one this reads")
      }
      config.centredNorms = (layout?["shifted_norm_count"] as? NSNumber)?.intValue ?? 0
    }
    if let activations = layout?["gdn_activation_layout"] as? String {
      guard activations == "grouped" else {
        throw BonsaiError.unsupportedModel(
          "JANG delta-net activation layout '\(activations)' is not grouped")
      }
      config.gdnActivationLayout = activations
    }

    let runtime = manifest?["runtime"] as? [String: Any]
    guard
      let hadamard = raw["hadamard"] as? [String: Any]
        ?? manifest?["hadamard"] as? [String: Any]
    else {
      if runtime?["requires_hadamard_activation_transform"] as? Bool == true {
        throw BonsaiError.unsupportedModel(
          "\(manifestFile) requires a Hadamard transform but names no rotated modules")
      }
      return
    }
    try rotate(&config, hadamard)
  }

  private static func checkFormat(_ manifest: [String: Any]) throws {
    if let format = manifest["format"] as? String, format != "jang" {
      throw BonsaiError.unsupportedModel("\(manifestFile) declares format '\(format)'")
    }
    if let weights = manifest["weight_format"] as? String, weights != "affine" {
      throw BonsaiError.unsupportedModel(
        "JANG weight format '\(weights)' is not one this reads; only affine is")
    }
    let runtime = manifest["runtime"] as? [String: Any]
    if runtime?["requires_jang_affine1_expansion"] as? Bool == true {
      throw BonsaiError.unsupportedModel("JANG affine1 expansion is not supported")
    }
  }

  private static func rotate(_ config: inout BonsaiConfig, _ h: [String: Any]) throws {
    let expected: [(String, String)] = [
      ("contract", rotationContract),
      ("transform", "normalized-sylvester-walsh-hadamard"),
      ("axis", "input-last-dimension"),
      ("sign_mode", "explicit"),
    ]
    for (key, value) in expected {
      guard let found = h[key] as? String else { continue }
      guard found == value else {
        throw BonsaiError.invalidTransform("hadamard \(key) is '\(found)', expected '\(value)'")
      }
    }
    if h["gdn_v_grouped"] as? Bool == false {
      throw BonsaiError.unsupportedModel("rotated pack stores delta-net value heads tiled")
    }
    guard let block = (h["block_size"] as? NSNumber)?.intValue, block > 0 else {
      throw BonsaiError.invalidTransform("hadamard block_size is missing")
    }

    func records(_ key: String, embedding: Bool) throws -> [BonsaiConfig.PackedModuleRecord] {
      try ((h[key] as? [String]) ?? []).map { name in
        guard name.hasPrefix(languagePrefix) else {
          throw BonsaiError.unsupportedModel(
            "rotated module \(name) sits outside the language model")
        }
        return BonsaiConfig.PackedModuleRecord(
          path: String(name.dropFirst(languagePrefix.count)), block: block,
          embedding: embedding, dtype: "float16")
      }
    }
    let modules =
      try records("forward_modules", embedding: false)
      + records("inverse_modules", embedding: true)
    guard !modules.isEmpty else {
      throw BonsaiError.invalidTransform("hadamard block names no modules")
    }

    config.baseModelType = config.modelType
    config.modelType = BonsaiConfig.hadamardModelType
    config.modules = modules
  }
}
