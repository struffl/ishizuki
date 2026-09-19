// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// A full-precision checkpoint, read a tensor at a time.
///
/// MLX maps safetensors rather than reading them: opening a 5 GB shard costs about a megabyte,
/// and a tensor is only paid for when it is evaluated. That is what lets a 52 GB source be
/// quantized on a machine that could not hold it — nothing is resident but the layer in hand.
public final class SourceCheckpoint: @unchecked Sendable {
  public let directory: URL
  public let config: [String: Any]
  public let textConfig: [String: Any]

  private let shardOf: [String: String]
  private let lock = NSLock()
  private var opened: [String: [String: MLXArray]] = [:]
  private var order: [String] = []
  private let residentShards: Int

  public init(directory: URL, residentShards: Int = 2) throws {
    self.directory = directory
    self.residentShards = max(1, residentShards)

    let configURL = directory.appending(path: "config.json")
    guard let raw = try? Data(contentsOf: configURL),
      let object = try? JSONSerialization.jsonObject(with: raw) as? [String: Any]
    else {
      throw BonsaiError.unsupportedModel("no readable config.json in \(directory.path)")
    }
    self.config = object
    self.textConfig = (object["text_config"] as? [String: Any]) ?? object

    let fm = FileManager.default
    let single = directory.appending(path: "model.safetensors")
    let index = directory.appending(path: "model.safetensors.index.json")
    if fm.fileExists(atPath: index.path) {
      guard
        let payload = try? JSONSerialization.jsonObject(with: try Data(contentsOf: index)),
        let map = (payload as? [String: Any])?["weight_map"] as? [String: String]
      else {
        throw BonsaiError.missingWeight("unreadable weight_map in \(index.path)")
      }
      self.shardOf = map
    } else if fm.fileExists(atPath: single.path) {
      let names = try loadArrays(url: single).keys
      self.shardOf = Dictionary(uniqueKeysWithValues: names.map { ($0, "model.safetensors") })
    } else {
      throw BonsaiError.missingWeight("no safetensors in \(directory.path)")
    }
  }

  public var tensorNames: [String] { Array(shardOf.keys) }

  public func has(_ name: String) -> Bool { shardOf[name] != nil }

  /// The tensor, still unevaluated. Reading it costs nothing until it is used.
  public func tensor(_ name: String) throws -> MLXArray {
    guard let shard = shardOf[name] else {
      throw BonsaiError.missingWeight(name)
    }
    lock.lock()
    defer { lock.unlock() }

    if let cached = opened[shard]?[name] { return cached }
    let arrays = try loadArrays(url: directory.appending(path: shard))
    opened[shard] = arrays
    order.removeAll { $0 == shard }
    order.append(shard)
    // Dropping a shard releases its mapping; anything already taken from it stays valid.
    while order.count > residentShards {
      opened.removeValue(forKey: order.removeFirst())
    }
    guard let array = arrays[name] else {
      throw BonsaiError.missingWeight("\(name) is named by the index but absent from \(shard)")
    }
    return array
  }

  public func optional(_ name: String) -> MLXArray? { try? tensor(name) }

  public var layerCount: Int {
    (textConfig["num_hidden_layers"] as? NSNumber)?.intValue ?? 0
  }

  public var hiddenSize: Int {
    (textConfig["hidden_size"] as? NSNumber)?.intValue ?? 0
  }

  /// The prefix the language model's tensors carry, which a VLM checkpoint nests.
  public var tensorPrefix: String {
    has("language_model.model.norm.weight") ? "language_model." : ""
  }

  public var layerTypes: [String] {
    if let types = textConfig["layer_types"] as? [String], types.count == layerCount {
      return types
    }
    let interval = (textConfig["full_attention_interval"] as? NSNumber)?.intValue ?? 4
    return (0..<layerCount).map { ($0 + 1) % interval == 0 ? "full_attention" : "linear_attention" }
  }
}
