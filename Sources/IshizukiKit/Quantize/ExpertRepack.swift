// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Splitting a sparse pack into the part that stays in memory and the part that does not.

import Foundation
import MLX

/// Writes a MoE pack out as a resident half and one file of routed experts per layer.
///
/// The routed experts are most of a sparse model's weight and the least of its work: a token
/// touches a sixth of them. Keeping them beside the model rather than inside it is what lets a
/// machine hold a model larger than its memory, and the split is the whole of the trick — the
/// values are copied through unchanged, never dequantized and requantized.
public enum ExpertRepack {
  public static let layoutFile = "experts/layout.json"

  public struct Plan: Sendable {
    public var layers: [Int]
    public var layout: ExpertLayout
    public var residentBytes: Int
    public var expertBytes: Int
  }

  /// The three projections an expert owns, each with its quantized companions.
  static let projections = ["gate_proj", "up_proj", "down_proj"]
  static let components = ["weight", "scales", "biases"]

  /// A multimodal pack nests the language model; a text-only one does not. Reading the wrong
  /// one finds no experts at all, which reads as a model that routes through none.
  static func prefix(_ store: WeightStore) -> String {
    store.has("language_model.model.norm.weight")
      || store.names(prefix: "language_model.model.layers.").first != nil
      ? "language_model." : ""
  }

  static func expertPath(
    _ prefix: String, _ layer: Int, _ projection: String, _ component: String
  ) -> String {
    "\(prefix)model.layers.\(layer).mlp.switch_mlp.\(projection).\(component)"
  }

  /// Splits `source` into `destination`, returning what it wrote.
  public static func run(
    source: URL, destination: URL,
    log: @escaping (String) -> Void = { _ in }
  ) throws -> Plan {
    let config = try BonsaiConfig.load(directory: source)
    let text = config.textConfig
    guard let expertCount = text.numExperts, expertCount > 0 else {
      throw BonsaiError.unsupportedModel("\(source.lastPathComponent) routes through no experts")
    }

    let store = try WeightStore(directory: source)
    let tensorPrefix = prefix(store)
    let sparse = text.isSparse
    let layers = (0..<text.numHiddenLayers).filter { sparse[$0] }

    let fm = FileManager.default
    try fm.createDirectory(
      at: destination.appending(path: "experts"), withIntermediateDirectories: true)

    // Every sparse layer has the same expert geometry, so the first one describes the file.
    var described: [(name: String, shape: [Int], dtype: DType)] = []
    guard let first = layers.first else {
      throw BonsaiError.unsupportedModel("no sparse layers to split out")
    }
    for projection in projections {
      for component in components {
        let name = expertPath(tensorPrefix, first, projection, component)
        guard store.has(name) else { continue }
        let array = try store(name)
        described.append(
          (
            name: "\(projection).\(component)",
            // The leading axis is the expert; one blob holds one expert's slice of it.
            shape: Array(array.shape.dropFirst()), dtype: array.dtype
          ))
      }
    }
    let layout = ExpertLayout.plan(expertCount: expertCount, tensors: described)

    var expertBytes = 0
    for layer in layers {
      let url = destination.appending(path: "experts/layer_\(String(format: "%02d", layer)).bin")
      var blob = Data(count: expertCount * layout.stride)
      for (name, part) in layout.parts {
        let pieces = name.split(separator: ".")
        let full = expertPath(tensorPrefix, layer, String(pieces[0]), String(pieces[1]))
        let array = try store(full)
        guard array.dim(0) == expertCount else {
          throw BonsaiError.shapeMismatch(
            "\(full) stacks \(array.dim(0)) experts, not \(expertCount)")
        }
        let bytes = array.asData().data
        for expert in 0..<expertCount {
          let from = expert * part.byteCount
          let to = expert * layout.stride + part.offset
          blob.replaceSubrange(
            to..<(to + part.byteCount),
            with: bytes[bytes.startIndex + from..<bytes.startIndex + from + part.byteCount])
        }
      }
      try blob.write(to: url)
      expertBytes += blob.count
      log("experts: layer \(layer) written")
    }

    // Everything the experts are not, saved once.
    var resident: [String: MLXArray] = [:]
    for name in store.arrays.keys where !name.contains(".switch_mlp.") {
      resident[name] = store.arrays[name]
    }
    try save(arrays: resident, url: destination.appending(path: "model.safetensors"))

    // A pack straight out of the HuggingFace cache is a tree of symlinks into a blob store,
    // and a link copied out of it points at nothing. What the sidecars say has to be copied,
    // not where they say it.
    for name in ["config.json", "tokenizer.json", "chat_template.jinja", "tokenizer_config.json"]
    where fm.fileExists(atPath: source.appending(path: name).path) {
      try? fm.removeItem(at: destination.appending(path: name))
      try fm.copyItem(
        at: source.appending(path: name).resolvingSymlinksInPath(),
        to: destination.appending(path: name))
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(layout).write(to: destination.appending(path: layoutFile))

    let residentBytes =
      (try? destination.appending(path: "model.safetensors").resourceValues(
        forKeys: [.fileSizeKey]))?.fileSize ?? 0
    return Plan(
      layers: layers, layout: layout, residentBytes: residentBytes, expertBytes: expertBytes)
  }
}
