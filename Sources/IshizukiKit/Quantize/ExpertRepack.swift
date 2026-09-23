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
  public static let folder = "experts"

  /// Two gigabytes of shard, rather than the quantizer's four: a machine reaching for this is
  /// short of memory, and a shard is held whole until it is flushed.
  public static let shardLimit = 2 << 30

  public struct Plan: Sendable {
    public var layers: [Int]
    public var layout: ExpertLayout
    public var residentBytes: Int
    public var expertBytes: Int

    public var expertCount: Int { layout.expertCount }
  }

  /// What a split would produce, read from the source's tensor metadata alone.
  public struct Preview: Sendable {
    public var layers: [Int]
    public var expertCount: Int
    public var residentBytes: Int
    public var expertBytes: Int
  }

  public struct Progress: Sendable {
    public var detail: String
    public var fraction: Double
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

  static func isExpert(_ name: String) -> Bool { name.contains(".switch_mlp.") }

  /// Which layer a stacked expert tensor belongs to, and what it is called inside the blob.
  public static func address(_ name: String) -> (layer: Int, part: String)? {
    guard let range = name.range(of: ".mlp.switch_mlp.") else { return nil }
    let head = name[name.startIndex..<range.lowerBound]
    guard let marker = head.range(of: "model.layers.", options: .backwards),
      let layer = Int(head[marker.upperBound...])
    else { return nil }
    let part = String(name[range.upperBound...])
    guard part.split(separator: ".").count == 2 else { return nil }
    return (layer, part)
  }

  static func layerFile(_ layer: Int) -> String {
    "\(folder)/layer_\(String(format: "%02d", layer)).bin"
  }

  /// Whether a pack already keeps its experts beside itself.
  public static func isSplit(_ directory: URL) -> Bool {
    FileManager.default.fileExists(atPath: directory.appending(path: layoutFile).path)
  }

  /// Reads the source's shapes and says what the halves would weigh. Safetensors are mapped,
  /// not read, so this costs a header parse rather than a pass over the weights.
  public static func preview(source: URL) throws -> Preview {
    let (store, layers, layout) = try survey(source: source)
    var expertBytes = 0
    var residentBytes = 0
    for (name, array) in store.arrays {
      let bytes = array.size * array.itemSize
      if isExpert(name) { expertBytes += bytes } else { residentBytes += bytes }
    }
    return Preview(
      layers: layers, expertCount: layout.expertCount,
      residentBytes: residentBytes,
      expertBytes: max(expertBytes, layers.count * layout.expertCount * layout.stride))
  }

  /// Splits `source` into `destination`, returning what it wrote.
  public static func run(
    source: URL, destination: URL, shardLimit: Int = ExpertRepack.shardLimit,
    log: (String) -> Void = { _ in },
    progress: (Progress) throws -> Void = { _ in }
  ) throws -> Plan {
    guard source.standardizedFileURL != destination.standardizedFileURL else {
      throw BonsaiError.unsupportedModel("a pack cannot be split over itself")
    }
    guard !isSplit(source) else {
      throw BonsaiError.unsupportedModel(
        "\(source.lastPathComponent) already keeps its experts on disk")
    }

    let (store, layers, layout) = try survey(source: source)
    let tensorPrefix = prefix(store)

    let fm = FileManager.default
    try fm.createDirectory(
      at: destination.appending(path: folder), withIntermediateDirectories: true)

    let resident = store.arrays.keys.filter { !isExpert($0) }.sorted()
    let steps = Double(layers.count + resident.count)
    var done = 0.0

    var expertBytes = 0
    for layer in layers {
      expertBytes += try write(
        layer: layer, prefix: tensorPrefix, store: store, layout: layout,
        to: destination.appending(path: layerFile(layer)))
      done += 1
      log("experts: layer \(layer) written")
      try progress(Progress(detail: "layer \(layer)", fraction: done / steps))
    }

    var writer = PackWriter(directory: destination, shardLimit: shardLimit)
    for name in resident {
      try writer.add(name, store.arrays[name]!)
      done += 1
      try progress(Progress(detail: name, fraction: done / steps))
    }
    let summary = try writer.finish()
    log("resident: \(summary.shards) shard\(summary.shards == 1 ? "" : "s")")

    try carrySidecars(from: source, to: destination)

    try writeLayout(layout, to: destination)

    return Plan(
      layers: layers, layout: layout, residentBytes: summary.byteCount,
      expertBytes: expertBytes)
  }

  /// The sparse layers, and the geometry every one of them shares.
  private static func survey(source: URL) throws -> (WeightStore, [Int], ExpertLayout) {
    let config = try BonsaiConfig.load(directory: source)
    let text = config.textConfig
    guard let expertCount = text.numExperts, expertCount > 0 else {
      throw BonsaiError.unsupportedModel("\(source.lastPathComponent) routes through no experts")
    }

    let store = try WeightStore(directory: source)
    let tensorPrefix = prefix(store)
    let sparse = text.isSparse
    let layers = (0..<text.numHiddenLayers).filter {
      sparse[$0] && store.has(expertPath(tensorPrefix, $0, "gate_proj", "weight"))
    }
    guard let first = layers.first else {
      throw BonsaiError.unsupportedModel("no sparse layers to split out")
    }

    var described: [(name: String, shape: [Int], dtype: DType)] = []
    for projection in projections {
      for component in components {
        let name = expertPath(tensorPrefix, first, projection, component)
        guard store.has(name) else { continue }
        let array = try store(name)
        guard array.dim(0) == expertCount else {
          throw BonsaiError.shapeMismatch(
            "\(name) stacks \(array.dim(0)) experts, not \(expertCount)")
        }
        described.append(
          (
            name: "\(projection).\(component)",
            shape: Array(array.shape.dropFirst()), dtype: array.dtype
          ))
      }
    }
    return (store, layers, ExpertLayout.plan(expertCount: expertCount, tensors: described))
  }

  /// The layout one layer's stacked tensors describe: every layer of a model shares it.
  public static func layout(of parts: [String: MLXArray], expertCount: Int) -> ExpertLayout {
    ExpertLayout.plan(
      expertCount: expertCount,
      tensors: parts.map {
        (name: $0.key, shape: Array($0.value.shape.dropFirst()), dtype: $0.value.dtype)
      })
  }

  /// One layer's stacked tensors cut into one blob per expert, written an expert at a time so
  /// a layer larger than memory still converts. Keys are part names — `gate_proj.scales`.
  @discardableResult
  public static func blob(
    parts: [String: MLXArray], layout: ExpertLayout, to url: URL
  ) throws -> Int {
    let ordered = layout.parts.sorted { $0.value.offset < $1.value.offset }
    var sources: [(part: ExpertLayout.Part, array: MLXArray)] = []
    for (name, part) in ordered {
      guard let array = parts[name] else {
        throw BonsaiError.missingWeight("\(name) is not among this layer's experts")
      }
      guard array.dim(0) == layout.expertCount, Array(array.shape.dropFirst()) == part.shape,
        array.dtype == (try part.type)
      else {
        throw BonsaiError.shapeMismatch("\(name) is not shaped like the layer before it")
      }
      sources.append((part, array))
    }

    let packed = (ordered.last?.value.offset ?? 0) + (ordered.last?.value.byteCount ?? 0)
    let padding = Data(count: layout.stride - packed)

    let fm = FileManager.default
    try fm.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
    fm.createFile(atPath: url.path, contents: nil)
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }

    for expert in 0..<layout.expertCount {
      for (part, array) in sources {
        let slice = array[expert]
        eval(slice)
        let bytes = slice.asData().data
        guard bytes.count == part.byteCount else {
          throw BonsaiError.shapeMismatch(
            "expert \(expert) is \(bytes.count) bytes, not \(part.byteCount)")
        }
        try handle.write(contentsOf: bytes)
      }
      if !padding.isEmpty { try handle.write(contentsOf: padding) }
    }
    return layout.expertCount * layout.stride
  }

  /// The layout a pack streams by, written once beside its experts.
  public static func writeLayout(_ layout: ExpertLayout, to destination: URL) throws {
    try write(layout, to: destination.appending(path: layoutFile))
  }

  /// One layer's own layout, for a pack whose allocation gave its layers different widths.
  /// The shared `layout.json` still marks the pack as split, and describes any layer without
  /// a file of its own.
  static func layerLayoutFile(_ layer: Int) -> String {
    "\(folder)/layer_\(String(format: "%02d", layer)).json"
  }

  public static func writeLayout(_ layout: ExpertLayout, layer: Int, to destination: URL) throws {
    try write(layout, to: destination.appending(path: layerLayoutFile(layer)))
  }

  private static func write(_ layout: ExpertLayout, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(layout).write(to: url)
  }

  private static func write(
    layer: Int, prefix: String, store: WeightStore, layout: ExpertLayout, to url: URL
  ) throws -> Int {
    var parts: [String: MLXArray] = [:]
    for name in layout.parts.keys {
      let pieces = name.split(separator: ".")
      parts[name] = try store(expertPath(prefix, layer, String(pieces[0]), String(pieces[1])))
    }
    return try blob(parts: parts, layout: layout, to: url)
  }

  /// Everything a pack needs that is not a weight: the config, the tokenizer, and any table
  /// the pack already streams.
  ///
  /// A pack straight out of the HuggingFace cache is a tree of symlinks into a blob store, and
  /// a link copied out of it points at nothing. What the sidecars say has to be copied, not
  /// where they say it.
  private static func carrySidecars(from source: URL, to destination: URL) throws {
    let fm = FileManager.default
    let skipped: Set<String> = ["bin", "pt", "pth", "gguf", "h5", "msgpack", "onnx"]
    for name in (try? fm.contentsOfDirectory(atPath: source.path))?.sorted() ?? [] {
      guard !name.hasPrefix("."), name != folder else { continue }
      guard !name.hasSuffix(".safetensors"), !name.hasSuffix(".safetensors.index.json") else {
        continue
      }
      let from = source.appending(path: name)
      var isDirectory: ObjCBool = false
      _ = fm.fileExists(atPath: from.resolvingSymlinksInPath().path, isDirectory: &isDirectory)
      let suffix = (name as NSString).pathExtension.lowercased()
      if !isDirectory.boolValue, skipped.contains(suffix) { continue }
      let to = destination.appending(path: name)
      if fm.fileExists(atPath: to.path) { try fm.removeItem(at: to) }
      if isDirectory.boolValue {
        try fm.createDirectory(at: to, withIntermediateDirectories: true)
        for inner in (try? fm.contentsOfDirectory(atPath: from.path))?.sorted() ?? []
        where !inner.hasPrefix(".") {
          try fm.copyItem(
            at: from.appending(path: inner).resolvingSymlinksInPath(),
            to: to.appending(path: inner))
        }
      } else {
        try fm.copyItem(at: from.resolvingSymlinksInPath(), to: to)
      }
    }
  }
}
