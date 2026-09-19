// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation

/// The packs on this machine that this runtime can actually load.
///
/// A directory qualifies by having a config.json this build understands and weights beside it,
/// so a half-finished download or an unrelated checkout is not offered as a model. A `.gguf`
/// qualifies on its own, since it carries its architecture, its vocabulary and its weights in
/// the one file.
public struct ModelCatalog: Sendable {
  public enum Format: String, Sendable, Equatable {
    /// This runtime's own directory of quantized shards beside a config.json.
    case pack
    /// One file in llama.cpp's container.
    case gguf
  }

  public struct Entry: Sendable, Equatable {
    public let id: String
    public let format: Format
    /// The pack's directory, or the `.gguf` file itself.
    public let url: URL
    public let byteCount: Int
    public let quantization: String
    public let hasVision: Bool
    public let hasMTP: Bool
    public let contextTokens: Int

    /// Where a pack's other files live — its chat template, its ANE bank, its prefix store. A
    /// GGUF has no such files, so this is the folder it happens to sit in and nothing more.
    public var directory: URL {
      format == .pack ? url : url.deletingLastPathComponent()
    }

    public var displayName: String { id }
  }

  public let entries: [Entry]

  public init(entries: [Entry]) {
    self.entries = entries
  }

  public subscript(id: String) -> Entry? {
    entries.first { $0.id == id }
  }

  /// Scans each directory for packs, one level deep, and the directory itself if it is one.
  /// The HuggingFace cache nests its checkouts under snapshots/<revision>, which is followed
  /// so a pack pulled with `hf download` is offered like any other.
  public static func discover(in roots: [URL]) -> ModelCatalog {
    var found: [String: Entry] = [:]
    for root in roots {
      for directory in candidates(under: root) {
        if let entry = inspect(directory), found[entry.id] == nil {
          // First root wins, so an explicitly managed copy outranks a cached one.
          found[entry.id] = entry
        }
        for file in ggufFiles(in: directory) {
          guard let entry = inspect(gguf: file), found[entry.id] == nil else { continue }
          found[entry.id] = entry
        }
      }
    }
    return ModelCatalog(entries: found.values.sorted { $0.id < $1.id })
  }

  private static func candidates(under root: URL) -> [URL] {
    let fm = FileManager.default
    var result: [URL] = [root]
    guard
      let children = try? fm.contentsOfDirectory(
        at: root, includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles])
    else { return result }

    for child in children {
      guard (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
      else { continue }
      result.append(child)
      // models--org--name/snapshots/<revision>
      let snapshots = child.appending(path: "snapshots")
      if let revisions = try? fm.contentsOfDirectory(
        at: snapshots, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
      {
        result.append(contentsOf: revisions)
      }
    }
    return result
  }

  /// The model files in a directory, which is every `.gguf` but the vision projector: an
  /// `mmproj` is half a model, loadable only beside the one it was split from.
  private static func ggufFiles(in directory: URL) -> [URL] {
    let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
    return
      names
      .filter { $0.hasSuffix(".gguf") && !$0.lowercased().hasPrefix("mmproj") }
      .sorted()
      .map { directory.appending(path: $0) }
  }

  private static func inspect(gguf file: URL) -> Entry? {
    guard let opened = try? GGUFFile(url: file),
      let architecture = try? GGUFArchitecture(file: opened)
    else { return nil }

    let size =
      (try? file.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]))?.fileSize

    return Entry(
      id: file.deletingPathExtension().lastPathComponent,
      format: .gguf,
      url: file,
      byteCount: size ?? 0,
      quantization: dominantType(of: opened),
      // The tower travels as its own file in this format, so a model is text-only until an
      // mmproj is found beside it.
      hasVision: hasProjector(beside: file),
      hasMTP: architecture.hasMTP,
      contextTokens: architecture.textConfig.maxPositionEmbeddings)
  }

  /// The type the file mostly *is*, weighted by bytes rather than by tensor count: a pack is
  /// hundreds of small f32 norms and a handful of enormous projections, and counting tensors
  /// would name it after the norms.
  private static func dominantType(of file: GGUFFile) -> String {
    var bytes: [GGMLType: Int] = [:]
    for tensor in file.tensors { bytes[tensor.type, default: 0] += tensor.byteCount }
    guard let winner = bytes.max(by: { $0.value < $1.value })?.key else { return "unknown" }
    return winner.name
  }

  private static func hasProjector(beside file: URL) -> Bool {
    let directory = file.deletingLastPathComponent()
    let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
    return names.contains {
      $0.lowercased().hasPrefix("mmproj") && $0.hasSuffix(".gguf")
    }
  }

  private static func inspect(_ directory: URL) -> Entry? {
    let fm = FileManager.default
    guard fm.fileExists(atPath: directory.appending(path: "config.json").path),
      let config = try? BonsaiConfig.load(directory: directory),
      (try? config.validate()) != nil
    else { return nil }

    let names = (try? fm.contentsOfDirectory(atPath: directory.path)) ?? []
    guard names.contains(where: { $0.hasSuffix(".safetensors") }) else { return nil }

    let widths = config.quantization.widths.sorted()
    let quantization =
      widths.count == 1
      ? "\(widths[0])-bit"
      : "\(widths.map(String.init).joined(separator: "/"))-bit"

    return Entry(
      id: name(for: directory),
      format: .pack,
      url: directory,
      byteCount: MemoryBudget.weightBytes(in: directory) ?? 0,
      quantization: "\(quantization) g\(config.quantization.groupSize)",
      hasVision: config.components?.vision == true,
      hasMTP: config.components?.mtp == true,
      contextTokens: config.textConfig.maxPositionEmbeddings)
  }

  /// A HuggingFace checkout is named by its revision, which says nothing; the repo name two
  /// levels up is what a reader recognises.
  private static func name(for directory: URL) -> String {
    let parts = directory.pathComponents
    if let index = parts.lastIndex(of: "snapshots"), index > 0 {
      let repo = parts[index - 1]
      if repo.hasPrefix("models--") {
        return repo.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
      }
      return repo
    }
    return directory.lastPathComponent
  }
}
