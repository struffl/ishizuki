// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

/// The packs on this machine that this runtime can actually load.
///
/// A directory qualifies by having a config.json this build understands and weights beside it,
/// so a half-finished download or an unrelated checkout is not offered as a model.
public struct ModelCatalog: Sendable {
  public struct Entry: Sendable, Equatable {
    public let id: String
    public let directory: URL
    public let byteCount: Int
    public let quantization: String
    public let hasVision: Bool
    public let hasMTP: Bool
    public let contextTokens: Int

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
        guard let entry = inspect(directory) else { continue }
        // First root wins, so an explicitly managed copy outranks a cached one.
        if found[entry.id] == nil { found[entry.id] = entry }
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
      directory: directory,
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
