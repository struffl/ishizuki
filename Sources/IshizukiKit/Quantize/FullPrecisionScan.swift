// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

/// Finds checkpoints worth quantizing: the full-precision ones.
///
/// The model catalog deliberately lists only what can be served, and an unquantized checkpoint
/// cannot be, so this looks for the opposite — a config with no quantization block and weights
/// beside it.
public enum FullPrecisionScan {
  public struct Candidate: Sendable {
    public let name: String
    public let directory: URL
    public let byteCount: Int
    public let dtype: String
    /// Whether all of it is actually here. A download in progress looks exactly like a
    /// checkpoint otherwise — a config, a name, and some of the weights — and offering it as
    /// something to quantize wastes however long it takes to reach the first gap.
    public let readiness: Readiness

    public var isComplete: Bool { readiness == .ready }
  }

  /// What a directory is still waiting for.
  public enum Readiness: Sendable, Equatable {
    case ready
    /// Shards are here but the index that names them is not, so nothing can be opened yet.
    /// This is the state a fresh `hf download` sits in for most of its run.
    case indexMissing(have: Int)
    case shardsMissing(have: Int, want: Int)

    public var summary: String {
      switch self {
      case .ready: ""
      case .indexMissing(let have):
        "still downloading — \(have) shard\(have == 1 ? "" : "s") here, "
          + "and the index that names the rest has not arrived"
      case .shardsMissing(let have, let want):
        "still downloading — \(have) of \(want) shards here"
      }
    }
  }

  public static func run(in roots: [URL]) -> [Candidate] {
    var found: [String: Candidate] = [:]
    for root in roots {
      for directory in candidates(under: root) {
        guard let candidate = inspect(directory) else { continue }
        if found[candidate.name] == nil { found[candidate.name] = candidate }
      }
    }
    return found.values.sorted { $0.name < $1.name }
  }

  private static func candidates(under root: URL) -> [URL] {
    let fm = FileManager.default
    var result: [URL] = [root]
    guard
      let children = try? fm.contentsOfDirectory(
        at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
    else { return result }
    for child in children {
      guard (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
      else { continue }
      result.append(child)
      if let revisions = try? fm.contentsOfDirectory(
        at: child.appending(path: "snapshots"), includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles])
      {
        result.append(contentsOf: revisions)
      }
    }
    return result
  }

  /// A path the user named directly, described the same way a discovered one is.
  public static func describe(_ directory: URL) -> Candidate {
    inspect(directory)
      ?? Candidate(
        name: name(for: directory), directory: directory,
        byteCount: MemoryBudget.weightBytes(in: directory) ?? 0, dtype: "unknown",
        readiness: .ready)
  }

  /// How much of a sharded checkpoint has arrived.
  ///
  /// The index names every file the weights are spread over, so what is missing is what it
  /// asks for and the directory does not have. When the index itself has not arrived there is
  /// nothing to compare against — and a directory holding `model-00001.safetensors` with no
  /// index cannot be opened at all, so that is the more incomplete state, not a safer one.
  /// One file called `model.safetensors` needs no index and is whole once it is readable.
  static func readiness(_ directory: URL, present names: [String]) -> Readiness {
    let shards = names.filter { $0.hasSuffix(".safetensors") }
    let index = directory.appending(path: "model.safetensors.index.json")
    guard let raw = try? Data(contentsOf: index),
      let payload = try? JSONSerialization.jsonObject(with: raw) as? [String: Any],
      let map = payload["weight_map"] as? [String: String]
    else {
      return shards == ["model.safetensors"] ? .ready : .indexMissing(have: shards.count)
    }
    let wanted = Set(map.values)
    let here = wanted.intersection(shards)
    return here.count == wanted.count
      ? .ready : .shardsMissing(have: here.count, want: wanted.count)
  }

  private static func inspect(_ directory: URL) -> Candidate? {
    let fm = FileManager.default
    let configURL = directory.appending(path: "config.json")
    guard fm.fileExists(atPath: configURL.path),
      let raw = try? Data(contentsOf: configURL),
      let config = try? JSONSerialization.jsonObject(with: raw) as? [String: Any]
    else { return nil }

    // Already quantized: not a source.
    guard config["quantization"] == nil, config["quantization_config"] == nil else { return nil }

    let names = (try? fm.contentsOfDirectory(atPath: directory.path)) ?? []
    guard names.contains(where: { $0.hasSuffix(".safetensors") }) else { return nil }

    let text = (config["text_config"] as? [String: Any]) ?? config
    let dtype =
      (text["dtype"] as? String) ?? (config["dtype"] as? String)
      ?? (config["torch_dtype"] as? String) ?? "unknown"

    return Candidate(
      name: name(for: directory),
      directory: directory,
      byteCount: MemoryBudget.weightBytes(in: directory) ?? 0,
      dtype: dtype,
      readiness: readiness(directory, present: names))
  }

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
