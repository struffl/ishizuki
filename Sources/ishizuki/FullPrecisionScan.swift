// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import IshizukiKit

/// Finds checkpoints worth quantizing: the full-precision ones.
///
/// The model catalog deliberately lists only what can be served, and an unquantized checkpoint
/// cannot be, so this looks for the opposite — a config with no quantization block and weights
/// beside it.
enum FullPrecisionScan {
  struct Candidate {
    let name: String
    let directory: URL
    let byteCount: Int
    let dtype: String
  }

  static func run(in roots: [URL]) -> [Candidate] {
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
  static func describe(_ directory: URL) -> Candidate {
    inspect(directory)
      ?? Candidate(
        name: name(for: directory), directory: directory,
        byteCount: MemoryBudget.weightBytes(in: directory) ?? 0, dtype: "unknown")
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
      dtype: dtype)
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
