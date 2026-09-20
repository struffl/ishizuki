// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Where packs, prefixes and the HuggingFace cache live, for the CLI and the app alike.

import Foundation

public enum IshizukiPaths {
  public static let defaultRepo = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
  public static let defaultPack = "Ternary-Bonsai-2-27B-mlx-2bit"
  public static let defaultServedName = "ternary-bonsai-2-27b"

  /// True when the process runs in an App Sandbox container, where `$HOME` is the container
  /// and anything outside it has to be handed over by the user.
  public static var isSandboxed: Bool {
    ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] != nil
  }

  public static var applicationSupport: URL {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
      ?? URL(filePath: NSHomeDirectory()).appending(path: "Library/Application Support")
  }

  public static var models: URL { applicationSupport.appending(path: "Ishizuki/models") }

  public static var prefixCache: URL {
    applicationSupport.appending(path: "Ishizuki/cache/prefixes")
  }

  public static var defaultModelPath: String { models.appending(path: defaultPack).path }

  /// HuggingFace's own download location, or nil in a sandbox, where `$HOME` is the container
  /// and the real one can only arrive as a bookmark the user granted.
  public static var huggingFaceCache: URL? {
    guard !isSandboxed else { return nil }
    return URL(filePath: NSHomeDirectory()).appending(path: ".cache/huggingface/hub")
  }

  /// Everywhere a loadable pack might sit, in the order a duplicate should be resolved:
  /// what ishizuki manages itself first, then what HuggingFace's own tooling has pulled,
  /// then any directory the user pointed at.
  public static func searchRoots(granted: [URL] = []) -> [URL] {
    var roots = [models]
    if let cache = huggingFaceCache { roots.append(cache) }
    for url in granted where !roots.contains(url) { roots.append(url) }
    return roots
  }

  public static func resolvedModelPath(_ model: String, repo: String) -> String {
    guard model == defaultModelPath, repo != defaultRepo else { return model }
    return models.appending(path: (repo as NSString).lastPathComponent).path
  }
}
