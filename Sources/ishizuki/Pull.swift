// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit

struct Pull: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "pull",
    abstract: "Download or repair the model pack from HuggingFace.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long, help: "HuggingFace repo id.") var repo: String = defaultRepo
  @Option(name: .long, help: "Repo revision: branch, tag, or commit.") var revision = "main"
  @Option(name: .long, help: "HuggingFace token. Falls back to $HF_TOKEN.") var token: String?
  @Flag(name: .long, help: "Verify every file's sha256 (slower).") var verify = false
  /// A GGUF repo holds every quantization of the model side by side, so pulling the tree would
  /// fetch a dozen copies to use one.
  @Option(name: .long, help: "Take only this file from the repo. Repeatable.")
  var file: [String] = []

  func run() throws {
    try ModelDownloader.ensure(
      directory: URL(filePath: resolvedModelPath(model, repo: repo)), repo: repo,
      revision: revision, token: token, verify: verify, only: file)
  }
}
