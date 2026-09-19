// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import IshizukiKit

@main
struct Ishizuki: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "ishizuki",
    abstract: "石付き — Bonsai 2 inference on MLX Swift.",
    version: BuildInfo.version,
    subcommands: [
      Demo.self, Generate.self, Serve.self, Launch.self, Pull.self, Update.self, InstallAgent.self,
      BatchCheck.self, KernelCheck.self, ANECheck.self,
      ContextBench.self, KVBench.self, PrefillBench.self, SpecBench.self, Verify.self,
      VerifyLogits.self,
    ])

  static func main() {
    Self.main(ANEOption.normalize(Array(CommandLine.arguments.dropFirst())))
  }
}
