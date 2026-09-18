// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import IshizukiKit

@main
struct Ishizuki: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "ishizuki",
    abstract: "石付き — Bonsai 2 inference on MLX Swift.",
    subcommands: [
      Generate.self, Serve.self, Launch.self, Pull.self, InstallAgent.self, BatchCheck.self, KernelCheck.self,
      ContextBench.self, KVBench.self, SpecBench.self, Verify.self, VerifyLogits.self,
    ])
}
