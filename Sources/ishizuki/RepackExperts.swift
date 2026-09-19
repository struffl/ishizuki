// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit

struct RepackExperts: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "repack-experts",
    abstract: "Split a sparse pack into resident weights and per-layer expert files.")

  @Option(name: .long, help: "The MoE pack to split.") var model: String
  @Option(name: .long, help: "Where to write the streamed copy.") var output: String

  func run() throws {
    let start = Date()
    let plan = try ExpertRepack.run(
      source: URL(filePath: model), destination: URL(filePath: output),
      log: { FileHandle.standardError.write(Data(($0 + "\r").utf8)) })

    print("")
    print(Style.banner("split \(plan.layers.count) sparse layer(s)"))
    print("")
    print(
      String(
        format: "  resident   %.2f GB", Double(plan.residentBytes) / 1_073_741_824))
    print(String(format: "  experts    %.2f GB", Double(plan.expertBytes) / 1_073_741_824))
    print("  per expert \(plan.layout.stride) bytes, \(plan.layout.expertCount) per layer")
    print(String(format: "  took       %.0fs", -start.timeIntervalSinceNow))
  }
}
