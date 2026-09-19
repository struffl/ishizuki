// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit

struct ModelsCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "models",
    abstract: "List the model packs on this machine that this runtime can load.")

  func run() throws {
    let catalog = ModelCatalog.discover(in: modelSearchRoots)
    guard !catalog.entries.isEmpty else {
      print("no loadable packs found in:")
      for root in modelSearchRoots { print("  \(root.path)") }
      return
    }
    print(Style.banner("\(catalog.entries.count) pack(s)"))
    print("")
    let width = (catalog.entries.map(\.id.count).max() ?? 20) + 2
    for entry in catalog.entries {
      print("  " + Style.field(entry.id, Style.faint(describe(entry)), width: width))
    }
  }

  static func describe(_ entry: ModelCatalog.Entry) -> String {
    var parts = [
      String(format: "%.1f GB", Double(entry.byteCount) / 1_073_741_824),
      entry.quantization,
      MemoryBudget.tokens(entry.contextTokens) + " ctx",
    ]
    if entry.hasVision { parts.append("vision") }
    if entry.hasMTP { parts.append("mtp") }
    return parts.joined(separator: " · ")
  }

  func describe(_ entry: ModelCatalog.Entry) -> String { Self.describe(entry) }
}
