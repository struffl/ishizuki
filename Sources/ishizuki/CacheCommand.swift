// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit

struct CacheCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "cache",
    abstract: "Show or clear the prefixes kept on disk between runs.")

  @Flag(name: .long, help: "Delete every archived prefix.")
  var clear = false

  @Option(name: .long, help: "Hold the cache to this many GB, dropping the coldest first.")
  var limitGB: Double?

  func run() throws {
    let store = PrefixStore(directory: prefixCacheDirectory)

    if clear {
      let freed = store.totalBytes
      store.removeAll()
      print("cleared \(Self.bytes(freed)) from \(prefixCacheDirectory.path)")
      return
    }

    if let limitGB {
      let before = store.totalBytes
      store.setByteLimit(Int(limitGB * 1_073_741_824))
      let after = store.totalBytes
      print("held to \(String(format: "%.1f", limitGB)) GB, freeing \(Self.bytes(before - after))")
      return
    }

    let entries = store.entries().sorted { $0.lastUsed > $1.lastUsed }
    guard !entries.isEmpty else {
      print("no prefixes archived in \(prefixCacheDirectory.path)")
      return
    }

    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd HH:mm"
    print(Style.banner("\(entries.count) prefix(es), \(Self.bytes(store.totalBytes))"))
    print("")
    for entry in entries {
      print(
        "  "
          + Style.field(
            MemoryBudget.tokens(entry.tokens.count) + " tok",
            "\(Self.bytes(entry.byteCount))  \(formatter.string(from: entry.lastUsed))"))
    }
    print("")
    print(Style.faint("  \(prefixCacheDirectory.path)"))
  }

  private static func bytes(_ count: Int) -> String {
    let gb = Double(count) / 1_073_741_824
    if gb >= 1 { return String(format: "%.1f GB", gb) }
    return String(format: "%.0f MB", Double(count) / 1_048_576)
  }
}
