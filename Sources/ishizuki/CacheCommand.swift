// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit

struct CacheCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "cache",
    abstract: "Browse, prune or clear the prefixes kept on disk between runs.")

  @Flag(name: .long, help: "Delete every archived prefix without asking.")
  var clear = false

  @Flag(name: .long, help: "Print the archives and exit, rather than browsing them.")
  var list = false

  @Option(name: .long, help: "Hold the cache to this many GB, dropping the coldest first.")
  var limitGB: Double?

  @Option(name: .long, help: "Read a cache somewhere other than the default location.")
  var directory: String?

  private var location: URL {
    directory.map { URL(filePath: $0) } ?? prefixCacheDirectory
  }

  func run() throws {
    let store = PrefixStore(directory: location)

    if clear {
      let freed = store.totalBytes
      store.removeAll()
      print("cleared \(Format.bytes(freed)) from \(location.path)")
      return
    }

    if let limitGB {
      let before = store.totalBytes
      store.setByteLimit(Int(limitGB * 1_073_741_824))
      print(
        "held to \(String(format: "%.1f", limitGB)) GB, freeing "
          + Format.bytes(before - store.totalBytes))
      return
    }

    if list || !Picker.isInteractive {
      printReport(store)
      return
    }
    try browse(store)
  }

  // MARK: - Browsing

  /// The archives as a list you can walk and delete from, because pruning a cache one entry at
  /// a time is otherwise retyping ids that exist only inside the store.
  private func browse(_ store: PrefixStore) throws {
    while true {
      let entries = store.entries().sorted { $0.lastUsed > $1.lastUsed }
      guard !entries.isEmpty else {
        print("no prefixes archived in \(location.path)")
        return
      }

      let rows =
        entries.map { entry in
          Picker.Row(title: label(entry), detail: detail(entry))
        }
        + [
          Picker.Row(
            title: "",
            detail: "",
            selectable: false),
          Picker.Row(
            title: "\(entries.count) archived · \(Format.bytes(store.totalBytes)) on disk",
            selectable: false),
        ]

      switch Picker.run(
        title: "archived prefixes", rows: rows, deletable: true)
      {
      case .delete(let index):
        guard entries.indices.contains(index) else { continue }
        store.remove(entries[index].id)
      case .chose(let index):
        guard entries.indices.contains(index) else { continue }
        describe(entries[index])
        return
      case .cancelled:
        return
      }
    }
  }

  private func describe(_ entry: PrefixStore.Entry) {
    print(Style.banner("prefix \(entry.id)"))
    print("")
    print("  " + Style.field("tokens", Style.accent("\(entry.tokens.count)")))
    print("  " + Style.field("size", Style.accent(Format.bytes(entry.byteCount))))
    print("  " + Style.field("last used", Style.accent(Format.stamp(entry.lastUsed))))
    print("")
    print(Style.faint("  \(location.appending(path: entry.id + ".safetensors").path)"))
  }

  private func printReport(_ store: PrefixStore) {
    let entries = store.entries().sorted { $0.lastUsed > $1.lastUsed }
    guard !entries.isEmpty else {
      print("no prefixes archived in \(location.path)")
      return
    }
    print(Style.banner("\(entries.count) prefix(es), \(Format.bytes(store.totalBytes))"))
    print("")
    for entry in entries {
      print("  " + Style.field(label(entry), Style.faint(detail(entry)), width: 16))
    }
    print("")
    print(Style.faint("  \(location.path)"))
  }

  private func label(_ entry: PrefixStore.Entry) -> String {
    MemoryBudget.tokens(entry.tokens.count) + " tok"
  }

  private func detail(_ entry: PrefixStore.Entry) -> String {
    "\(Format.bytes(entry.byteCount))  ·  \(Format.stamp(entry.lastUsed))"
  }
}

enum Format {
  static func bytes(_ count: Int) -> String {
    let gb = Double(count) / 1_073_741_824
    if gb >= 1 { return String(format: "%.1f GB", gb) }
    let mb = Double(count) / 1_048_576
    if mb >= 1 { return String(format: "%.0f MB", mb) }
    return String(format: "%.0f KB", Double(count) / 1024)
  }

  static func stamp(_ date: Date) -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd HH:mm"
    return formatter.string(from: date)
  }
}
