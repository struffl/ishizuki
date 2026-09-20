// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Packs the app can reach: its own downloads, plus any folder the user handed it.

import AppKit
import Foundation
import IshizukiKit
import Observation

@MainActor
@Observable
final class ModelLibrary {
  struct Download: Identifiable {
    var id: String { repo }
    var repo: String
    var file = ""
    var completedBytes = 0
    var totalBytes = 0
    var failure: String?

    var fraction: Double {
      totalBytes > 0 ? min(max(Double(completedBytes) / Double(totalBytes), 0), 1) : 0
    }
  }

  private(set) var grantedFolders: [URL] = []
  private(set) var downloads: [Download] = []

  /// Read from the URLSession thread that is writing the file, so it cannot live on the actor.
  private final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flagged = false

    var isSet: Bool {
      lock.lock()
      defer { lock.unlock() }
      return flagged
    }

    func set() {
      lock.lock()
      flagged = true
      lock.unlock()
    }
  }

  private let bookmarkKey = "grantedFolderBookmarks"
  private var flags: [String: CancelFlag] = [:]

  init() {
    restoreGrantedFolders()
  }

  func searchRoots() -> [URL] {
    IshizukiPaths.searchRoots(granted: grantedFolders)
  }

  // MARK: folders the user lends us

  /// The account's home, not the sandbox container that `NSHomeDirectory` reports — the packs
  /// worth pointing at live out there.
  static var realHome: URL {
    guard let entry = getpwuid(getuid()), let dir = entry.pointee.pw_dir else {
      return URL(filePath: NSHomeDirectory())
    }
    return URL(filePath: String(cString: dir))
  }

  static let wellKnownRoots: [(label: String, url: URL)] = [
    ("ishizuki models", realHome.appending(path: "Library/Application Support/Ishizuki/models")),
    ("HuggingFace cache", realHome.appending(path: ".cache/huggingface/hub")),
  ]

  /// A sandboxed app cannot see `~/.cache/huggingface` on its own; the user points at it once
  /// and the bookmark is what survives the next launch.
  func grantFolder(startingAt suggestion: URL? = nil) {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    panel.prompt = "Use Folder"
    panel.message = "Choose a folder holding model packs — a HuggingFace cache, say."
    panel.directoryURL = suggestion ?? Self.wellKnownRoots.last?.url
    guard panel.runModal() == .OK, let url = panel.url else { return }
    guard
      let bookmark = try? url.bookmarkData(
        options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
    else { return }

    var stored = UserDefaults.standard.array(forKey: bookmarkKey) as? [Data] ?? []
    stored.append(bookmark)
    UserDefaults.standard.set(stored, forKey: bookmarkKey)
    _ = url.startAccessingSecurityScopedResource()
    if !grantedFolders.contains(url) { grantedFolders.append(url) }
  }

  func forget(_ url: URL) {
    url.stopAccessingSecurityScopedResource()
    grantedFolders.removeAll { $0 == url }
    let remaining = grantedFolders.compactMap {
      try? $0.bookmarkData(
        options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
    }
    UserDefaults.standard.set(remaining, forKey: bookmarkKey)
  }

  private func restoreGrantedFolders() {
    let stored = UserDefaults.standard.array(forKey: bookmarkKey) as? [Data] ?? []
    for bookmark in stored {
      var stale = false
      guard
        let url = try? URL(
          resolvingBookmarkData: bookmark, options: .withSecurityScope,
          relativeTo: nil, bookmarkDataIsStale: &stale),
        url.startAccessingSecurityScopedResource()
      else { continue }
      grantedFolders.append(url)
    }
  }

  // MARK: fetching

  func download(repo: String, only: [String] = []) {
    guard !downloads.contains(where: { $0.repo == repo }) else { return }
    downloads.append(Download(repo: repo, totalBytes: 0))
    let flag = CancelFlag()
    flags[repo] = flag

    let directory = IshizukiPaths.models.appending(path: (repo as NSString).lastPathComponent)
    Task.detached(priority: .utility) {
      do {
        try ModelDownloader.ensure(
          directory: directory,
          repo: repo,
          only: only,
          log: { _ in },
          progress: { step in
            Task { @MainActor [weak self] in self?.record(repo: repo, step: step) }
          },
          isCancelled: { flag.isSet })
        await MainActor.run { self.finish(repo: repo, failure: nil) }
      } catch is ModelDownloader.Cancelled {
        await MainActor.run { self.finish(repo: repo, failure: nil) }
      } catch {
        await MainActor.run { self.finish(repo: repo, failure: String(describing: error)) }
      }
    }
  }

  func cancel(repo: String) {
    flags[repo]?.set()
  }

  func delete(_ entry: ModelCatalog.Entry) throws {
    try FileManager.default.removeItem(at: entry.directory)
  }

  func isManaged(_ entry: ModelCatalog.Entry) -> Bool {
    entry.directory.path.hasPrefix(IshizukiPaths.models.path)
  }

  private func record(repo: String, step: ModelDownloader.Progress) {
    guard let index = downloads.firstIndex(where: { $0.repo == repo }) else { return }
    downloads[index].file = (step.file as NSString).lastPathComponent
    downloads[index].completedBytes = step.completedBytes
    downloads[index].totalBytes = step.totalBytes
  }

  private func finish(repo: String, failure: String?) {
    guard let index = downloads.firstIndex(where: { $0.repo == repo }) else { return }
    if let failure {
      downloads[index].failure = failure
    } else {
      downloads.remove(at: index)
    }
    flags[repo] = nil
  }

  func dismiss(repo: String) {
    downloads.removeAll { $0.repo == repo }
  }
}
