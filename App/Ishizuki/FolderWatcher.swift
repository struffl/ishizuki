// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Noticing that a folder changed. A pack pulled from a terminal, or deleted from the Finder,
// should reach the list without anyone pressing rescan.

import Foundation

/// Watches a handful of directories and says so, once, a moment after they settle.
///
/// The roots are watched along with their immediate children, because that is where a pack
/// actually lands: writing a file inside `models/<pack>/` does not touch `models` itself, and a
/// watcher on the root alone would never hear a download finish.
@MainActor
final class FolderWatcher {
  private let queue = DispatchQueue(label: "studio.ishizuki.folders", qos: .utility)
  private var sources: [DispatchSourceFileSystemObject] = []
  private var pending: DispatchWorkItem?
  private let settle: DispatchTimeInterval
  private let onChange: @Sendable () -> Void

  /// `onChange` is called on the main queue, coalesced: unpacking a pack is hundreds of writes
  /// and one thing worth hearing about.
  init(settle: DispatchTimeInterval = .milliseconds(700), onChange: @escaping @Sendable () -> Void)
  {
    self.settle = settle
    self.onChange = onChange
  }

  func watch(_ roots: [URL]) {
    stop()
    for url in expand(roots) {
      let descriptor = open(url.path, O_EVTONLY)
      guard descriptor >= 0 else { continue }
      let source = DispatchSource.makeFileSystemObjectSource(
        fileDescriptor: descriptor,
        eventMask: [.write, .rename, .delete, .extend],
        queue: queue)
      // Both handlers are made outside the actor. A closure written here would take this
      // class's own main-actor isolation with it, and dispatch calls a source's handlers on
      // the source's queue — which is the isolation check failing, and a trap on the first
      // event or the first cancel.
      source.setEventHandler(handler: Self.firing(self))
      source.setCancelHandler(handler: Self.closing(descriptor))
      source.resume()
      sources.append(source)
    }
  }

  func stop() {
    pending?.cancel()
    pending = nil
    for source in sources { source.cancel() }
    sources = []
  }

  private func expand(_ roots: [URL]) -> [URL] {
    let fm = FileManager.default
    var out: [URL] = []
    for root in roots {
      var isDirectory: ObjCBool = false
      guard fm.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue
      else { continue }
      out.append(root)
      let children =
        (try? fm.contentsOfDirectory(
          at: root, includingPropertiesForKeys: [.isDirectoryKey],
          options: [.skipsHiddenFiles])) ?? []
      for child in children
      where (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
        out.append(child)
      }
    }
    // A cache with hundreds of checkouts is not worth a descriptor apiece; the roots alone
    // still catch a checkout arriving or leaving.
    return out.count > 256 ? roots : out
  }

  private nonisolated static func firing(_ watcher: FolderWatcher) -> @Sendable () -> Void {
    { [weak watcher] in
      Task { @MainActor in watcher?.schedule() }
    }
  }

  private nonisolated static func closing(_ descriptor: Int32) -> @Sendable () -> Void {
    { close(descriptor) }
  }

  private func schedule() {
    pending?.cancel()
    let work = DispatchWorkItem { [weak self] in
      MainActor.assumeIsolated {
        self?.pending = nil
        self?.onChange()
      }
    }
    pending = work
    DispatchQueue.main.asyncAfter(deadline: .now() + settle, execute: work)
  }
}
