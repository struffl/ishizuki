// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The Mac's files from the phone: the folders it shares, down through them, and a text file read
// a screenful at a time.

import IshizukiKit
import IshizukiLink
import SwiftUI

struct FilesView: View {
  let store: LinkStore
  let chats: ChatsModel

  var body: some View {
    NavigationStack {
      List {
        Section("Folders this Mac shares") {
          ForEach(chats.roots) { root in
            NavigationLink(root.name) {
              DirectoryView(store: store, path: root.path)
            }
          }
          if chats.roots.isEmpty {
            Text("Nothing shared yet. Choose folders in the Mac's companion settings.")
              .foregroundStyle(.secondary)
          }
        }
      }
      .glassList()
      .navigationTitle("Files")
      .refreshable { await chats.refresh() }
      .task { if chats.roots.isEmpty { await chats.refresh() } }
    }
  }
}

struct DirectoryView: View {
  let store: LinkStore
  let path: String

  @State private var listing: DirectoryListing?
  @State private var loading = false
  @State private var cached = false
  @State private var failure: String?

  var body: some View {
    List {
      if cached {
        Label("Saved copy · refreshing…", systemImage: "clock.arrow.circlepath").font(.caption)
      }
      if loading && listing == nil { ProgressView("Loading folder…") }
      if let listing {
        ForEach(listing.entries) { entry in
          if entry.isDirectory {
            NavigationLink {
              DirectoryView(store: store, path: entry.path)
            } label: {
              Label(entry.name, systemImage: "folder")
            }
          } else {
            NavigationLink {
              FileView(store: store, path: entry.path)
            } label: {
              HStack {
                Label(entry.name, systemImage: "doc.text")
                Spacer()
                Text(ReadoutFormat.bytes(entry.size))
                  .font(.system(size: 10, design: .monospaced))
                  .foregroundStyle(.tertiary)
              }
            }
          }
        }
      }
      if let failure {
        Text(failure).font(.callout).foregroundStyle(.red)
      }
    }
    .glassList()
    .navigationTitle(URL(filePath: path).lastPathComponent)
    .navigationBarTitleDisplayMode(.inline)
    .task {
      while !Task.isCancelled {
        await load()
        guard failure != nil else { return }
        do { try await Task.sleep(for: .seconds(3)) } catch { return }
      }
    }
    .refreshable { await load() }
  }

  private func load() async {
    guard !loading else { return }
    loading = true
    defer { loading = false }
    let serverID = store.known?.id
    let cache = store.cache
    if listing == nil {
      listing = await cache?.load(DirectoryListing.self, key: "directory-" + path)
      cached = listing != nil
    }
    if store.client == nil { await store.connect() }
    guard let client = store.client else {
      failure = "Mac offline. Pull to retry."
      return
    }
    do {
      let fresh = try await client.list(path)
      guard store.known?.id == serverID, !Task.isCancelled else { return }
      listing = fresh
      cached = false
      await cache?.save(fresh, key: "directory-" + path)
      failure = nil
    } catch {
      guard !Task.isCancelled else { return }
      failure = error.localizedDescription
      await store.connect()
    }
  }
}

struct FileView: View {
  let store: LinkStore
  let path: String

  @State private var slice: FileSlice?
  @State private var cached = false
  @State private var failure: String?
  @State private var loading = false

  private let page = 400

  var body: some View {
    ScrollView([.horizontal, .vertical]) {
      VStack(alignment: .leading, spacing: 0) {
        if cached {
          Label("Saved copy", systemImage: "clock.arrow.circlepath").font(.caption).padding(
            .bottom, 8)
        }
        if loading && slice == nil { ProgressView("Reading file…") }
        if let slice {
          if slice.isBinary {
            Text("This file is not text.")
              .foregroundStyle(.secondary)
              .padding()
          } else {
            ForEach(Array(slice.lines.enumerated()), id: \.offset) { offset, line in
              HStack(alignment: .top, spacing: 8) {
                Text("\(slice.start + offset)")
                  .font(.system(size: 10, design: .monospaced))
                  .foregroundStyle(.tertiary)
                  .frame(width: 44, alignment: .trailing)
                Text(line.isEmpty ? " " : line)
                  .font(.system(size: 12, design: .monospaced))
                  .textSelection(.enabled)
              }
            }
            if slice.end < slice.total {
              Button(loading ? "Reading…" : "Read \(slice.total - slice.end) more lines") {
                Task { await load(from: slice.end + 1, appending: true) }
              }
              .padding(.vertical, 12)
              .disabled(loading)
            }
          }
        }
        if let failure {
          Text(failure).font(.callout).foregroundStyle(.red).padding()
        }
      }
      .padding(12)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(.regularMaterial, in: .rect(cornerRadius: 14))
      .padding(12)
    }
    .navigationTitle(URL(filePath: path).lastPathComponent)
    .navigationBarTitleDisplayMode(.inline)
    .task {
      while !Task.isCancelled {
        await load(from: 1, appending: false)
        guard failure != nil else { return }
        do { try await Task.sleep(for: .seconds(3)) } catch { return }
      }
    }
    .refreshable { await load(from: 1, appending: false) }
  }

  private func load(from offset: Int, appending: Bool) async {
    guard !loading else { return }
    loading = true
    defer { loading = false }
    let serverID = store.known?.id
    let cache = store.cache
    if slice == nil {
      slice = await cache?.load(FileSlice.self, key: "file-" + path)
      cached = slice != nil
    }
    if store.client == nil { await store.connect() }
    guard let client = store.client else {
      failure = "Mac offline. Pull to retry."
      return
    }
    do {
      let next = try await client.read(path, offset: cached ? 1 : offset, limit: page)
      guard store.known?.id == serverID, !Task.isCancelled else { return }
      if appending, var held = slice, !cached {
        held.lines += next.lines
        held.end = next.end
        held.total = next.total
        slice = held
      } else {
        slice = next
      }
      cached = false
      if let slice { await cache?.save(slice, key: "file-" + path) }
      failure = nil
    } catch {
      guard !Task.isCancelled else { return }
      failure = error.localizedDescription
      await store.connect()
    }
  }
}
