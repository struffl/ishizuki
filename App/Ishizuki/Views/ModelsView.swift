// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What is installed, what can be fetched, and where else to look for packs.

import AppKit
import IshizukiKit
import SwiftUI

struct ModelsView: View {
  @Bindable var controller: ServerController
  @State private var deleteTarget: ModelCatalog.Entry?

  private var library: ModelLibrary { controller.library }

  var body: some View {
    ScrollView {
      GlassEffectContainer(spacing: 12) {
        VStack(alignment: .leading, spacing: 18) {
          if !library.downloads.isEmpty {
            section("Downloading") {
              ForEach(library.downloads) { download in
                DownloadRow(download: download, library: library, controller: controller)
              }
            }
          }

          section("Installed") {
            if controller.catalog.entries.isEmpty {
              GlassCard {
                VStack(alignment: .leading, spacing: 8) {
                  Text("No packs in reach yet.")
                    .font(.callout.weight(.medium))
                  Text(
                    "Ishizuki can only read folders you hand it. If you already have packs, "
                      + "point it at them — nothing is copied."
                  )
                  .font(.system(size: 11))
                  .foregroundStyle(.secondary)
                  HStack {
                    ForEach(ModelLibrary.wellKnownRoots, id: \.label) { root in
                      Button(root.label) {
                        library.grantFolder(startingAt: root.url)
                        controller.rescan()
                      }
                      .buttonStyle(.glass)
                      .controlSize(.small)
                    }
                  }
                }
              }
            }
            ForEach(controller.catalog.entries, id: \.id) { entry in
              InstalledRow(entry: entry, controller: controller, deleteTarget: $deleteTarget)
            }
          }

          section("Available") {
            ForEach(available) { model in
              CuratedRow(model: model, library: library)
            }
          }

          section("This Mac") {
            Text(Machine.summary)
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(.secondary)
          }

          section("Folders") {
            ForEach(IshizukiPaths.searchRoots(), id: \.self) { root in
              FolderRow(url: root, removable: false, library: library)
            }
            ForEach(library.grantedFolders, id: \.self) { root in
              FolderRow(url: root, removable: true, library: library)
            }
            Button("Add Folder…") {
              library.grantFolder()
              controller.rescan()
            }
            .buttonStyle(.glass)
            .controlSize(.small)
          }
        }
        .padding(16)
      }
    }
    .scrollContentBackground(.hidden)
    .alert(item: $deleteTarget) { entry in
      Alert(
        title: Text("Delete \(entry.displayName)?"),
        message: Text("\(ReadoutFormat.gigabytes(entry.byteCount)) will be removed from disk."),
        primaryButton: .destructive(Text("Delete")) {
          try? library.delete(entry)
          controller.rescan()
        },
        secondaryButton: .cancel())
    }
  }

  private var available: [CuratedModel] {
    CuratedModel.all.filter { !$0.isInstalled(in: controller.catalog) }
  }

  @ViewBuilder private func section<Content: View>(
    _ title: String, @ViewBuilder content: () -> Content
  ) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(title)
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(.secondary)
        .textCase(.uppercase)
      content()
    }
  }
}

extension ModelCatalog.Entry: @retroactive Identifiable {}

private struct InstalledRow: View {
  let entry: ModelCatalog.Entry
  @Bindable var controller: ServerController
  @Binding var deleteTarget: ModelCatalog.Entry?

  private var isActive: Bool { controller.settings.activeModelID == entry.id }

  private var detail: String {
    var parts = [entry.format.rawValue, entry.quantization]
    parts.append(ReadoutFormat.gigabytes(entry.byteCount))
    parts.append(MemoryBudget.tokens(entry.contextTokens) + " ctx")
    if entry.hasVision { parts.append("vision") }
    if entry.hasMTP { parts.append("mtp") }
    return parts.filter { !$0.isEmpty }.joined(separator: " · ")
  }

  var body: some View {
    HStack(spacing: 10) {
      Image(systemName: isActive ? "largecircle.fill.circle" : "circle")
        .foregroundStyle(isActive ? Color.accentColor : Color.secondary)
      VStack(alignment: .leading, spacing: 2) {
        Text(entry.displayName).font(.system(size: 12, weight: .medium))
        Text(detail).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
      }
      Spacer()
      if !isActive {
        Button("Use") { controller.activate(entry.id) }.controlSize(.small)
      }
      Menu {
        Button("Reveal in Finder") {
          NSWorkspace.shared.activateFileViewerSelecting([entry.directory])
        }
        if controller.library.isManaged(entry) {
          Button("Delete…", role: .destructive) { deleteTarget = entry }
        }
      } label: {
        Image(systemName: "ellipsis.circle")
      }
      .menuStyle(.borderlessButton)
      .fixedSize()
    }
    .padding(10)
    .glassEffect(.regular, in: .rect(cornerRadius: 12))
  }
}

private struct CuratedRow: View {
  let model: CuratedModel
  @Bindable var library: ModelLibrary

  var body: some View {
    HStack(spacing: 10) {
      VStack(alignment: .leading, spacing: 2) {
        Text(model.name).font(.system(size: 12, weight: .medium))
        Text(model.summary).font(.system(size: 10)).foregroundStyle(.secondary)
      }
      Spacer()
      if !Machine.fits(weightBytes: model.bytes) {
        Label("won't fit", systemImage: "exclamationmark.triangle")
          .font(.system(size: 10))
          .foregroundStyle(.orange)
          .help(
            "This pack needs more than this Mac will keep resident — \(Machine.summary).")
      }
      Text(ReadoutFormat.gigabytes(model.bytes))
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(.tertiary)
      Button("Download") { library.download(repo: model.repo, only: model.only) }
        .buttonStyle(.glass)
        .controlSize(.small)
        .disabled(library.downloads.contains { $0.repo == model.repo })
    }
    .padding(10)
    .glassEffect(.regular, in: .rect(cornerRadius: 12))
  }
}

private struct DownloadRow: View {
  let download: ModelLibrary.Download
  @Bindable var library: ModelLibrary
  @Bindable var controller: ServerController

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack {
        Text(download.repo).font(.system(size: 11, design: .monospaced))
        Spacer()
        if let failure = download.failure {
          Text(failure).font(.system(size: 10)).foregroundStyle(.red).lineLimit(1)
          Button("Dismiss") {
            library.dismiss(repo: download.repo)
            controller.rescan()
          }
          .controlSize(.small)
        } else {
          Button("Cancel") { library.cancel(repo: download.repo) }.controlSize(.small)
        }
      }
      if download.failure == nil {
        Bar(fraction: download.fraction, tint: .accentColor, width: 320)
        Text(
          "\(download.file) — \(ReadoutFormat.compact(download.completedBytes)) "
            + "/ \(ReadoutFormat.compact(download.totalBytes))"
        )
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(.tertiary)
      }
    }
    .padding(10)
    .glassEffect(.regular, in: .rect(cornerRadius: 12))
  }
}

private struct FolderRow: View {
  let url: URL
  let removable: Bool
  @Bindable var library: ModelLibrary

  var body: some View {
    HStack(spacing: 8) {
      Image(systemName: "folder").foregroundStyle(.secondary)
      Text(url.path(percentEncoded: false))
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .truncationMode(.middle)
      Spacer()
      if removable {
        Button("Forget") { library.forget(url) }.controlSize(.small)
      }
    }
  }
}
