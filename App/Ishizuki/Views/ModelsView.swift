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
  @State private var repo = ""

  private var library: ModelLibrary { controller.library }

  /// `org/model:file.gguf` names one file out of a repo that carries every quantization of
  /// the same model side by side.
  private func pull() {
    let parts = repo.split(separator: ":", maxSplits: 1)
    guard let name = parts.first, name.contains("/") else { return }
    let files = parts.count > 1 ? [String(parts[1])] : []
    library.download(repo: String(name), only: files)
    repo = ""
  }

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: Spacing.xl) {
        if !library.downloads.isEmpty {
          section("Downloading") {
            ForEach(library.downloads) { download in
              DownloadRow(download: download, library: library, controller: controller)
            }
          }
        }

        if #available(macOS 27.0, *), !controller.offeredAppleModels.isEmpty {
          section("Apple Intelligence") {
            ForEach(controller.offeredAppleModels) { model in
              AppleModelRow(model: model, controller: controller)
            }
          }
        }

        section("Installed") {
          if controller.catalog.entries.isEmpty {
            GlassCard {
              VStack(alignment: .leading, spacing: 9) {
                Text("No packs in reach yet.")
                  .font(.body.weight(.medium))
                Text(
                  "Ishizuki can only read folders you hand it. If you already have packs, "
                    + "point it at them — nothing is copied."
                )
                .font(.callout)
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
          GlassCard {
            VStack(alignment: .leading, spacing: 7) {
              Text("Any HuggingFace repo")
                .font(.system(.body, weight: .medium))
              HStack {
                TextField("org/model", text: $repo)
                  .textFieldStyle(.roundedBorder)
                  .font(.callout)
                  .onSubmit(pull)
                Button("Download", action: pull)
                  .buttonStyle(.glass)
                  .controlSize(.small)
                  .disabled(!repo.contains("/"))
              }
              Text(
                "A pack is taken whole. For a GGUF repo add the file after a colon, "
                  + "e.g. org/model:Qwen3.8-27B-IQ3_S.gguf"
              )
              .font(.subheadline)
              .foregroundStyle(.secondary)
            }
          }
        }

        section("This Mac") {
          GlassCard {
            Text(Machine.summary)
              .font(.subheadline.monospacedDigit())
              .foregroundStyle(.secondary)
          }
        }

        section("Folders") {
          GlassCard {
            VStack(alignment: .leading, spacing: 7) {
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
              .padding(.top, 2)
            }
          }
        }
      }
      .padding(Spacing.xl)
      .frame(maxWidth: 820).frame(maxWidth: .infinity)
    }
    .scrollContentBackground(.hidden)
    .alert(
      "Delete this pack?", isPresented: deleting, presenting: deleteTarget
    ) { entry in
      Button("Delete", role: .destructive) {
        try? library.delete(entry)
        controller.rescan()
      }
      Button("Cancel", role: .cancel) {}
    } message: { entry in
      Text(
        "\(entry.displayName) — \(ReadoutFormat.gigabytes(entry.byteCount)) "
          + "will be removed from disk.")
    }
  }

  private var deleting: Binding<Bool> {
    Binding(get: { deleteTarget != nil }, set: { if !$0 { deleteTarget = nil } })
  }

  private var available: [CuratedModel] {
    CuratedModel.all.filter { !$0.isInstalled(in: controller.catalog) }
  }

  @ViewBuilder private func section<Content: View>(
    _ title: String, @ViewBuilder content: () -> Content
  ) -> some View {
    VStack(alignment: .leading, spacing: Spacing.s) {
      SectionHeader(title: title)
      content()
    }
  }
}

@available(macOS 27.0, *)
private struct AppleModelRow: View {
  let model: AppleFoundationModel
  @Bindable var controller: ServerController
  @State private var settingsShown = false

  private var isActive: Bool { controller.settings.appleModel == model }

  var body: some View {
    HStack(spacing: 11) {
      Button {
        controller.settings.appleModel = model
      } label: {
        Image(systemName: isActive ? "largecircle.fill.circle" : "circle")
          .foregroundStyle(isActive ? Color.moss : Color.secondary)
          .contentShape(.circle)
      }
      .buttonStyle(.plain)
      .disabled(isActive)
      .accessibilityLabel(isActive ? "\(model.displayName), in use" : "Use \(model.displayName)")
      .help(isActive ? "This model is answering" : "Answer with this model")
      VStack(alignment: .leading, spacing: 2) {
        Text(model.displayName).font(.system(.body, weight: .medium))
        Text(AppleModelVariant.subtitle(for: model))
          .font(.subheadline.monospacedDigit())
          .foregroundStyle(.secondary)
      }
      Spacer()
      if !isActive {
        Button("Use") { controller.settings.appleModel = model }.controlSize(.small)
      }
      Button {
        settingsShown = true
      } label: {
        Image(systemName: "slider.horizontal.3")
          .hitTarget()
      }
      .buttonStyle(.plain)
      .accessibilityLabel("Settings for \(model.displayName)")
    }
    .padding(.horizontal, Spacing.m)
    .padding(.vertical, Spacing.s + 2)
    .paperCard()
    .overlay {
      RoundedRectangle(cornerRadius: 13)
        .strokeBorder(Color.hairline, lineWidth: 1)
    }
    .sheet(isPresented: $settingsShown) {
      AppleModelSettingsView(model: model, controller: controller) { settingsShown = false }
    }
  }
}

private struct InstalledRow: View {
  let entry: ModelCatalog.Entry
  @Bindable var controller: ServerController
  @Binding var deleteTarget: ModelCatalog.Entry?
  @State private var samplerSettings = false

  private var isActive: Bool {
    controller.settings.appleModel == nil && controller.settings.activeModelID == entry.id
  }

  private var detail: String {
    var parts = [entry.format.rawValue, entry.quantization]
    parts.append(ReadoutFormat.gigabytes(entry.byteCount))
    // Most of one of these packs is a table read a row at a time, so the size on disk says
    // very little about what it will cost to run. Say which part never becomes resident.
    if entry.streamedBytes > 0 {
      parts.append(ReadoutFormat.gigabytes(entry.streamedBytes) + " streamed")
    }
    parts.append(MemoryBudget.tokens(entry.contextTokens) + " ctx")
    if entry.hasVision { parts.append("vision") }
    if entry.hasMTP { parts.append("mtp") }
    return parts.filter { !$0.isEmpty }.joined(separator: " · ")
  }

  var body: some View {
    HStack(spacing: 11) {
      // The dot is what the eye reads as "this one", so it is what the hand goes for. It
      // switches packs exactly as the button on the right does; the button stays because a
      // row with only a dot to click does not look like it can be clicked.
      Button {
        controller.activate(entry.id)
      } label: {
        Image(systemName: isActive ? "largecircle.fill.circle" : "circle")
          .foregroundStyle(isActive ? Color.moss : Color.secondary)
          .contentShape(.circle)
      }
      .buttonStyle(.plain)
      .disabled(isActive)
      .accessibilityLabel(isActive ? "\(entry.displayName), in use" : "Use \(entry.displayName)")
      .help(isActive ? "This pack is answering" : "Answer with this pack")
      VStack(alignment: .leading, spacing: 2) {
        Text(entry.displayName).font(.system(.body, weight: .medium))
        Text(detail).font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
      }
      Spacer()
      if !isActive {
        Button("Use") { controller.activate(entry.id) }.controlSize(.small)
      }
      Menu {
        Button("Sampler Settings…") { samplerSettings = true }
        Divider()
        Button("Reveal in Finder") {
          NSWorkspace.shared.activateFileViewerSelecting([entry.directory])
        }
        if controller.library.isManaged(entry) {
          Button("Delete…", role: .destructive) { deleteTarget = entry }
        }
      } label: {
        Image(systemName: "ellipsis.circle")
          .hitTarget()
      }
      .menuStyle(.borderlessButton)
      .fixedSize()
      .accessibilityLabel("More actions for \(entry.displayName)")
    }
    .padding(.horizontal, Spacing.m)
    .padding(.vertical, Spacing.s + 2)
    .paperCard()
    .overlay {
      RoundedRectangle(cornerRadius: 13)
        .strokeBorder(Color.hairline, lineWidth: 1)
    }
    .sheet(isPresented: $samplerSettings) {
      SamplerSettingsView(entry: entry, controller: controller) { samplerSettings = false }
    }
  }
}

private struct CuratedRow: View {
  let model: CuratedModel
  @Bindable var library: ModelLibrary

  var body: some View {
    HStack(spacing: 11) {
      VStack(alignment: .leading, spacing: 2) {
        Text(model.name).font(.system(.body, weight: .medium))
        Text(model.summary).font(.subheadline).foregroundStyle(.secondary)
      }
      Spacer()
      if !Machine.fits(weightBytes: model.bytes) {
        Label("won't fit", systemImage: "exclamationmark.triangle")
          .font(.subheadline)
          .foregroundStyle(Color.clay)
          .help(
            "This pack needs more than this Mac will keep resident — \(Machine.summary).")
      }
      Text(ReadoutFormat.gigabytes(model.bytes))
        .font(.subheadline.monospacedDigit())
        .foregroundStyle(.tertiary)
      Button("Download") { library.download(repo: model.repo, only: model.only) }
        .buttonStyle(.glass)
        .controlSize(.small)
        .disabled(library.downloads.contains { $0.repo == model.repo })
    }
    .padding(.horizontal, Spacing.m)
    .padding(.vertical, Spacing.s + 2)
    .paperCard()
    .overlay {
      RoundedRectangle(cornerRadius: 13)
        .strokeBorder(Color.hairline, lineWidth: 1)
    }
  }
}

private struct DownloadRow: View {
  let download: ModelLibrary.Download
  @Bindable var library: ModelLibrary
  @Bindable var controller: ServerController

  var body: some View {
    VStack(alignment: .leading, spacing: 7) {
      HStack {
        Text(download.repo).font(.callout)
        Spacer()
        if let failure = download.failure {
          Text(failure).font(.subheadline).foregroundStyle(.red).lineLimit(1)
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
        .font(.subheadline.monospacedDigit())
        .foregroundStyle(.tertiary)
      }
    }
    .padding(.horizontal, Spacing.m)
    .padding(.vertical, Spacing.s + 2)
    .paperCard()
    .overlay {
      RoundedRectangle(cornerRadius: 13)
        .strokeBorder(Color.hairline, lineWidth: 1)
    }
  }
}

private struct FolderRow: View {
  let url: URL
  let removable: Bool
  @Bindable var library: ModelLibrary

  var body: some View {
    HStack(spacing: 9) {
      Image(systemName: "folder").foregroundStyle(.secondary)
      Text(url.path(percentEncoded: false))
        .font(.subheadline)
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
