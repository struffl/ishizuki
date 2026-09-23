// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The prefixes kept between runs: what is archived, and getting rid of it.

import IshizukiKit
import SwiftUI

struct CacheSection: View {
  @Bindable var controller: ServerController
  @State private var entries: [PrefixStore.Entry] = []
  @State private var totalBytes = 0
  @State private var sharedBytes = 0
  @State private var conversations = 0
  @State private var confirmingClear = false

  var body: some View {
    VStack(alignment: .leading, spacing: Spacing.s) {
      SectionHeader(title: "Prefix cache")

      GlassCard {
        VStack(alignment: .leading, spacing: 11) {
          HStack(spacing: 9) {
            Text(
              entries.isEmpty
                ? "Nothing archived yet."
                : "\(entries.count) archived · \(ReadoutFormat.bytes(totalBytes)) on disk"
            )
            .font(.callout.monospacedDigit())
            .foregroundStyle(.secondary)
            Spacer()
            Button("Reload") { reload() }
              .buttonStyle(.glass)
              .controlSize(.small)
            Button("Clear All") { confirmingClear = true }
              .buttonStyle(.glass)
              .controlSize(.small)
              .disabled(entries.isEmpty)
          }

          if !entries.isEmpty {
            Text(
              "\(ReadoutFormat.bytes(sharedBytes)) shared by every conversation · "
                + "\(ReadoutFormat.bytes(totalBytes - sharedBytes)) across \(conversations) "
                + (conversations == 1 ? "conversation" : "conversations")
            )
            .font(.subheadline.monospacedDigit())
            .foregroundStyle(.tertiary)
            .help("The instructions and tool schemas every conversation starts from are kept once")
          }

          ForEach(entries.prefix(12), id: \.id) { entry in
            HStack(spacing: 11) {
              Text(MemoryBudget.tokens(entry.tokens.count) + " tok")
                .frame(width: 88, alignment: .leading)
              Text(ReadoutFormat.bytes(entry.byteCount))
                .foregroundStyle(.secondary)
                .frame(width: 77, alignment: .leading)
              Text(entry.lastUsed.formatted(date: .abbreviated, time: .shortened))
                .foregroundStyle(.tertiary)
              Spacer()
              Button {
                _ = controller.prefixStore.remove(entry.id)
                reload()
              } label: {
                Image(systemName: "trash")
                  .hitTarget()
              }
              .buttonStyle(.borderless)
              .accessibilityLabel("Forget this prefix")
              .help("Forget this prefix")
            }
            .font(.subheadline.monospacedDigit())
          }

          if entries.count > 12 {
            Text("+\(entries.count - 12) more")
              .font(.subheadline.monospacedDigit())
              .foregroundStyle(.tertiary)
          }
        }
      }
    }
    .task { reload() }
    .alert("Clear every archived prefix?", isPresented: $confirmingClear) {
      Button("Clear", role: .destructive) {
        controller.prefixStore.removeAll()
        reload()
      }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text("\(ReadoutFormat.bytes(totalBytes)) will be freed. Sessions re-prefill instead.")
    }
  }

  private func reload() {
    let store = controller.prefixStore
    entries = store.entries().sorted { $0.lastUsed > $1.lastUsed }
    totalBytes = store.totalBytes
    sharedBytes = entries.filter { $0.tag == nil }.reduce(0) { $0 + $1.byteCount }
    conversations = Set(entries.compactMap(\.tag)).count
  }
}
