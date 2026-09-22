// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The Mac, from the phone's side: how it was reached, what it is holding, which pack it is
// serving, and how to forget it.

import IshizukiKit
import IshizukiLink
import SwiftUI

struct PhoneSettingsView: View {
  let store: LinkStore
  let chats: ChatsModel
  let local: LocalSession

  @State private var status: LinkStatus?
  @State private var forgetting = false

  var body: some View {
    NavigationStack {
      Form {
        Section("Connection") {
          LabeledContent("Mac", value: store.known?.name ?? "—")
          LabeledContent("Reached at", value: store.known?.hosts.first ?? "—")
          LabeledContent("State", value: state)
          if case .offline = store.phase {
            Button("Reconnect") { Task { await store.connect() } }
          }
        }

        if let info = store.info {
          Section("What it is running") {
            LabeledContent("Ishizuki", value: info.version)
            LabeledContent("Pack", value: info.model ?? "none")
            LabeledContent("Loaded", value: info.modelLoaded ? "yes" : "no")
            LabeledContent("Conversations", value: "\(info.chats)")
          }
        }

        if let status {
          Section("Load") {
            LabeledContent("Held", value: ReadoutFormat.compact(status.heldBytes))
            LabeledContent(
              "Context",
              value: "\(ReadoutFormat.group(status.contextTokens)) of "
                + ReadoutFormat.group(status.contextCeiling))
            if status.tokensPerSecond > 0 {
              LabeledContent(
                "Writing", value: String(format: "%.1f tok/s", status.tokensPerSecond))
            }
            if let gpu = status.gpu {
              LabeledContent("GPU", value: ReadoutFormat.percent(gpu))
            }
            if let thermal = status.thermal {
              LabeledContent("Thermal", value: thermal)
            }
          }
        }

        if let models = chats.models, !models.entries.isEmpty {
          Section("Packs on the Mac") {
            ForEach(models.entries) { entry in
              Button {
                Task { await chats.activate(model: entry.id) }
              } label: {
                HStack {
                  VStack(alignment: .leading, spacing: 2) {
                    Text(entry.name)
                    Text(
                      "\(entry.quantization ?? "—") · "
                        + ReadoutFormat.bytes(entry.sizeBytes)
                    )
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                  }
                  Spacer()
                  if entry.id == models.active {
                    Image(systemName: "checkmark")
                      .foregroundStyle(Color.reading)
                  }
                }
              }
              .buttonStyle(.plain)
            }
          }
        }

        Section("This iPhone") {
          LabeledContent(
            "On-device model", value: local.isAvailable ? "ready" : (local.blocker ?? "off"))
        }

        Section {
          Button("Forget this Mac", role: .destructive) { forgetting = true }
        }
      }
      .glassList()
      .navigationTitle(store.known?.name ?? "Mac")
      .refreshable { await refresh() }
      .task { await refresh() }
      .confirmationDialog(
        "Forget this Mac?", isPresented: $forgetting, titleVisibility: .visible
      ) {
        Button("Forget", role: .destructive) { store.forget() }
        Button("Cancel", role: .cancel) {}
      } message: {
        Text("The key and token are deleted from this iPhone. Pair again to come back.")
      }
    }
  }

  private var state: String {
    switch store.phase {
    case .unpaired: "not paired"
    case .connecting: "connecting"
    case .ready: "connected"
    case .offline(let why): why
    }
  }

  private func refresh() async {
    await store.refresh()
    await chats.refresh()
    status = try? await store.client?.status()
  }
}
