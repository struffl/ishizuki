// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What the bonsai drops down: state at a glance, and the two controls worth reaching for.

import IshizukiKit
import SwiftUI

struct MenuBarView: View {
  @Bindable var controller: ServerController
  @Environment(\.openWindow) private var openWindow

  var body: some View {
    VStack(alignment: .leading, spacing: 13) {
      header

      if let readout = controller.readout {
        VStack(alignment: .leading, spacing: 4) {
          Field(label: "Decode") {
            HStack(spacing: 6) {
              Text(String(format: "%.1f", readout.totals.decodeRate)).fontWeight(.semibold)
              Text("tok/s").foregroundStyle(.secondary)
            }
          }
          Field(label: "Memory") {
            HStack(spacing: 9) {
              Bar(fraction: readout.load.fraction, width: 70)
              Text(ReadoutFormat.gigabytes(readout.load.held)).foregroundStyle(.secondary)
            }
          }
          Field(label: "In flight") {
            Text("\(readout.running) running · \(readout.queued) queued")
              .foregroundStyle(.secondary)
          }
          Field(label: "Uptime") {
            Text(ReadoutFormat.duration(readout.state.uptime)).foregroundStyle(.secondary)
          }
        }
      } else {
        Text(status)
          .font(.callout)
          .foregroundStyle(.secondary)
      }

      Divider()

      if !controller.catalog.entries.isEmpty {
        Picker("Model", selection: modelSelection) {
          ForEach(controller.catalog.entries, id: \.id) { entry in
            Text(entry.displayName).tag(entry.id)
          }
        }
        .labelsHidden()
      }

      HStack {
        if controller.phase.isRunning {
          Button("Stop") { controller.stop() }
            .buttonStyle(.glass)
        } else {
          Button("Start") { controller.start() }
            .buttonStyle(.glassProminent)
            .disabled(controller.phase.isBusy || controller.catalog.entries.isEmpty)
        }
        Button("Dashboard") { openWindow(id: "dashboard") }
          .buttonStyle(.glass)
        Spacer()
        Button("Quit") { NSApplication.shared.terminate(nil) }
          .buttonStyle(.glass)
      }
    }
    .padding(Spacing.l)
    .font(.base)
    .tint(.moss)
    .fontDesign(.serif)
    .frame(width: 352)
  }

  private var modelSelection: Binding<String> {
    Binding(
      get: { controller.settings.activeModelID },
      set: { controller.activate($0) })
  }

  private var header: some View {
    HStack(spacing: 9) {
      Text(controller.readout?.modelName ?? controller.activeEntry?.displayName ?? "No model")
        .font(.system(size: 14.5, weight: .semibold))
        .lineLimit(1)
        .truncationMode(.middle)
      Spacer()
      if controller.phase.isRunning {
        Text("\(controller.settings.port)")
          .font(.callout.monospacedDigit())
          .foregroundStyle(.secondary)
      }
    }
  }

  private var status: String {
    switch controller.phase {
    case .stopped: "Stopped"
    case .starting(let name): "Loading \(name)…"
    case .running: "Starting up…"
    case .failed(let message): message
    }
  }
}
