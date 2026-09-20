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
    VStack(alignment: .leading, spacing: 12) {
      header

      if let readout = controller.readout {
        VStack(alignment: .leading, spacing: 4) {
          Field(label: "decode") {
            HStack(spacing: 5) {
              Text(String(format: "%.1f", readout.totals.decodeRate)).fontWeight(.semibold)
              Text("tok/s").foregroundStyle(.secondary)
            }
          }
          Field(label: "memory") {
            HStack(spacing: 8) {
              Bar(fraction: readout.load.fraction, width: 70)
              Text(ReadoutFormat.gigabytes(readout.load.held)).foregroundStyle(.secondary)
            }
          }
          Field(label: "in flight") {
            Text("\(readout.running) running · \(readout.queued) queued")
              .foregroundStyle(.secondary)
          }
          Field(label: "up") {
            Text(ReadoutFormat.duration(readout.state.uptime)).foregroundStyle(.secondary)
          }
        }
      } else {
        Text(status)
          .font(.system(size: 11, design: .monospaced))
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
        Button(controller.phase.isRunning ? "Stop" : "Start") {
          controller.phase.isRunning ? controller.stop() : controller.start()
        }
        .buttonStyle(.glassProminent)
        .disabled(controller.phase.isBusy || controller.catalog.entries.isEmpty)
        Button("Dashboard") { openWindow(id: "dashboard") }
          .buttonStyle(.glass)
        Spacer()
        Button("Quit") { NSApplication.shared.terminate(nil) }
          .buttonStyle(.glass)
      }
    }
    .padding(14)
    .frame(width: 320)
  }

  private var modelSelection: Binding<String> {
    Binding(
      get: { controller.settings.activeModelID },
      set: { controller.activate($0) })
  }

  private var header: some View {
    HStack(spacing: 8) {
      Text(controller.readout?.modelName ?? controller.activeEntry?.displayName ?? "no model")
        .font(.system(size: 12, weight: .semibold, design: .monospaced))
        .lineLimit(1)
        .truncationMode(.middle)
      Spacer()
      if controller.phase.isRunning {
        Text("\(controller.settings.port)")
          .font(.system(size: 11, design: .monospaced))
          .foregroundStyle(.secondary)
      }
    }
  }

  private var status: String {
    switch controller.phase {
    case .stopped: "stopped"
    case .starting(let name): "loading \(name)…"
    case .running: "starting up…"
    case .failed(let message): message
    }
  }
}
