// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Building packs and checking the engine — the work that used to be subcommands.

import IshizukiKit
import SwiftUI

struct ToolsView: View {
  @Bindable var controller: ServerController
  @Bindable var runner: JobRunner
  @Bindable var quantize: QuantizeController

  var body: some View {
    ScrollView {
      GlassEffectContainer(spacing: 12) {
        VStack(alignment: .leading, spacing: 18) {
          ConsoleView(runner: runner)
          quantizeSection
        }
        .padding(16)
      }
    }
    .scrollContentBackground(.hidden)
    .task { quantize.rescan(roots: controller.library.searchRoots()) }
  }

  @ViewBuilder private var quantizeSection: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Quantize")
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(.secondary)
        .textCase(.uppercase)

      GlassCard {
        if quantize.candidates.isEmpty {
          VStack(alignment: .leading, spacing: 8) {
            Text("No full-precision checkpoints in reach.")
              .font(.callout.weight(.medium))
            Text(
              "Quantizing needs an unquantized checkpoint to read. Add the folder one sits in "
                + "from the Models tab."
            )
            .font(.system(size: 11))
            .foregroundStyle(.secondary)
          }
        } else {
          VStack(alignment: .leading, spacing: 10) {
            Picker("Checkpoint", selection: $quantize.sourceName) {
              ForEach(quantize.candidates, id: \.name) { candidate in
                Text("\(candidate.name)  ·  \(ReadoutFormat.bytes(candidate.byteCount))")
                  .tag(candidate.name)
              }
            }

            Picker("Profile", selection: $quantize.profileName) {
              ForEach(QuantProfile.all, id: \.name) { profile in
                Text(String(format: "%@ — ~%.1f bpw", profile.name, profile.targetBpw))
                  .tag(profile.name)
              }
            }

            Text(quantize.profile.summary)
              .font(.system(size: 10))
              .foregroundStyle(.secondary)

            if let plan = quantize.plan() {
              VStack(alignment: .leading, spacing: 2) {
                Field(label: "estimate") {
                  Text(
                    "\(ReadoutFormat.bytes(plan.estimateBytes))"
                      + "  from \(ReadoutFormat.bytes(plan.sourceBytes))"
                  )
                  .foregroundStyle(.secondary)
                }
                Field(label: "output") {
                  Text(plan.destination.lastPathComponent).foregroundStyle(.secondary)
                }
                if plan.destinationExists {
                  Field(label: "") {
                    Label("that pack already exists", systemImage: "exclamationmark.triangle")
                      .foregroundStyle(.orange)
                  }
                }
              }
            }

            Toggle(
              "Measure activations first (slower, closer to a calibrated pack)",
              isOn: $quantize.calibrate
            )
            .font(.system(size: 11))
            Toggle("Replace an existing pack of that name", isOn: $quantize.replace)
              .font(.system(size: 11))

            HStack {
              Button("Build Pack") { quantize.start(on: runner) }
                .buttonStyle(.glassProminent)
                .disabled(runner.isRunning || quantize.plan() == nil)
              Button("Rescan") {
                quantize.rescan(roots: controller.library.searchRoots())
              }
              .buttonStyle(.glass)
            }
          }
        }
      }
    }
  }
}
