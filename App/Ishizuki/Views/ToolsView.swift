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
  @Bindable var bench: BenchController
  @Bindable var split: ExpertSplitController

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        ConsoleView(runner: runner)
        benchSection
        quantizeSection
        expertsSection
        CacheSection(controller: controller)
      }
      .padding(16)
    }
    .scrollContentBackground(.hidden)
    .task {
      quantize.rescan(roots: controller.library.searchRoots())
      split.rescan(catalog: controller.catalog)
    }
  }

  @ViewBuilder private var benchSection: some View {
    VStack(alignment: .leading, spacing: 8) {
      SectionHeader(title: "Measure")

      GlassCard {
        VStack(alignment: .leading, spacing: 10) {
          Picker("Check", selection: $bench.kind) {
            ForEach(BenchKind.allCases) { kind in
              Text(kind.title).tag(kind)
            }
          }
          Text(bench.kind.summary)
            .font(.footnote)
            .foregroundStyle(.secondary)

          if let entry = controller.activeEntry {
            Field(label: "against") {
              Text("\(entry.displayName)  ·  \(entry.format.rawValue)")
                .foregroundStyle(.secondary)
            }
            if bench.kind.needsGGUF, entry.format != .gguf {
              Field(label: "") {
                Label("select a GGUF to run this", systemImage: "exclamationmark.triangle")
                  .foregroundStyle(.orange)
              }
            }
          }

          Button("Run") {
            guard let entry = controller.activeEntry else { return }
            bench.start(
              on: runner, entry: entry, neuralEngine: controller.settings.neuralEngine)
          }
          .buttonStyle(.glassProminent)
          .disabled(runner.isRunning || !bench.canRun(controller.activeEntry))
        }
      }
    }
  }

  /// A sparse pack's routed experts are most of its weight and a sixth of its work. Moving
  /// them onto disk is the difference between a model this machine can hold and one it cannot.
  @ViewBuilder private var expertsSection: some View {
    VStack(alignment: .leading, spacing: 8) {
      SectionHeader(title: "Stream Experts")

      GlassCard {
        if split.candidates.isEmpty {
          VStack(alignment: .leading, spacing: 8) {
            Text("No packs here route through experts they still hold.")
              .font(.callout.weight(.medium))
            Text(
              "This splits a mixture-of-experts pack in two: the shared half stays in memory, "
                + "the routed experts are read off disk a few at a time. Packs already split "
                + "are not offered again."
            )
            .font(.subheadline)
            .foregroundStyle(.secondary)
          }
        } else {
          VStack(alignment: .leading, spacing: 10) {
            Picker("Pack", selection: $split.sourceID) {
              ForEach(split.candidates) { candidate in
                Text("\(candidate.id)  ·  \(ReadoutFormat.bytes(candidate.byteCount))")
                  .tag(candidate.id)
              }
            }

            if let source = split.source {
              VStack(alignment: .leading, spacing: 2) {
                Field(label: "experts") {
                  Text(
                    "\(source.expertCount) across \(source.layers) sparse layer"
                      + (source.layers == 1 ? "" : "s")
                  )
                  .foregroundStyle(.secondary)
                }
                Field(label: "resident") {
                  Text(
                    "\(ReadoutFormat.bytes(source.residentBytes)) stays in memory"
                      + "  ·  \(ReadoutFormat.bytes(source.expertBytes)) moves to disk"
                  )
                  .foregroundStyle(.secondary)
                }
                Field(label: "output") {
                  Text(source.destination.lastPathComponent).foregroundStyle(.secondary)
                }
                if source.destinationExists {
                  Field(label: "") {
                    Label("that pack already exists", systemImage: "exclamationmark.triangle")
                      .foregroundStyle(.orange)
                  }
                }
              }
            }

            Text(
              "Streaming trades speed for room: the same answers, token for token, at a "
                + "fraction of the rate. The slot budget is under Settings › Model."
            )
            .font(.footnote)
            .foregroundStyle(.secondary)

            Toggle("Replace an existing pack of that name", isOn: $split.replace)
              .font(.subheadline)

            HStack {
              Button("Split Pack") {
                split.start(on: runner) {
                  controller.rescan()
                  split.rescan(catalog: controller.catalog)
                }
              }
              .buttonStyle(.glassProminent)
              .disabled(runner.isRunning || split.source == nil)
              Button("Rescan") { split.rescan(catalog: controller.catalog) }
                .buttonStyle(.glass)
            }
          }
        }
      }
    }
  }

  @ViewBuilder private var quantizeSection: some View {
    VStack(alignment: .leading, spacing: 8) {
      SectionHeader(title: "Quantize")

      GlassCard {
        if quantize.candidates.isEmpty {
          VStack(alignment: .leading, spacing: 8) {
            Text("No full-precision checkpoints in reach.")
              .font(.callout.weight(.medium))
            Text(
              "Quantizing needs an unquantized checkpoint to read. Add the folder one sits in "
                + "from the Models tab."
            )
            .font(.subheadline)
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
              .font(.footnote)
              .foregroundStyle(.secondary)

            // A download in progress is a config and some of the weights, which is what a
            // checkpoint looks like. Saying so beats a Build button that is greyed out for
            // reasons the window keeps to itself.
            if let source = quantize.source, !source.isComplete {
              Field(label: "waiting") {
                Label(source.readiness.summary, systemImage: "arrow.down.circle")
                  .foregroundStyle(.orange)
              }
            }

            if let plan = quantize.plan() {
              VStack(alignment: .leading, spacing: 2) {
                Field(label: "estimate") {
                  Text(
                    "\(ReadoutFormat.bytes(plan.estimateBytes))"
                      + "  from \(ReadoutFormat.bytes(plan.sourceBytes))"
                  )
                  .foregroundStyle(.secondary)
                }
                if plan.engramBytes > 0 {
                  Field(label: "n-grams") {
                    Text(
                      "\(ReadoutFormat.bytes(plan.engramBytes)) carried whole"
                        + "  ·  read from disk, not held"
                    )
                    .foregroundStyle(.secondary)
                  }
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
            .font(.subheadline)
            Toggle("Replace an existing pack of that name", isOn: $quantize.replace)
              .font(.subheadline)

            HStack {
              Button("Build Pack") { quantize.start(on: runner) }
                .buttonStyle(.glassProminent)
                .disabled(
                  runner.isRunning || quantize.plan() == nil
                    || quantize.source?.isComplete == false)
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
