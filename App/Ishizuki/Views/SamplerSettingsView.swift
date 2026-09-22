// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A slider and a precise number for each sampler knob, split into the two worth touching and
// the four that mostly make answers worse. Every edit takes effect immediately, the way the
// rest of this app's settings do — there is nothing here destructive enough to need a Cancel.

import IshizukiKit
import SwiftUI

struct SamplerSettingsView: View {
  let entry: ModelCatalog.Entry
  @Bindable var controller: ServerController
  let done: () -> Void

  @State private var draft: SamplerSettings

  init(entry: ModelCatalog.Entry, controller: ServerController, done: @escaping () -> Void) {
    self.entry = entry
    self.controller = controller
    self.done = done
    _draft = State(initialValue: controller.samplerSettings.settings(for: entry.id))
  }

  var body: some View {
    VStack(spacing: 0) {
      Form {
        Section {
          SamplerRow(
            label: "Temperature", value: $draft.temperature, range: 0...2, step: 0.05,
            help: "How much randomness rides on top of the ranking. 0 always takes the top "
              + "token; higher spreads the odds across more of them.")
          SamplerRow(
            label: "Min P", value: $draft.minP, range: 0...1, step: 0.01,
            help: "Drops any token whose odds fall below this fraction of the best one — a "
              + "gentler, self-scaling stand-in for top-k or top-p.")
        } header: {
          Text("Sampling")
        }

        Section {
          ToggleableSamplerRow(
            label: "Top K", enabled: $draft.topKEnabled, value: topKValue, range: 1...200,
            step: 1, decimals: 0,
            help: "Keeps only the K most likely tokens before sampling.")
          ToggleableSamplerRow(
            label: "Top P", enabled: $draft.topPEnabled, value: $draft.topP, range: 0.05...1,
            step: 0.01,
            help: "Keeps the smallest set of tokens whose odds add up to P.")
          ToggleableSamplerRow(
            label: "Repetition Penalty", enabled: $draft.repetitionPenaltyEnabled,
            value: $draft.repetitionPenalty, range: 1...2, step: 0.01,
            help: "Scales down tokens already seen recently, more for ones seen more often.")
          ToggleableSamplerRow(
            label: "Presence Penalty", enabled: $draft.presencePenaltyEnabled,
            value: $draft.presencePenalty, range: -2...2, step: 0.05,
            help: "A flat penalty on any token that has already appeared, regardless of how "
              + "many times.")
        } header: {
          Text("Not recommended")
        } footer: {
          Text("Stacking these with Min P rarely helps. Off by default.")
        }
      }
      .formStyle(.grouped)
      .scrollBounceBehavior(.basedOnSize)

      HStack {
        Button("Reset to Defaults") { draft = .default }
          .buttonStyle(.borderless)
        Spacer()
        Button("Done", action: done)
          .keyboardShortcut(.defaultAction)
      }
      .padding([.horizontal, .bottom], 16)
      .padding(.top, 4)
    }
    .frame(width: 420, height: 480)
    .onChange(of: draft) { _, newValue in
      controller.updateSamplerSettings(newValue, for: entry.id)
    }
  }

  private var topKValue: Binding<Float> {
    Binding(
      get: { Float(draft.topK) },
      set: { draft.topK = Int($0.rounded()) })
  }
}

/// A slider with its own precise number beside it, always in effect.
private struct SamplerRow: View {
  let label: String
  @Binding var value: Float
  let range: ClosedRange<Float>
  var step: Float = 0.01
  var decimals: Int = 2
  var help: String

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack {
        Text(label).font(.system(.callout, weight: .medium))
        Spacer()
        TextField(
          label, value: $value, format: .number.precision(.fractionLength(decimals))
        )
        .labelsHidden()
        .textFieldStyle(.roundedBorder)
        .font(.system(.subheadline, design: .monospaced))
        .multilineTextAlignment(.trailing)
        .frame(width: 60)
      }
      Slider(value: $value, in: range, step: step)
        .accessibilityLabel(label)
    }
    .help(help)
    .padding(.vertical, 2)
  }
}

/// The same slider-and-number row, gated behind a switch — the value stays remembered while
/// the switch is off, so turning a knob on and off never loses what it was set to.
private struct ToggleableSamplerRow: View {
  let label: String
  @Binding var enabled: Bool
  @Binding var value: Float
  let range: ClosedRange<Float>
  var step: Float = 0.01
  var decimals: Int = 2
  var help: String

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Toggle(isOn: $enabled) {
        Text(label).font(.system(.callout, weight: .medium))
      }
      .toggleStyle(.switch)
      .controlSize(.small)
      HStack {
        Slider(value: $value, in: range, step: step)
          .disabled(!enabled)
          .accessibilityLabel(label)
        TextField(
          label, value: $value, format: .number.precision(.fractionLength(decimals))
        )
        .labelsHidden()
        .textFieldStyle(.roundedBorder)
        .font(.system(.subheadline, design: .monospaced))
        .multilineTextAlignment(.trailing)
        .frame(width: 60)
        .disabled(!enabled)
      }
    }
    .opacity(enabled ? 1 : 0.6)
    .help(help)
    .padding(.vertical, 2)
  }
}
