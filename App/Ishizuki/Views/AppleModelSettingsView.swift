// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The override sheet for one of Apple's own models: not sampler knobs, since those mean
// nothing to a model this app never quantized, but the two things OS 27 does let a turn ask
// for — how hard Private Cloud Compute thinks, and how tightly the on-device model guards
// itself. Kept in the same settings table as every pack's sampler knobs, keyed the same way.

import FoundationModels
import IshizukiKit
import SwiftUI

@available(macOS 27.0, *)
struct AppleModelSettingsView: View {
  let model: AppleFoundationModel
  @Bindable var controller: ServerController
  let done: () -> Void

  @State private var draft: SamplerSettings

  init(model: AppleFoundationModel, controller: ServerController, done: @escaping () -> Void) {
    self.model = model
    self.controller = controller
    self.done = done
    _draft = State(initialValue: controller.samplerSettings.settings(for: model.id))
  }

  var body: some View {
    VStack(spacing: 0) {
      Form {
        switch model {
        case .privateCloudCompute:
          Section {
            Picker("Reasoning", selection: reasoningLevel) {
              ForEach(AppleReasoningLevel.allCases, id: \.self) { level in
                Text(level.displayName).tag(level)
              }
            }
            .pickerStyle(.segmented)
          } header: {
            Text("Reasoning Level")
          } footer: {
            Text(
              "Light answers faster; deep spends more of Apple's own compute for a better "
                + "answer. Chosen fresh on every turn.")
          }
          Section {
            LabeledContent("Availability", value: availabilityText)
            if let usage = quotaText {
              LabeledContent("Quota", value: usage)
            }
          } header: {
            Text("Private Cloud Compute")
          } footer: {
            Text(
              "Free with no per-token cost, under a quota this Mac does not control. No "
                + "account, key or sign-in of its own.")
          }
        case .onDevice:
          Section {
            Picker("Guardrails", selection: guardrails) {
              ForEach(AppleGuardrails.allCases, id: \.self) { level in
                Text(level.displayName).tag(level)
              }
            }
            .pickerStyle(.segmented)
          } header: {
            Text("Guardrails")
          } footer: {
            Text(
              "Permissive loosens the refusals a chat assistant would otherwise make on "
                + "mature or sensitive material. Standard is Apple's own default posture.")
          }
          Section {
            LabeledContent("Availability", value: availabilityText)
          } header: {
            Text("On-Device")
          } footer: {
            Text("Runs on this Mac with no network at all, at no cost.")
          }
        }
      }
      .formStyle(.grouped)
      .scrollBounceBehavior(.basedOnSize)

      HStack {
        Spacer()
        Button("Done", action: done)
          .keyboardShortcut(.defaultAction)
      }
      .padding([.horizontal, .bottom], 18)
      .padding(.top, 4)
    }
    .frame(width: 462, height: model == .privateCloudCompute ? 360 : 300)
    .onChange(of: draft) { _, newValue in
      controller.updateSamplerSettings(newValue, for: model.id)
    }
  }

  private var reasoningLevel: Binding<AppleReasoningLevel> {
    Binding(
      get: { draft.resolvedAppleReasoningLevel },
      set: { draft.appleReasoningLevel = $0 })
  }

  private var guardrails: Binding<AppleGuardrails> {
    Binding(
      get: { draft.resolvedAppleGuardrails },
      set: { draft.appleGuardrails = $0 })
  }

  @available(macOS 27.0, *)
  private var availabilityText: String {
    switch model {
    case .onDevice:
      switch SystemLanguageModel.default.availability {
      case .available: return "Available"
      case .unavailable(.deviceNotEligible): return "This Mac is not eligible"
      case .unavailable(.appleIntelligenceNotEnabled): return "Apple Intelligence is off"
      case .unavailable(.modelNotReady): return "Still downloading"
      @unknown default: return "Unavailable"
      }
    case .privateCloudCompute:
      switch PrivateCloudComputeLanguageModel().availability {
      case .available: return "Available"
      case .unavailable(.deviceNotEligible): return "This Mac is not eligible"
      case .unavailable(.systemNotReady): return "Not ready yet"
      @unknown default: return "Unavailable"
      }
    }
  }

  @available(macOS 27.0, *)
  private var quotaText: String? {
    guard model == .privateCloudCompute else { return nil }
    switch PrivateCloudComputeLanguageModel().quotaUsage.status {
    case .belowLimit(let below):
      return below.isApproachingLimit ? "Below limit, close to it" : "Below limit"
    case .limitReached:
      return "Limit reached"
    @unknown default:
      return "Unavailable"
    }
  }
}
