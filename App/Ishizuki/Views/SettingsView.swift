// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The serve flags, as a form. Changes that reshape the server take a restart.

import IshizukiKit
import SwiftUI

struct SettingsView: View {
  @Bindable var controller: ServerController
  @State private var openAtLogin = LoginItem.isEnabled
  @State private var loginItemError: String?

  private var settings: ServerSettings { controller.settings }

  var body: some View {
    Form {
      Section("Endpoint") {
        TextField("Port", value: Bindable(settings).port, format: .number.grouping(.never))
        Toggle("Start serving when Ishizuki opens", isOn: Bindable(settings).startOnLaunch)
        Toggle("Open Ishizuki at login", isOn: $openAtLogin)
          .onChange(of: openAtLogin) { _, wanted in
            do {
              try LoginItem.set(wanted)
            } catch {
              loginItemError = error.localizedDescription
              openAtLogin = LoginItem.isEnabled
            }
          }
        if let loginItemError {
          Text(loginItemError).font(.caption).foregroundStyle(.red)
        }
      }

      Section("Glass") {
        LabeledContent("Window") {
          VeilSlider(key: GlassTuning.windowKey, fallback: GlassTuning.windowDefault)
        }
        LabeledContent("Behind text") {
          VeilSlider(key: GlassTuning.plateKey, fallback: GlassTuning.plateDefault)
        }
      }

      Section("Cache") {
        Picker("KV cache", selection: Bindable(settings).kvBits) {
          Text("3.5-bit (default)").tag(3.5)
          Text("4-bit").tag(4.0)
          Text("8-bit").tag(8.0)
          Text("16-bit").tag(16.0)
        }
        TextField(
          "Unquantized window", value: Bindable(settings).kvWindow,
          format: .number.grouping(.never))
        LabeledContent("Prefix cache on disk") {
          HStack {
            Slider(value: Bindable(settings).prefixCacheGB, in: 0...64, step: 1)
            Text(settings.prefixCacheGB == 0 ? "off" : "\(Int(settings.prefixCacheGB)) GB")
              .font(.system(size: 11, design: .monospaced))
              .frame(width: 44, alignment: .trailing)
          }
        }
      }

      Section("Residency") {
        LabeledContent("Release buffer pool after") {
          TimeoutField(seconds: Bindable(settings).idleTimeout)
        }
        LabeledContent("Unload the model after") {
          TimeoutField(seconds: Bindable(settings).evictTimeout)
        }
        LabeledContent("Ask for wired memory") {
          HStack {
            Slider(value: Bindable(settings).wireGB, in: 0...64, step: 1)
            Text(settings.wireGB == 0 ? "off" : "\(Int(settings.wireGB)) GB")
              .font(.system(size: 11, design: .monospaced))
              .frame(width: 44, alignment: .trailing)
          }
        }
        Toggle("Load the pack at startup", isOn: Bindable(settings).preload)
        Toggle("Read the vision tower too", isOn: Bindable(settings).hot)
        Toggle("Prefill on the Neural Engine", isOn: Bindable(settings).neuralEngine)
          .help(
            "Splits the wide projections between the GPU and the ANE, timed so both finish "
              + "together. Only packs carrying exported slices can use it.")
      }

      Section("Scheduling") {
        Picker("Politeness", selection: Bindable(settings).politeness) {
          ForEach(Politeness.Level.allCases, id: \.rawValue) { level in
            Text(level.rawValue.capitalized).tag(level.rawValue)
          }
        }
        LabeledContent("Context stretch") {
          HStack {
            Slider(value: Bindable(settings).contextScale, in: 1...4, step: 0.5)
            Text(
              settings.contextScale == 1
                ? "262K" : MemoryBudget.tokens(settings.maxContextTokens)
            )
            .font(.system(size: 11, design: .monospaced))
            .frame(width: 52, alignment: .trailing)
          }
        }
      }

      Section {
        HStack {
          Button("Restart Server") { controller.restart() }
            .disabled(!controller.phase.isRunning)
          Text("Settings apply when the server next starts.")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        LabeledContent("Version") {
          Text(Machine.version)
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(.secondary)
        }
      }
    }
    .formStyle(.grouped)
  }
}

private struct TimeoutField: View {
  @Binding var seconds: Double

  var body: some View {
    HStack {
      Slider(value: $seconds, in: 0...1800, step: 30)
      Text(seconds == 0 ? "never" : ReadoutFormat.duration(seconds))
        .font(.system(size: 11, design: .monospaced))
        .frame(width: 60, alignment: .trailing)
    }
  }
}

/// How much of what is behind the window comes through. Kept as a slider because the right
/// answer depends on the wallpaper it is sitting on.
private struct VeilSlider: View {
  let key: String
  let fallback: Double

  @AppStorage private var veil: Double

  init(key: String, fallback: Double) {
    self.key = key
    self.fallback = fallback
    _veil = AppStorage(wrappedValue: fallback, key)
  }

  var body: some View {
    HStack {
      Slider(value: $veil, in: 0...1)
      Text(ReadoutFormat.percent(veil))
        .font(.system(size: 11, design: .monospaced))
        .frame(width: 44, alignment: .trailing)
    }
  }
}
