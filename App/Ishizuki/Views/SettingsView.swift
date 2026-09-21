// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The serve flags, in panes: a toolbar to pick one, all cut to one size that fits the
// tallest, rather than one long scroll. Changes that reshape the server take a restart.

import IshizukiKit
import SwiftUI

struct SettingsView: View {
  /// Which pane is showing, remembered so the window opens where it was left: related
  /// settings tend to be adjusted more than once.
  private enum Pane: String {
    case general, model, cache, companion
  }

  @Bindable var controller: ServerController
  var companion: CompanionServer

  @AppStorage("settings.pane") private var pane = Pane.general.rawValue

  var body: some View {
    TabView(selection: $pane) {
      GeneralPane(controller: controller)
        .tabItem { Label("General", systemImage: "gearshape") }
        .tag(Pane.general.rawValue)
      ModelPane(controller: controller)
        .tabItem { Label("Model", systemImage: "cpu") }
        .tag(Pane.model.rawValue)
      CachePane(controller: controller)
        .tabItem { Label("Cache", systemImage: "archivebox") }
        .tag(Pane.cache.rawValue)
      CompanionPane(companion: companion)
        .tabItem { Label("Companion", systemImage: "iphone") }
        .tag(Pane.companion.rawValue)
    }
  }
}

/// The shape every pane takes: a grouped form at one size for all of them, cut to fit the
/// tallest. The window takes its measurements from whichever pane opened first and will not
/// grow for a taller one, so a pane left to its own height comes back clipped; one size means
/// switching panes never needs a resize that isn't coming. A pane that outgrows it — the
/// companion's list of phones, as more are let in — scrolls inside instead.
private struct PaneForm<Content: View>: View {
  @ViewBuilder var content: Content

  var body: some View {
    Form { content }
      .formStyle(.grouped)
      .scrollBounceBehavior(.basedOnSize)
      .frame(width: 480, height: 460)
  }
}

private struct GeneralPane: View {
  @Bindable var controller: ServerController
  @State private var openAtLogin = LoginItem.isEnabled
  @State private var loginItemError: String?

  private var settings: ServerSettings { controller.settings }

  var body: some View {
    PaneForm {
      Section("Endpoint") {
        TextField("Port", value: Bindable(settings).port, format: .number.grouping(.never))
        Toggle("Serve when Ishizuki opens", isOn: Bindable(settings).startOnLaunch)
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

      Section("Scheduling") {
        Picker("Politeness", selection: Bindable(settings).politeness) {
          ForEach(Politeness.Level.allCases, id: \.rawValue) { level in
            Text(level.rawValue.capitalized).tag(level.rawValue)
          }
        }
      }

      Section {
        LabeledContent("Version") {
          Text(Machine.version)
            .font(.system(.subheadline, design: .monospaced))
            .foregroundStyle(.secondary)
        }
      } footer: {
        RestartNote(controller: controller)
      }
    }
  }
}

private struct ModelPane: View {
  @Bindable var controller: ServerController

  private var settings: ServerSettings { controller.settings }

  var body: some View {
    PaneForm {
      Section("Residency") {
        Toggle("Load the pack at startup", isOn: Bindable(settings).preload)
        Toggle("Read the vision tower too", isOn: Bindable(settings).hot)
        Toggle("Prefill on the Neural Engine", isOn: Bindable(settings).neuralEngine)
          .help(
            "Splits the wide projections between the GPU and the ANE, timed so both finish "
              + "together. Only packs carrying exported slices can use it.")
        AmountSlider(
          label: "Wired memory", value: Bindable(settings).wireGB, range: 0...64,
          reading: settings.wireGB == 0 ? "off" : "\(Int(settings.wireGB)) GB")
      }

      Section("Let go when idle") {
        LabeledContent("Buffer pool") {
          TimeoutField(seconds: Bindable(settings).idleTimeout)
        }
        LabeledContent("The model itself") {
          TimeoutField(seconds: Bindable(settings).evictTimeout)
        }
      }

      Section {
        AmountSlider(
          label: "Context stretch", value: Bindable(settings).contextScale, range: 1...4,
          step: 0.5,
          reading: settings.contextScale == 1
            ? "262K" : MemoryBudget.tokens(settings.maxContextTokens),
          readingWidth: 52)
      } footer: {
        RestartNote(controller: controller)
      }
    }
  }
}

private struct CachePane: View {
  @Bindable var controller: ServerController

  private var settings: ServerSettings { controller.settings }

  var body: some View {
    PaneForm {
      Section("Keys and values") {
        Picker("KV cache", selection: Bindable(settings).kvBits) {
          Text("3.5-bit (default)").tag(3.5)
          Text("4-bit").tag(4.0)
          Text("8-bit").tag(8.0)
          Text("16-bit").tag(16.0)
        }
        TextField(
          "Unquantized window", value: Bindable(settings).kvWindow,
          format: .number.grouping(.never))
      }

      Section {
        AmountSlider(
          label: "Prefix cache on disk", value: Bindable(settings).prefixCacheGB,
          range: 0...64,
          reading: settings.prefixCacheGB == 0
            ? "off" : "\(Int(settings.prefixCacheGB)) GB")
      } footer: {
        RestartNote(controller: controller)
      }
    }
  }
}

private struct CompanionPane: View {
  var companion: CompanionServer

  var body: some View {
    PaneForm { CompanionSection(companion: companion) }
  }
}

/// Said once per pane, under the settings it applies to, rather than as a row of its own in
/// the middle of the form.
private struct RestartNote: View {
  @Bindable var controller: ServerController

  var body: some View {
    HStack {
      Text("Takes effect when the server next starts.")
      Spacer()
      Button("Restart Server") { controller.restart() }
        .controlSize(.small)
        .disabled(!controller.phase.isRunning)
    }
  }
}

/// A slider with its own value spelled out beside it, at a fixed width so the row doesn't
/// shuffle as the reading changes length.
private struct AmountSlider: View {
  let label: String
  @Binding var value: Double
  let range: ClosedRange<Double>
  var step: Double = 1
  let reading: String
  var readingWidth: CGFloat = 44

  var body: some View {
    LabeledContent(label) {
      HStack {
        Slider(value: $value, in: range, step: step)
          .accessibilityLabel(label)
          .accessibilityValue(reading)
        Text(reading)
          .font(.system(.subheadline, design: .monospaced))
          .frame(width: readingWidth, alignment: .trailing)
      }
    }
  }
}

private struct TimeoutField: View {
  @Binding var seconds: Double

  var body: some View {
    HStack {
      Slider(value: $seconds, in: 0...1800, step: 30)
      Text(seconds == 0 ? "never" : ReadoutFormat.duration(seconds))
        .font(.system(.subheadline, design: .monospaced))
        .frame(width: 60, alignment: .trailing)
    }
  }
}
