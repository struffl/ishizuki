// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The corner cluster: how much context is spent, how far this turn has come, which pack is
// answering and how hard it is being asked to think.

import IshizukiKit
import SwiftUI

/// Two rings on a disc of glass, veiled to the same lightness as the plates beside it: bare
/// glass renders as the system's own grey, which next to a veiled pill reads as a dark blot.
/// The empty part matters as much as the full part, so both tracks are drawn faintly at nought
/// per cent and the rings grow into them.
struct ContextDial: View {
  var used: Int
  var ceiling: Int
  var prefill: Double?
  var decoding: Bool

  @AppStorage(GlassTuning.plateKey) private var veil = GlassTuning.plateDefault

  private var fraction: Double {
    ceiling > 0 ? min(1, Double(used) / Double(ceiling)) : 0
  }

  private var spent: Color {
    fraction > 0.9 ? .orange : .primary.opacity(0.4)
  }

  var body: some View {
    ZStack {
      Circle()
        .fill(.clear)
        .glassEffect(.regular, in: .circle)
        .overlay { Circle().fill(.background.opacity(veil)) }
        .frame(width: 46, height: 46)

      Circle()
        .stroke(.primary.opacity(0.08), lineWidth: 4)
        .frame(width: 32, height: 32)
      Circle()
        .trim(from: 0, to: fraction)
        .stroke(spent, style: StrokeStyle(lineWidth: 4, lineCap: .round))
        .rotationEffect(.degrees(-90))
        .frame(width: 32, height: 32)

      // The outer track is always there so the turn's progress has somewhere to appear.
      Circle()
        .stroke(.primary.opacity(0.07), lineWidth: 2.5)
        .frame(width: 43, height: 43)
      if let prefill {
        Circle()
          .trim(from: 0, to: max(0.015, min(1, prefill)))
          .stroke(Color.accentSoft, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
          .rotationEffect(.degrees(-90))
          .frame(width: 43, height: 43)
      } else if decoding {
        Circle()
          .stroke(Color.generating, lineWidth: 2.5)
          .frame(width: 43, height: 43)
      }

      Text(ReadoutFormat.percent(fraction))
        .font(.system(size: 9, weight: .semibold, design: .monospaced))
        .foregroundStyle(.secondary)
    }
    .frame(width: 50, height: 50)
    .animation(.easeOut(duration: 0.2), value: fraction)
    .animation(.easeOut(duration: 0.2), value: prefill)
    .help(helpText)
  }

  private var helpText: String {
    var parts = ["\(ReadoutFormat.group(used)) of \(ReadoutFormat.group(ceiling)) tokens held"]
    if let prefill {
      parts.append("reading \(ReadoutFormat.percent(prefill))")
    } else if decoding {
      parts.append("writing")
    }
    return parts.joined(separator: " · ")
  }
}

@available(macOS 27.0, *)
struct ChatReadoutBar: View {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController

  var body: some View {
    HStack(spacing: 10) {
      meters
      Spacer(minLength: 8)
      if controller.phase.isBusy {
        ProgressView()
          .controlSize(.small)
          .scaleEffect(0.7)
      }
      HStack(spacing: 8) {
        modelSwitcher
        Divider().frame(height: 12)
        effortDial
      }
      .textPlate(radius: 9, horizontal: 9, vertical: 5)
      ContextDial(
        used: chat.readout?.context.peakTokens ?? 0,
        ceiling: chat.readout?.context.ceilingTokens ?? 0,
        prefill: chat.prefillFraction,
        decoding: chat.isGenerating)
    }
    .padding(.horizontal, 10)
    .padding(.vertical, 6)
  }

  /// What a turn costs, which is the number a person actually waits on.
  @ViewBuilder private var meters: some View {
    HStack(spacing: 12) {
      if chat.meter.turns > 0 {
        reading(String(format: "%.1fs", chat.meter.averageSeconds), "per turn")
        reading("\(chat.meter.averageTokens)", "tokens")
      }
      if let request = chat.inFlight, request.rate > 0 {
        reading(
          String(format: "%.0f/s", request.rate),
          request.phase == .prefill ? "reading" : "writing")
      }
    }
    .textPlate(radius: 9, horizontal: 9, vertical: 4)
  }

  private func reading(_ value: String, _ label: String) -> some View {
    VStack(alignment: .leading, spacing: 0) {
      Text(value)
        .font(.system(size: 11, weight: .semibold, design: .monospaced))
        .foregroundStyle(.primary)
      Text(label)
        .font(.system(size: 9))
        .foregroundStyle(.secondary)
    }
  }

  @ViewBuilder private var modelSwitcher: some View {
    Menu {
      ForEach(controller.catalog.entries, id: \.id) { entry in
        Button {
          controller.activate(entry.id)
        } label: {
          if entry.id == controller.settings.activeModelID {
            Label(entry.displayName, systemImage: "checkmark")
          } else {
            Text(entry.displayName)
          }
        }
      }
    } label: {
      Text(controller.activeEntry?.displayName ?? "No pack")
        .font(.system(size: 11))
        .lineLimit(1)
    }
    .menuStyle(.borderlessButton)
    .fixedSize()
    .disabled(controller.catalog.entries.isEmpty)
  }

  @ViewBuilder private var effortDial: some View {
    Menu {
      ForEach(ReasoningEffort.allCases, id: \.self) { level in
        Button {
          chat.effort = level
          chat.saveEffort()
        } label: {
          if level == chat.effort {
            Label(level.rawValue, systemImage: "checkmark")
          } else {
            Text(level.rawValue)
          }
        }
      }
    } label: {
      Text(chat.effort.rawValue)
        .font(.system(size: 11, weight: .medium, design: .monospaced))
    }
    .menuStyle(.borderlessButton)
    .fixedSize()
    .help("How long the model is asked to think")
  }
}

/// What the turn is doing, said plainly, with a bar that only fills for the part that has an
/// end. Generation has no bound worth showing, so it gets a colour and a count instead.
@available(macOS 27.0, *)
struct TurnStatus: View {
  @Bindable var chat: ChatController

  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      HStack(spacing: 6) {
        Text(label)
          .font(.system(size: 11, weight: .medium))
          .foregroundStyle(tint)
        AnimatedDots(size: 3.5, tint: tint)
        Spacer()
        Text(detail)
          .font(.system(size: 10, design: .monospaced))
          .foregroundStyle(.secondary)
      }

      if let fraction = chat.prefillFraction {
        ProgressView(value: fraction)
          .progressViewStyle(.linear)
          .tint(.accentSoft)
          .frame(height: 3)
      }
    }
    .textPlate(radius: 8)
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private var reading: Bool { chat.prefillFraction != nil }

  private var label: String { reading ? "Reading" : "Writing" }

  private var tint: Color { reading ? .accentSoft : .generating }

  private var detail: String {
    guard let request = chat.inFlight else { return "" }
    if reading {
      return "\(ReadoutFormat.group(request.prefilled))"
        + " / \(ReadoutFormat.group(request.prefillTotal))"
    }
    return "\(ReadoutFormat.group(request.generated)) tokens"
  }
}
