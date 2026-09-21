// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The corner cluster: how much context is spent, how far this turn's prefill has come, which
// pack is answering and how hard it is being asked to think.

import IshizukiKit
import SwiftUI

/// Two rings. The inner one is the conversation's hold on the context; the outer one fills
/// only while a turn is in flight, so the wait has something to show for itself.
struct ContextDial: View {
  var used: Int
  var ceiling: Int
  var prefill: Double?
  var decoding: Bool

  private var fraction: Double {
    ceiling > 0 ? min(1, Double(used) / Double(ceiling)) : 0
  }

  var body: some View {
    ZStack {
      Circle()
        .stroke(.white.opacity(0.08), lineWidth: 5)
        .frame(width: 44, height: 44)
      Circle()
        .trim(from: 0, to: fraction)
        .stroke(
          fraction > 0.9 ? Color.orange : Color.accentSoft,
          style: StrokeStyle(lineWidth: 5, lineCap: .round))
        .rotationEffect(.degrees(-90))
        .frame(width: 44, height: 44)

      if let prefill {
        Circle()
          .trim(from: 0, to: max(0.01, min(1, prefill)))
          .stroke(
            Color.white.opacity(0.55),
            style: StrokeStyle(lineWidth: 2, lineCap: .round))
          .rotationEffect(.degrees(-90))
          .frame(width: 56, height: 56)
      } else if decoding {
        Circle()
          .stroke(Color.white.opacity(0.22), lineWidth: 2)
          .frame(width: 56, height: 56)
      }

      Text(ReadoutFormat.percent(fraction))
        .font(.system(size: 10, weight: .medium, design: .monospaced))
        .foregroundStyle(.secondary)
    }
    .frame(width: 60, height: 60)
    .animation(.easeOut(duration: 0.2), value: fraction)
    .animation(.easeOut(duration: 0.2), value: prefill)
    .help(helpText)
  }

  private var helpText: String {
    var lines = ["\(ReadoutFormat.group(used)) of \(ReadoutFormat.group(ceiling)) tokens held"]
    if let prefill {
      lines.append("prefill \(ReadoutFormat.percent(prefill))")
    } else if decoding {
      lines.append("generating")
    }
    return lines.joined(separator: " · ")
  }
}

@available(macOS 27.0, *)
struct ChatReadoutBar: View {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController

  private var inFlight: ServeStats.Request? {
    chat.readout?.inFlight.first { $0.phase == .prefill || $0.phase == .decode }
  }

  private var prefill: Double? {
    guard let request = inFlight, request.phase == .prefill, request.prefillTotal > 0 else {
      return nil
    }
    return Double(request.prefilled) / Double(request.prefillTotal)
  }

  var body: some View {
    HStack(spacing: 12) {
      meters

      Spacer(minLength: 8)

      if controller.phase.isBusy {
        ProgressView()
          .controlSize(.small)
          .scaleEffect(0.7)
      }
      modelSwitcher
      effortDial
      ContextDial(
        used: chat.readout?.context.peakTokens ?? 0,
        ceiling: chat.readout?.context.ceilingTokens ?? 0,
        prefill: prefill,
        decoding: inFlight?.phase == .decode)
    }
    .padding(.horizontal, 12)
    .padding(.vertical, 6)
  }

  /// What a turn costs, which is the number a person actually waits on.
  @ViewBuilder private var meters: some View {
    HStack(spacing: 14) {
      if chat.meter.turns > 0 {
        reading(String(format: "%.1fs", chat.meter.averageSeconds), "per turn")
        reading("\(chat.meter.averageTokens)", "tokens")
      }
      if let request = inFlight, request.rate > 0 {
        reading(
          String(format: "%.0f/s", request.rate),
          request.phase == .prefill ? "prefill" : "decode")
      }
    }
  }

  private func reading(_ value: String, _ label: String) -> some View {
    VStack(alignment: .leading, spacing: 0) {
      Text(value)
        .font(.system(size: 11, weight: .medium, design: .monospaced))
      Text(label)
        .font(.system(size: 9))
        .foregroundStyle(.tertiary)
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
        .font(.system(size: 11, design: .monospaced))
    }
    .menuStyle(.borderlessButton)
    .fixedSize()
    .help("How long the model is asked to think")
  }
}
