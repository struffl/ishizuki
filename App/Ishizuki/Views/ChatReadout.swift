// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The corner cluster: how much context is spent, how far this turn has come, which pack is
// answering and how hard it is being asked to think.

import IshizukiKit
import SwiftUI

/// Two rings on a disc of glass. The empty part matters as much as the full part, so both
/// tracks are drawn faintly at nought per cent and the rings grow into them.
struct ContextDial: View {
  var used: Int
  var ceiling: Int
  var prefill: Double?
  var decoding: Bool

  private var fraction: Double {
    ceiling > 0 ? min(1, Double(used) / Double(ceiling)) : 0
  }

  private var spent: Color {
    fraction > 0.9 ? .orange : .accentSoft
  }

  var body: some View {
    ZStack {
      Circle()
        .fill(.clear)
        .glassEffect(.clear, in: .circle)
        .frame(width: 46, height: 46)

      Circle()
        .stroke(.white.opacity(0.1), lineWidth: 4)
        .frame(width: 32, height: 32)
      Circle()
        .trim(from: 0, to: fraction)
        .stroke(spent, style: StrokeStyle(lineWidth: 4, lineCap: .round))
        .rotationEffect(.degrees(-90))
        .frame(width: 32, height: 32)

      // The outer track is always there so the turn's progress has somewhere to appear.
      Circle()
        .stroke(.white.opacity(0.09), lineWidth: 2.5)
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

  private var live: ServeStats.Request? {
    guard let request = chat.inFlight, request.rate > 0 else { return nil }
    return request
  }

  /// What a turn costs, which is the number a person actually waits on. Nothing to report
  /// means no plate at all, rather than an empty one sitting in the corner.
  @ViewBuilder private var meters: some View {
    if chat.meter.turns > 0 || live != nil {
      HStack(spacing: 12) {
        if chat.meter.turns > 0 {
          reading(String(format: "%.1fs", chat.meter.averageSeconds), "per turn")
          reading("\(chat.meter.averageTokens)", "tokens")
        }
        if let request = live {
          reading(
            String(format: "%.0f/s", request.rate),
            request.phase == .prefill ? "reading" : "writing")
        }
      }
      .textPlate(radius: 9, horizontal: 9, vertical: 4)
    }
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
    .disabled(controller.catalog.entries.isEmpty || chat.isResponding)
    .help(chat.isResponding ? "Finish or stop the turn before switching packs" : "")
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

  /// The model's own bubble, still being written into: a label, the dots that say it is going,
  /// and for reading the bar it is filling. Sized to its contents rather than to the window,
  /// so it reads as the next message arriving rather than as a banner.
  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      HStack(spacing: 6) {
        Text(label)
          .font(.system(size: 11, weight: .medium))
          .foregroundStyle(tint)
        AnimatedDots(size: 4, tint: tint)
        if !detail.isEmpty {
          Text(detail)
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(.secondary)
            .padding(.leading, 2)
        }
      }

      // The command as it is written, so the wait for it is not a blank one.
      if case .writingCommand = chat.activity, !chat.writingCommand.isEmpty {
        Text(chat.writingCommand)
          .font(.system(size: 10, design: .monospaced))
          .foregroundStyle(.secondary)
          .lineLimit(2)
          .frame(maxWidth: 320, alignment: .leading)
      }

      // Only reading has an end to fill towards; a bar that cannot finish is a lie.
      if case .reading = chat.activity, let request = chat.inFlight {
        ReadingBar(
          read: request.prefilled,
          total: request.prefillTotal,
          instructions: chat.systemTokens
        )
        .frame(width: 168)
      }
    }
    .glassBubble()
    .padding(.leading, 10)
    .padding(.trailing, 44)
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private var label: String {
    switch chat.activity {
    case .queued: "Queued"
    case .reading: "Reading"
    case .writing: "Writing"
    case .writingCommand: "Writing command"
    case .unknown: "Working"
    }
  }

  private var tint: Color {
    switch chat.activity {
    case .reading: .accentSoft
    case .writing, .writingCommand: .generating
    case .queued, .unknown: .secondary
    }
  }

  private var detail: String {
    guard let request = chat.inFlight else { return "" }
    switch chat.activity {
    case .reading:
      return "\(ReadoutFormat.group(request.prefilled))"
        + " / \(ReadoutFormat.group(request.prefillTotal))"
    case .writing, .writingCommand:
      return "\(ReadoutFormat.group(request.generated)) tokens"
    case .queued, .unknown:
      return ""
    }
  }
}

/// The prefill, split where the instructions end. Yellow is the part that is the same every
/// turn; blue is what this turn added. Seeing the two apart is the difference between a wait
/// that looks arbitrary and one that explains itself.
struct ReadingBar: View {
  let read: Int
  let total: Int
  let instructions: Int

  var body: some View {
    GeometryReader { frame in
      let width = frame.size.width
      let scale = total > 0 ? width / CGFloat(total) : 0
      let boundary = min(CGFloat(instructions), CGFloat(total)) * scale
      let filled = min(CGFloat(read), CGFloat(total)) * scale

      ZStack(alignment: .leading) {
        Rectangle()
          .fill(.primary.opacity(0.09))
        // Where the instructions end, marked whether or not they have been read yet.
        Rectangle()
          .fill(Color.instructing.opacity(0.18))
          .frame(width: boundary)
        HStack(spacing: 0) {
          Rectangle()
            .fill(Color.instructing)
            .frame(width: min(filled, boundary))
          Rectangle()
            .fill(Color.accentSoft)
            .frame(width: max(0, filled - boundary))
        }
      }
      .clipShape(.capsule)
    }
    .frame(height: 4)
    .animation(.easeOut(duration: 0.15), value: read)
  }
}
