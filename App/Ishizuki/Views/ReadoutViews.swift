// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The dashboard's vocabulary: a labelled field, a bar, a request line.

import IshizukiKit
import SwiftUI

struct Field<Content: View>: View {
  let label: String
  @ViewBuilder var content: Content

  var body: some View {
    HStack(alignment: .firstTextBaseline, spacing: Spacing.m) {
      Text(label)
        .font(.callout)
        .foregroundStyle(.secondary)
        .frame(width: 84, alignment: .leading)
      content
        .font(.callout.monospacedDigit())
      Spacer(minLength: 0)
    }
    .padding(.vertical, 2)
  }
}

struct Bar: View {
  let fraction: Double
  var tint: Color?
  var width: CGFloat = 90

  private var level: Color {
    if let tint { return tint }
    if fraction >= 0.85 { return .red }
    if fraction >= 0.6 { return Color.clay }
    return Color.moss
  }

  var body: some View {
    let clamped = min(max(fraction, 0), 1)
    Capsule()
      .fill(.quaternary)
      .frame(width: width, height: 6)
      .overlay(alignment: .leading) {
        Capsule()
          .fill(level)
          .frame(width: width * clamped, height: 6)
      }
      // Every bar is read out in words beside it, so it would only say the number twice.
      .accessibilityHidden(true)
  }
}

struct RequestRow: View {
  let request: ServeStats.Request

  private var phaseColor: Color {
    switch request.phase {
    case .queued: Color.clay
    case .prefill: .accentColor
    case .decode: Color.moss
    case .finishing: .secondary
    }
  }

  private var progress: (fraction: Double, counts: String, rate: String)? {
    switch request.phase {
    case .prefill:
      let total = max(request.prefillTotal, 1)
      var counts =
        "\(ReadoutFormat.group(request.prefilled))/\(ReadoutFormat.group(total)) tok"
      if request.cachedTokens > 0 {
        counts += " +\(ReadoutFormat.group(request.cachedTokens)) cached"
      }
      return (
        Double(request.prefilled) / Double(total), counts,
        String(format: "%.0f tok/s", request.rate)
      )
    case .decode:
      let total = max(request.maxTokens, 1)
      let counts =
        "\(ReadoutFormat.group(request.generated))/\(ReadoutFormat.group(request.maxTokens)) tok"
      return (
        Double(request.generated) / Double(total), counts,
        String(format: "%.1f tok/s", request.rate)
      )
    case .queued, .finishing:
      return nil
    }
  }

  var body: some View {
    HStack(spacing: 9) {
      Text("#\(request.id)")
        .foregroundStyle(.tertiary)
        .frame(width: 37, alignment: .trailing)
      Text(request.api)
        .foregroundStyle(Color.moss)
        .frame(width: 73, alignment: .leading)
      Text(request.stream ? "stream" : "block")
        .foregroundStyle(.tertiary)
        .frame(width: 48, alignment: .leading)
      Text(request.phase.rawValue)
        .foregroundStyle(phaseColor)
        .frame(width: 64, alignment: .leading)

      if let progress {
        Bar(fraction: progress.fraction, tint: .accentColor, width: 70)
        Text(progress.rate)
          .frame(width: 73, alignment: .trailing)
        Text(progress.counts)
          .foregroundStyle(.tertiary)
          .frame(width: 165, alignment: .leading)
      } else {
        Spacer(minLength: 0).frame(width: 77)
        Text("").frame(width: 73)
        Text("").frame(width: 165)
      }

      Text(String(format: "%.1fs", request.elapsed))
        .foregroundStyle(.secondary)
        .frame(width: 53, alignment: .trailing)
      Spacer(minLength: 0)
    }
    .font(.callout.monospacedDigit())
    .lineLimit(1)
  }
}
