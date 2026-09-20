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
    HStack(alignment: .firstTextBaseline, spacing: 10) {
      Text(label)
        .font(.system(size: 11, design: .monospaced))
        .foregroundStyle(.secondary)
        .frame(width: 62, alignment: .trailing)
      content
        .font(.system(size: 11, design: .monospaced))
      Spacer(minLength: 0)
    }
  }
}

struct Bar: View {
  let fraction: Double
  var tint: Color?
  var width: CGFloat = 90

  private var level: Color {
    if let tint { return tint }
    if fraction >= 0.85 { return .red }
    if fraction >= 0.6 { return .orange }
    return .green
  }

  var body: some View {
    let clamped = min(max(fraction, 0), 1)
    RoundedRectangle(cornerRadius: 2)
      .fill(.quaternary)
      .frame(width: width, height: 5)
      .overlay(alignment: .leading) {
        RoundedRectangle(cornerRadius: 2)
          .fill(level)
          .frame(width: width * clamped, height: 5)
      }
  }
}

struct RequestRow: View {
  let request: ServeStats.Request

  private var phaseColor: Color {
    switch request.phase {
    case .queued: .orange
    case .prefill: .accentColor
    case .decode: .green
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
    HStack(spacing: 8) {
      Text("#\(request.id)")
        .foregroundStyle(.tertiary)
        .frame(width: 34, alignment: .trailing)
      Text(request.api)
        .foregroundStyle(Color.accentColor)
        .frame(width: 66, alignment: .leading)
      Text(request.stream ? "stream" : "block")
        .foregroundStyle(.tertiary)
        .frame(width: 44, alignment: .leading)
      Text(request.phase.rawValue)
        .foregroundStyle(phaseColor)
        .frame(width: 58, alignment: .leading)

      if let progress {
        Bar(fraction: progress.fraction, tint: .accentColor, width: 70)
        Text(progress.rate)
          .frame(width: 66, alignment: .trailing)
        Text(progress.counts)
          .foregroundStyle(.tertiary)
          .frame(width: 150, alignment: .leading)
      } else {
        Spacer(minLength: 0).frame(width: 70)
        Text("").frame(width: 66)
        Text("").frame(width: 150)
      }

      Text(String(format: "%.1fs", request.elapsed))
        .foregroundStyle(.secondary)
        .frame(width: 48, alignment: .trailing)
      Spacer(minLength: 0)
    }
    .font(.system(size: 11, design: .monospaced))
    .lineLimit(1)
  }
}
