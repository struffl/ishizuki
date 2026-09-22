// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One row of a transcript on a phone: what was asked, what was thought, what was run, and what
// came back. The prose is drawn by the same markdown views the Mac uses.

import IshizukiKit
import IshizukiLink
import SwiftUI

struct TranscriptRowView: View {
  let row: TranscriptRow
  /// True for the last row while a turn is still running, which is what earns a fading edge.
  var live = false

  @State private var open: Bool?

  private let size = 14.0
  private var mono: Font { .system(size: 12, design: .monospaced) }

  var body: some View {
    switch row.kind {
    case .prompt:
      prompt
    case .answer:
      answer
    case .reasoning:
      folded(title: "Thought", icon: "brain", tint: .secondary)
    case .toolCall:
      folded(title: row.tool ?? "tool", icon: "terminal", tint: Color.reading)
    case .toolOutput:
      folded(title: "\(row.tool ?? "tool") said", icon: "text.alignleft", tint: .secondary)
    case .notice:
      notice
    case .system, .steer:
      EmptyView()
    }
  }

  /// Where a turn stopped or came apart, in the conversation at the point it happened.
  private var notice: some View {
    let stopped = row.tool == "stopped"
    return HStack(alignment: .firstTextBaseline, spacing: 6) {
      Image(systemName: stopped ? "stop.circle" : "exclamationmark.triangle")
        .font(.footnote)
      Text(row.text)
        .font(.footnote)
        .textSelection(.enabled)
      Spacer(minLength: 0)
    }
    .foregroundStyle(stopped ? Color.secondary : .orange)
    .padding(.horizontal, 10)
    .padding(.vertical, 6)
    .background(.thinMaterial, in: .rect(cornerRadius: 10))
  }

  private var prompt: some View {
    HStack {
      Spacer(minLength: 32)
      Text(row.text)
        .font(.system(size: size))
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.reading.opacity(0.16), in: .rect(cornerRadius: 14))
        .textSelection(.enabled)
    }
  }

  private var answer: some View {
    StreamedMarkdown(text: row.text, mono: mono, size: size, live: live)
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(.horizontal, 12)
      .padding(.vertical, 8)
      .background(.thinMaterial, in: .rect(cornerRadius: 14))
      .textSelection(.enabled)
      .contextMenu {
        Button("Copy text", systemImage: "doc.on.doc") { Clipboard.copy(row.text) }
        ShareLink(item: row.text)
      }
  }

  private var isOpen: Bool { open ?? live }

  private func folded(title: String, icon: String, tint: Color) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      Button {
        open = !isOpen
      } label: {
        HStack(spacing: 6) {
          Image(systemName: icon)
            .font(.system(size: 10))
          Text(title)
            .font(.system(size: 11, weight: .medium, design: .monospaced))
          if live {
            AnimatedDots(size: 3, tint: tint)
          }
          Spacer()
          if let seconds = row.seconds, seconds > 0 {
            Text(ReadoutFormat.duration(seconds))
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(.tertiary)
          }
          Image(systemName: isOpen ? "chevron.down" : "chevron.right")
            .font(.system(size: 9))
        }
        .foregroundStyle(tint)
      }
      .buttonStyle(.plain)

      if isOpen {
        Text(body(of: row))
          .font(mono)
          .foregroundStyle(.primary.opacity(0.85))
          .textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading)
          .padding(10)
          .background(.quaternary.opacity(0.4), in: .rect(cornerRadius: 10))
      }
    }
  }

  /// Tool calls arrive as JSON. Shown as its fields rather than as a blob, because a phone has
  /// no room to scroll sideways through one.
  private func body(of row: TranscriptRow) -> String {
    guard row.kind == .toolCall,
      let data = row.text.data(using: .utf8),
      let fields = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else { return row.text }
    return
      fields
      .sorted { $0.key < $1.key }
      .map { "\($0.key): \(describe($0.value))" }
      .joined(separator: "\n")
  }

  private func describe(_ value: Any) -> String {
    switch value {
    case let text as String: text
    case let number as NSNumber: number.stringValue
    default: String(describing: value)
    }
  }
}
