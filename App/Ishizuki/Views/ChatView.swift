// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The conversation: the transcript as it fills, a composer under it, the readout in the corner.

import IshizukiKit
import SwiftUI

@available(macOS 27.0, *)
struct ChatView: View {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController

  @AppStorage("chat.monoFont") private var monoFont = ""
  @AppStorage("chat.fontSize") private var fontSize = 12.0

  var body: some View {
    VStack(spacing: 0) {
      header
      Divider().opacity(0.3)
      transcript
      Divider().opacity(0.3)
      ChatReadoutBar(chat: chat, controller: controller)
      composer
    }
    .windowBackdrop()
  }

  @ViewBuilder private var header: some View {
    HStack(spacing: 8) {
      Image(systemName: "folder")
        .foregroundStyle(.secondary)
        .font(.system(size: 11))
      Button {
        chat.chooseWorkspace()
      } label: {
        Text(chat.workspace?.lastPathComponent ?? "Choose a folder…")
          .font(.system(size: 11, design: .monospaced))
          .lineLimit(1)
      }
      .buttonStyle(.plain)
      .help(chat.workspace?.path ?? "The one directory the agent may touch")

      Spacer()

      if let failure = chat.failure {
        Text(failure)
          .font(.system(size: 10))
          .foregroundStyle(.orange)
          .lineLimit(1)
      }
    }
    .textPlate(radius: 8)
    .padding(.horizontal, 10)
    .padding(.top, 8)
  }

  @ViewBuilder private var transcript: some View {
    ScrollViewReader { scroller in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 10) {
          ForEach(chat.rows) { row in
            ChatRowView(row: row, mono: mono, size: fontSize)
              .id(row.id)
          }
          if chat.isResponding {
            TurnStatus(chat: chat)
              .id("tail")
          }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
      }
      .onChange(of: chat.rows.count) {
        withAnimation(.easeOut(duration: 0.15)) {
          scroller.scrollTo(chat.isResponding ? "tail" : chat.rows.last?.id, anchor: .bottom)
        }
      }
    }
  }

  @ViewBuilder private var composer: some View {
    HStack(alignment: .center, spacing: 8) {
      TextField(
        chat.isResponding ? "Steer the next turn…" : "What needs doing?",
        text: $chat.draft, axis: .vertical)
        .textFieldStyle(.plain)
        .font(mono)
        .lineLimit(1...6)
        .onSubmit { chat.submit() }

      if chat.isResponding {
        Button("Stop", systemImage: "stop.fill") { chat.stop() }
          .labelStyle(.iconOnly)
          .help("Stop this turn")
      }
      Button(chat.submissionLabel) { chat.submit() }
        .buttonStyle(.borderedProminent)
        .disabled(!chat.canSubmit)
    }
    .textPlate(radius: 10, horizontal: 12, vertical: 10)
    .padding(.horizontal, 10)
    .padding(.bottom, 10)
  }

  private var mono: Font {
    monoFont.isEmpty
      ? .system(size: fontSize, design: .monospaced)
      : .custom(monoFont, size: fontSize)
  }
}

@available(macOS 27.0, *)
struct ChatRowView: View {
  let row: ChatController.Row
  let mono: Font
  let size: Double

  @State private var expanded = false

  var body: some View {
    switch row.kind {
    case .prompt:
      Text(row.text)
        .font(.system(size: size))
        .textSelection(.enabled)
        .textPlate(radius: 12)
        .frame(maxWidth: .infinity, alignment: .trailing)

    // Steering is not an interruption, and saying so is the difference between a message that
    // looks ignored and one that is plainly waiting its turn.
    case .steer:
      VStack(alignment: .trailing, spacing: 3) {
        Text(row.text)
          .font(.system(size: size))
          .textSelection(.enabled)
        HStack(spacing: 4) {
          Image(systemName: "arrow.turn.down.right")
            .font(.system(size: 8))
          Text("queued for the next turn")
            .font(.system(size: 9))
        }
        .foregroundStyle(.secondary)
      }
      .textPlate(radius: 12)
      .overlay {
        RoundedRectangle(cornerRadius: 12)
          .strokeBorder(Color.accentSoft.opacity(0.35), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
      }
      .frame(maxWidth: .infinity, alignment: .trailing)

    case .answer:
      MarkdownText(text: row.text, mono: mono, size: size)
        .textPlate(radius: 12, horizontal: 12, vertical: 9)
        .frame(maxWidth: .infinity, alignment: .leading)

    case .reasoning:
      disclosure(
        title: "thought", icon: "brain", tint: .secondary,
        body: row.text, monospaced: false)

    case .toolCall(let name):
      disclosure(
        title: name, icon: icon(for: name), tint: .accentSoft,
        body: row.text, monospaced: true)

    case .toolOutput(let name):
      disclosure(
        title: "\(name) →", icon: "arrow.turn.down.right", tint: .secondary,
        body: row.text, monospaced: true)
    }
  }

  /// Collapsed by default: a coding turn is mostly tool traffic, and the answer is the part
  /// worth reading first.
  @ViewBuilder private func disclosure(
    title: String, icon: String, tint: Color, body: String, monospaced: Bool
  ) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Button {
        expanded.toggle()
      } label: {
        HStack(spacing: 5) {
          Image(systemName: icon)
            .font(.system(size: 9))
          Text(title)
            .font(.system(size: 10, weight: .medium, design: .monospaced))
          Text(summary(of: body))
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(.secondary)
            .lineLimit(1)
          Image(systemName: expanded ? "chevron.down" : "chevron.right")
            .font(.system(size: 7))
            .foregroundStyle(.secondary)
        }
        .foregroundStyle(tint)
      }
      .buttonStyle(.plain)

      if expanded {
        Text(body)
          .font(monospaced ? mono : .system(size: size - 1))
          .foregroundStyle(.primary.opacity(0.85))
          .textSelection(.enabled)
          .textPlate(radius: 8)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private func summary(of body: String) -> String {
    let flat = body.replacingOccurrences(of: "\n", with: " ")
    return flat.count > 80 ? String(flat.prefix(80)) + "…" : flat
  }

  private func icon(for tool: String) -> String {
    switch tool {
    case "read": "doc.text"
    case "write": "square.and.pencil"
    case "edit": "pencil.line"
    case "grep": "magnifyingglass"
    case "glob": "folder.badge.questionmark"
    case "shell": "terminal"
    default: "wrench"
    }
  }
}
