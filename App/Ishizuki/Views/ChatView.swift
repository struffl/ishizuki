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

  /// How far the transcript is dragged aside to show what each row cost.
  @State private var reveal: CGFloat = 0
  private let gutter: CGFloat = 116

  var body: some View {
    VStack(spacing: 0) {
      header
      Divider().opacity(0.3)
      transcript
      Divider().opacity(0.3)
      ChatReadoutBar(chat: chat, controller: controller)
      queued
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
            HStack(spacing: 8) {
              ChatRowView(
                row: row, mono: mono, size: fontSize,
                live: chat.isResponding && row.id == chat.rows.last?.id)
              RowCost(meta: chat.meta(for: row))
                .frame(width: gutter, alignment: .leading)
                .opacity(reveal / gutter)
            }
            .padding(.trailing, -gutter)
            .offset(x: -reveal)
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
      // Pulled aside and let go, the way a message list gives up its timestamps.
      .gesture(
        DragGesture(minimumDistance: 14)
          .onChanged { value in
            guard value.translation.width < 0 else { return }
            reveal = min(gutter, -value.translation.width)
          }
          .onEnded { _ in
            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) { reveal = 0 }
          }
      )
      .onChange(of: chat.rows.count) {
        withAnimation(.easeOut(duration: 0.15)) {
          scroller.scrollTo(chat.isResponding ? "tail" : chat.rows.last?.id, anchor: .bottom)
        }
      }
    }
  }

  /// What is waiting for the next turn, one line each, with the means to send it now. Sitting
  /// above the composer rather than in the transcript, because it has not been said yet.
  @ViewBuilder private var queued: some View {
    if !chat.pendingSteers.isEmpty {
      VStack(spacing: 4) {
        ForEach(chat.pendingSteers) { row in
          HStack(spacing: 8) {
            Image(systemName: "arrow.turn.down.right")
              .font(.system(size: 9))
              .foregroundStyle(.secondary)
            Text(row.text)
              .font(.system(size: 11))
              .lineLimit(1)
              .truncationMode(.tail)
            Spacer(minLength: 8)
            Button {
              chat.sendQueuedNow()
            } label: {
              HStack(spacing: 3) {
                Text("Send now")
                  .font(.system(size: 10, weight: .medium))
                Image(systemName: "return")
                  .font(.system(size: 9))
              }
            }
            .buttonStyle(.plain)
            .foregroundStyle(Color.accentSoft)
            .help("Stop this turn and send it now")
            Button {
              chat.drop(row)
            } label: {
              Image(systemName: "xmark")
                .font(.system(size: 9))
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Drop it")
          }
          .textPlate(radius: 9, horizontal: 10, vertical: 6)
          .overlay {
            RoundedRectangle(cornerRadius: 9)
              .strokeBorder(
                Color.accentSoft.opacity(0.3),
                style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
          }
        }
      }
      .padding(.horizontal, 10)
      .padding(.top, 2)
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
  /// The row the model is writing into right now, which opens itself so the thinking can be
  /// watched rather than waited out.
  var live = false

  @State private var expanded = false

  private var open: Bool { expanded || live }

  var body: some View {
    switch row.kind {
    // Yellow for what the model was given, blue for what was asked of it: between them they
    // account for the tokens someone is waiting on before a word comes back.
    case .system:
      disclosure(
        title: "instructions", icon: "list.bullet.rectangle", tint: .instructing,
        body: row.text, monospaced: false)

    case .prompt:
      Text(row.text)
        .font(.system(size: size))
        .foregroundStyle(.white)
        .textSelection(.enabled)
        .bubble(mine: true)
        .padding(.trailing, 10)
        .padding(.leading, 44)
        .frame(maxWidth: .infinity, alignment: .trailing)

    case .steer:
      EmptyView()

    case .answer:
      MarkdownText(text: row.text, mono: mono, size: size)
        .glassBubble()
        .padding(.leading, 10)
        .padding(.trailing, 44)
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
    VStack(alignment: .leading, spacing: 2) {
      Button {
        expanded.toggle()
      } label: {
        HStack(spacing: 5) {
          Image(systemName: icon)
            .font(.system(size: 9))
          Text(title)
            .font(.system(size: 10, weight: .medium, design: .monospaced))
          if !open {
            Text(summary(of: body))
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(.secondary)
              .lineLimit(1)
          }
          Image(systemName: open ? "chevron.down" : "chevron.right")
            .font(.system(size: 7))
            .foregroundStyle(.secondary)
        }
        .foregroundStyle(tint)
      }
      .buttonStyle(.plain)

      if open, !body.isEmpty {
        Text(body)
          .font(monospaced ? mono : .system(size: size - 1))
          .foregroundStyle(.primary.opacity(0.85))
          .textSelection(.enabled)
          // Once there is more than one line the block takes the width rather than sizing
          // itself to whichever line happens to be longest, which left a ragged right edge
          // that moved as the text streamed in.
          .frame(maxWidth: body.contains("\n") ? .infinity : nil, alignment: .leading)
          .textPlate(radius: 8, horizontal: 9, vertical: 5)
          // Indented to sit under its own title rather than beside it.
          .padding(.leading, 15)
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

/// What a row cost, shown in the gutter: when it happened, how long that side of the turn
/// took, and how many tokens it was.
@available(macOS 27.0, *)
struct RowCost: View {
  let meta: ChatController.RowMeta?

  var body: some View {
    if let meta {
      VStack(alignment: .leading, spacing: 1) {
        Text(meta.at, format: .dateTime.hour().minute().second())
          .foregroundStyle(.secondary)
        if let seconds = meta.seconds, seconds > 0 {
          Text(
            (meta.wasRead ? "read " : "wrote ")
              + String(format: seconds < 10 ? "%.1fs" : "%.0fs", seconds))
        }
        if let tokens = meta.tokens, tokens > 0 {
          Text("\(ReadoutFormat.group(tokens)) tok")
        }
      }
      .font(.system(size: 9, design: .monospaced))
      .foregroundStyle(.tertiary)
    }
  }
}
