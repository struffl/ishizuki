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
    NavigationSplitView {
      ChatSidebar(chat: chat, controller: controller)
        .navigationSplitViewColumnWidth(min: 170, ideal: 210, max: 320)
    } detail: {
      VStack(spacing: 0) {
        header
        Divider().opacity(0.3)
        transcript
        Divider().opacity(0.3)
        ChatReadoutBar(chat: chat, controller: controller)
        queued
        composer
      }
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
            // RowCost sits past the row's trailing edge rather than beside it, so the row
            // never reports a width wider than the column actually is — the earlier version
            // did that with negative padding, which left the true content wider than anything
            // downstream believed, and that gap could paint past the window instead of hiding.
            ZStack(alignment: .trailing) {
              ChatRowView(
                row: row, mono: mono, size: fontSize,
                live: chat.isResponding && row.id == chat.rows.last?.id,
                caption: chat.captioner.caption(for: row.id)
              )
              .frame(maxWidth: .infinity, alignment: .leading)
              .offset(x: -reveal)
              RowCost(meta: chat.meta(for: row))
                .frame(width: gutter, alignment: .leading)
                .opacity(reveal / gutter)
                .offset(x: gutter - reveal)
            }
            .clipped()
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
      // A drag left mid-gesture by switching chats should not keep shifting the next
      // conversation's rows aside.
      .onChange(of: chat.current.id) { reveal = 0 }
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
        text: $chat.draft, axis: .vertical
      )
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
  /// What the system model made of this, when it has had a look.
  var caption: String?

  /// nil follows the default (open while live, closed once it settles); set the moment someone
  /// clicks, so a click during streaming can still close a row that would otherwise force itself
  /// open every frame.
  @State private var expanded: Bool?

  /// A thought stays collapsed even while live: reopening itself every time new text lands is
  /// what left one stuck open, a frame behind the row it belonged to.
  private var autoOpensLive: Bool {
    if case .reasoning = row.kind { false } else { true }
  }

  private var open: Bool { expanded ?? (live && autoOpensLive) }

  var body: some View {
    switch row.kind {
    // Yellow for what the model was given, blue for what was asked of it: between them they
    // account for the tokens someone is waiting on before a word comes back.
    case .system:
      disclosure(
        title: "instructions", icon: "list.bullet.rectangle", tint: .instructing,
        rawBody: row.text, monospaced: false)

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
        title: live ? "thinking…" : "thought", icon: "brain", tint: .secondary,
        rawBody: row.text, monospaced: false)

    case .toolCall(let name):
      disclosure(
        title: name, icon: icon(for: name), tint: .accentSoft,
        rawBody: Self.spelled(arguments: row.text), monospaced: true)

    case .toolOutput(let name):
      disclosure(
        title: "\(name) →", icon: "arrow.turn.down.right", tint: .secondary,
        rawBody: row.text, monospaced: true)
    }
  }

  /// Collapsed by default: a coding turn is mostly tool traffic, and the answer is the part
  /// worth reading first.
  @ViewBuilder private func disclosure(
    title: String, icon: String, tint: Color, rawBody: String, monospaced: Bool
  ) -> some View {
    // Command output arrives with its trailing newlines, which a Text keeps as blank lines and
    // the plate then paints around: a shell row sat on a band of empty space no other row had.
    let body = rawBody.trimmingCharacters(in: .whitespacesAndNewlines)
    VStack(alignment: .leading, spacing: 2) {
      Button {
        expanded = !open
      } label: {
        HStack(spacing: 5) {
          Image(systemName: icon)
            .font(.system(size: 9))
          Text(title)
            .font(.system(size: 10, weight: .medium, design: .monospaced))
          if !open {
            Text(caption ?? summary(of: body))
              .font(
                caption == nil
                  ? .system(size: 10, design: .monospaced) : .system(size: 10)
              )
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
        Group {
          if monospaced {
            Text(body)
              .font(mono)
              .textSelection(.enabled)
          } else {
            MarkdownText(text: body, mono: mono, size: size - 1)
          }
        }
        .foregroundStyle(.primary.opacity(0.85))
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

  /// The arguments as the model wrote them, spelled out rather than left as the JSON they
  /// arrived in: a shell row should read as the command it ran. A single argument stands on
  /// its own; several are labelled, longest last so the command keeps the first line.
  static func spelled(arguments: String) -> String {
    guard
      let data = arguments.data(using: .utf8),
      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      !object.isEmpty
    else { return arguments }

    func written(_ value: Any) -> String {
      switch value {
      case let string as String: string
      case let bool as Bool: bool ? "true" : "false"
      case let number as NSNumber: number.stringValue
      default: String(describing: value)
      }
    }

    if object.count == 1, let only = object.values.first {
      return written(only)
    }
    return object.keys.sorted {
      (object[$0].map { written($0).count } ?? 0) < (object[$1].map { written($0).count } ?? 0)
    }
    .map { "\($0): \(written(object[$0] ?? ""))" }
    .joined(separator: "\n")
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
