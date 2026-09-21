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

  /// Whether the true bottom of the transcript is on screen right now, tracked from the
  /// marker at its end rather than guessed from the last scroll gesture.
  @State private var isAtBottom = true
  /// Shown once someone has scrolled up more than two-thirds of a screen, so there is a way
  /// back without hunting for a scrollbar.
  @State private var showJumpToBottom = false
  /// The transcript's own height, watched so growth can be told apart from a scroll: only
  /// growth should pull the view back down while pinned to the bottom.
  @State private var contentHeight: CGFloat = 0

  private struct BottomMarkerKey: PreferenceKey {
    nonisolated(unsafe) static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
  }

  private struct ContentHeightKey: PreferenceKey {
    nonisolated(unsafe) static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
  }

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
      GeometryReader { outer in
        ZStack(alignment: .bottomTrailing) {
          ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
              ForEach(chat.rows) { row in
                // RowCost sits past the row's trailing edge rather than beside it, so the row
                // never reports a width wider than the column actually is — the earlier
                // version did that with negative padding, which left the true content wider
                // than anything downstream believed, and that gap could paint past the window
                // instead of hiding.
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
              // A zero-height marker at the very end. Its own position, read back through a
              // preference, is how the view knows whether the true bottom is on screen —
              // sturdier than inferring it from scroll offsets and content sizes by hand.
              Color.clear
                .frame(height: 1)
                .id("bottomAnchor")
                .background(
                  GeometryReader { marker in
                    Color.clear.preference(
                      key: BottomMarkerKey.self,
                      value: marker.frame(in: .named("transcriptScroll")).maxY)
                  }
                )
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
              GeometryReader { content in
                Color.clear.preference(key: ContentHeightKey.self, value: content.size.height)
              }
            )
          }
          .coordinateSpace(name: "transcriptScroll")
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
          .onPreferenceChange(BottomMarkerKey.self) { maxY in
            let distance = maxY - outer.size.height
            isAtBottom = distance < 40
            let farEnough = distance > outer.size.height * (2.0 / 3.0)
            if farEnough != showJumpToBottom {
              withAnimation(.easeOut(duration: 0.15)) { showJumpToBottom = farEnough }
            }
          }
          .onChange(of: contentHeight) {
            // A row growing while the transcript is pinned to the bottom should read as the
            // content pushing itself up, not as the view panning down after it — so this
            // follow is a snap, never an animation.
            guard isAtBottom else { return }
            scroller.scrollTo("bottomAnchor", anchor: .bottom)
          }
          .onPreferenceChange(ContentHeightKey.self) { contentHeight = $0 }
          .onChange(of: chat.rows.count) {
            withAnimation(.easeOut(duration: 0.15)) {
              scroller.scrollTo("bottomAnchor", anchor: .bottom)
            }
          }
          // A drag left mid-gesture by switching chats should not keep shifting the next
          // conversation's rows aside.
          .onChange(of: chat.current.id) { reveal = 0 }

          if showJumpToBottom {
            jumpToBottomButton(scroller: scroller)
              .padding(16)
          }
        }
      }
    }
  }

  /// The way back down once someone has scrolled far enough that the composer's own edge
  /// no longer suggests it. Subtle on purpose — a glass circle, not a banner.
  private func jumpToBottomButton(scroller: ScrollViewProxy) -> some View {
    Button {
      withAnimation(.easeOut(duration: 0.2)) {
        scroller.scrollTo("bottomAnchor", anchor: .bottom)
      }
    } label: {
      Image(systemName: "arrow.down")
        .font(.system(size: 12, weight: .semibold))
        .foregroundStyle(.secondary)
        .frame(width: 30, height: 30)
    }
    .buttonStyle(.plain)
    .glassEffect(.clear, in: .circle)
    .transition(.opacity.combined(with: .scale(scale: 0.85)))
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
struct ChatRowView: View, Equatable {
  nonisolated static func == (a: ChatRowView, b: ChatRowView) -> Bool {
    a.row == b.row && a.mono == b.mono && a.size == b.size && a.live == b.live
      && a.caption == b.caption
  }

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

  /// Set once someone clicks past the height cap on a long body. A shell dump that scrolls
  /// the whole transcript off-screen is worse than one more click.
  @State private var showFull = false

  /// A thought stays collapsed even while live: reopening itself every time new text lands is
  /// what left one stuck open, a frame behind the row it belonged to.
  private var autoOpensLive: Bool {
    if case .reasoning = row.kind { false } else { true }
  }

  /// How tall an open body is let grow before it is capped with a "see more".
  private static let bodyHeightCap: Double = 260

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
      StreamedMarkdown(text: row.text, mono: mono, size: size, live: live)
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
        let long = Self.isLong(body)
        // Output worth capping is worth reading from its tail: the last lines of a shell
        // dump are the ones that say how it ended, so a capped block stays pinned to the
        // bottom and "see more" sits above it, pointing at what is hidden above.
        let capped = long && !showFull

        VStack(alignment: .leading, spacing: 4) {
          if long {
            Button {
              showFull.toggle()
            } label: {
              HStack(spacing: 3) {
                Image(systemName: capped ? "chevron.up" : "chevron.down")
                  .font(.system(size: 7))
                Text(capped ? "see more" : "see less")
              }
            }
            .buttonStyle(.plain)
            .font(.system(size: 9, weight: .medium, design: .monospaced))
            .foregroundStyle(.secondary)
          }

          Group {
            if monospaced {
              Text(body)
                .font(mono)
                .textSelection(.enabled)
            } else {
              StreamedMarkdown(text: body, mono: mono, size: size - 1, live: live)
            }
          }
          .foregroundStyle(.primary.opacity(0.85))
          // Once there is more than one line the block takes the width rather than sizing
          // itself to whichever line happens to be longest, which left a ragged right edge
          // that moved as the text streamed in.
          .frame(maxWidth: body.contains("\n") ? .infinity : nil, alignment: .leading)
          .frame(maxHeight: capped ? Self.bodyHeightCap : nil, alignment: .bottom)
          .clipped()
        }
        .textPlate(radius: 8, horizontal: 9, vertical: 5)
        // Indented to sit under its own title rather than beside it.
        .padding(.leading, 15)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  /// Long enough that showing it in full would push the rest of the transcript off-screen.
  private static func isLong(_ body: String) -> Bool {
    body.utf8.count > 1200 || body.reduce(into: 0) { count, ch in if ch == "\n" { count += 1 } } > 14
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
