// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The conversation: the transcript as it fills, a composer under it, and the line between them
// that says where the context went and where the work is happening.

import IshizukiKit
import SwiftUI

@available(macOS 27.0, *)
struct ChatView: View {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController

  @AppStorage("chat.monoFont") private var monoFont = ""
  @AppStorage("chat.fontSize") private var fontSize = 13.0

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
  /// Whether the mouse is down somewhere in the transcript. A click is a press and a release
  /// in the same place, and the bottom-follow moves the transcript under the pointer between
  /// the two — which is why a chevron could not be clicked while a turn was streaming. The
  /// follow holds off until the button comes back up.
  @State private var isPressing = false
  /// Runs of tool traffic someone has opened up. Held here rather than on the rows, because a
  /// run is a thing the view makes and the transcript knows nothing about.
  @State private var openRuns: Set<String> = []
  /// The detail column's height — the window's, in effect — which is what the composer is
  /// allowed to grow into.
  ///
  /// Measured here rather than on the transcript. The transcript is whatever the composer
  /// leaves it, so feeding its height back into how tall the composer may grow is a loop:
  /// the box grows, the transcript shrinks, the box is allowed less, and AppKit gives up
  /// partway through a constraint pass. This column's height is the window's and does not
  /// move when the composer does.
  @State private var viewportHeight: CGFloat = 0

  private struct ViewportHeightKey: PreferenceKey {
    nonisolated(unsafe) static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
  }

  /// Three fifths of the window, in lines. Past that the box scrolls rather than grows: a
  /// composer that can eat the whole window is not a composer any more.
  private var composerLines: Int {
    guard viewportHeight.isFinite, viewportHeight > 0 else { return 6 }
    let line = max(8, fontSize + 5)
    return min(48, max(3, Int(viewportHeight * 0.6 / line)))
  }

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
        transcript
        Divider().opacity(0.3)
        ChatReadoutBar(chat: chat, controller: controller)
        ContextStrip(chat: chat)
        if let failure = chat.failure {
          HStack(spacing: 5) {
            Image(systemName: "exclamationmark.triangle")
            Text(failure)
              .lineLimit(2)
            Spacer(minLength: 0)
          }
          .font(.footnote)
          .foregroundStyle(.orange)
          .padding(.horizontal, 14)
          .padding(.top, 2)
        }
        asking
        queued
        Composer(chat: chat, controller: controller, mono: mono, maxLines: composerLines)
      }
      .background(
        GeometryReader { column in
          Color.clear.preference(key: ViewportHeightKey.self, value: column.size.height)
        }
      )
      .onPreferenceChange(ViewportHeightKey.self) { height in
        // Rounded to a coarse step, so a point of drift as a row settles is not a fresh pass
        // over the whole window.
        let stepped = (height / 40).rounded(.down) * 40
        if stepped != viewportHeight { viewportHeight = stepped }
      }
    }
    .windowBackdrop()
  }

  @ViewBuilder private var transcript: some View {
    ScrollViewReader { scroller in
      GeometryReader { outer in
        ZStack(alignment: .bottomTrailing) {
          ScrollView {
            // Spacing is set per row rather than once for the stack, so a run of tool traffic
            // closes up into one block and air is spent only where the voice changes.
            LazyVStack(alignment: .leading, spacing: 0) {
              if blocks.isEmpty, !chat.isResponding {
                emptyTranscript
              }
              ForEach(Array(blocks.enumerated()), id: \.element.id) { index, block in
                // RowCost sits past the row's trailing edge rather than beside it, so the row
                // never reports a width wider than the column actually is — the earlier
                // version did that with negative padding, which left the true content wider
                // than anything downstream believed, and that gap could paint past the window
                // instead of hiding.
                ZStack(alignment: .trailing) {
                  content(for: block)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .offset(x: -reveal)
                  RowCost(meta: chat.meta(for: block.last))
                    .frame(width: gutter, alignment: .leading)
                    .opacity(reveal / gutter)
                    .offset(x: gutter - reveal)
                }
                .clipped()
                .padding(
                  .top,
                  Self.gap(after: index > 0 ? blocks[index - 1] : nil, before: block)
                )
                .id(block.id)
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
          // Tracked alongside everything else rather than in place of it: a zero-distance drag
          // recognises on the press and ends on the release without taking the click away from
          // whatever is under it.
          .simultaneousGesture(
            DragGesture(minimumDistance: 0)
              .onChanged { _ in if !isPressing { isPressing = true } }
              .onEnded { _ in isPressing = false }
          )
          .onPreferenceChange(BottomMarkerKey.self) { maxY in
            let distance = maxY - outer.size.height
            isAtBottom = distance < 16
            let farEnough = distance > outer.size.height * (2.0 / 3.0)
            if farEnough != showJumpToBottom {
              withAnimation(.easeOut(duration: 0.15)) { showJumpToBottom = farEnough }
            }
          }
          .onChange(of: contentHeight) {
            // A row growing while the transcript is pinned to the bottom should read as the
            // content pushing itself up, not as the view panning down after it — so this
            // follow is a snap, never an animation.
            //
            // Only while a turn is in flight. A lazy row settling on its true height as it
            // scrolls in also moves this, and following that was the transcript hauling itself
            // back down under someone who was reading it.
            //
            // And never under a pressed mouse button: whatever is being clicked stays where it
            // was until the click has been made. The follow resumes on the next tick of growth.
            guard chat.isResponding, isAtBottom, !isPressing else { return }
            scroller.scrollTo("bottomAnchor", anchor: .bottom)
          }
          .onPreferenceChange(ContentHeightKey.self) { contentHeight = $0 }
          .onChange(of: rows.count) {
            // Unanimated, like the follow above: a new row lands at the same moment the height
            // changes, and two animated scrolls towards the same anchor read as a lurch.
            //
            // Someone reading back through a turn is left where they are, but sending something
            // always goes to it — that jump is the answer to their own click, not the view
            // wandering off on its own.
            guard !isPressing, isAtBottom || rows.last?.kind == .prompt else { return }
            scroller.scrollTo("bottomAnchor", anchor: .bottom)
          }
          // A drag left mid-gesture by switching chats should not keep shifting the next
          // conversation's rows aside, and the new transcript opens at its end.
          .onChange(of: chat.current.id) {
            reveal = 0
            isPressing = false
            isAtBottom = true
            openRuns.removeAll()
            scroller.scrollTo("bottomAnchor", anchor: .bottom)
          }

          if showJumpToBottom {
            jumpToBottomButton(scroller: scroller)
              .padding(16)
          }
        }
      }
    }
  }

  @ViewBuilder private func content(for block: Block) -> some View {
    switch block {
    case .row(let row):
      // Equatable, and taken at its word: a poll twenty times a second replaces the whole
      // array, and without this every row in the transcript is built again for the sake of the
      // one being written into.
      ChatRowView(
        row: row, mono: mono, size: fontSize,
        live: chat.isResponding && row.id == rows.last?.id,
        caption: chat.captioner.caption(for: row.id),
        meta: chat.meta(for: row),
        show: chat.display(for: row),
        onExpand: { open in
          withAnimation(.easeOut(duration: 0.18)) { chat.setExpanded(open, for: row.id) }
        },
        onShowFull: { full in
          withAnimation(.easeOut(duration: 0.18)) { chat.setShowFull(full, for: row.id) }
        }
      )
      .equatable()

    case .summary(_, let summary):
      TurnSummaryCard(summary: summary, workspace: chat.workspace)

    case .run(let members):
      ToolRunView(
        members: members,
        mono: mono,
        size: fontSize,
        open: openRuns.contains(block.id),
        seconds: members.compactMap { chat.meta(for: $0)?.elapsed }.reduce(0, +),
        onToggle: { open in
          withAnimation(.easeOut(duration: 0.2)) {
            if open { openRuns.insert(block.id) } else { openRuns.remove(block.id) }
          }
        },
        rowView: { row in
          ChatRowView(
            row: row, mono: mono, size: fontSize, live: false,
            caption: chat.captioner.caption(for: row.id),
            meta: chat.meta(for: row),
            show: chat.display(for: row),
            onExpand: { open in
              withAnimation(.easeOut(duration: 0.18)) { chat.setExpanded(open, for: row.id) }
            },
            onShowFull: { full in
              withAnimation(.easeOut(duration: 0.18)) { chat.setShowFull(full, for: row.id) }
            })
        })
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
        .font(.system(.callout, weight: .semibold))
        .foregroundStyle(.secondary)
        .frame(width: 30, height: 30)
    }
    .buttonStyle(.plain)
    .glassEffect(.clear, in: .circle)
    .accessibilityLabel("Jump to the end")
    .help("Jump to the end")
    .transition(.opacity.combined(with: .scale(scale: 0.85)))
  }

  /// The question the turn is waiting on, with whatever answers it offered as buttons. Typing
  /// and sending answers it too.
  @ViewBuilder private var asking: some View {
    if let question = chat.question {
      VStack(alignment: .leading, spacing: 6) {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
          Image(systemName: "questionmark.bubble")
            .foregroundStyle(Color.reading)
          Text(question.text)
            .font(.subheadline)
            .textSelection(.enabled)
        }
        if !question.options.isEmpty {
          HStack(spacing: 6) {
            ForEach(question.options, id: \.self) { option in
              Button(option) { chat.answer(option) }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
          }
        }
      }
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(.horizontal, 14)
      .padding(.top, 6)
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
              .font(.footnote)
              .foregroundStyle(.secondary)
            Text(row.text)
              .font(.subheadline)
              .lineLimit(1)
              .truncationMode(.tail)
            Spacer(minLength: 8)
            Button {
              chat.sendQueuedNow()
            } label: {
              HStack(spacing: 3) {
                Text("Send now")
                  .font(.system(.footnote, weight: .medium))
                Image(systemName: "return")
                  .font(.footnote)
              }
              .hitTarget()
            }
            .buttonStyle(.plain)
            .foregroundStyle(Color.reading)
            .help("Stop this turn and send it now")
            Button {
              chat.drop(row)
            } label: {
              Image(systemName: "xmark")
                .font(.footnote)
                .hitTarget()
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .accessibilityLabel("Drop this steer")
            .help("Drop it")
          }
          .textPlate(radius: 9, horizontal: 10, vertical: 6)
          .overlay {
            RoundedRectangle(cornerRadius: 9)
              .strokeBorder(
                Color.reading.opacity(0.3),
                style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
          }
        }
      }
      .padding(.horizontal, 10)
      .padding(.top, 2)
    }
  }

  private var rows: [ChatController.Row] { chat.visibleRows }

  /// A new conversation is not an empty screen: say what the one prominent action will do.
  @ViewBuilder private var emptyTranscript: some View {
    VStack(spacing: 10) {
      Image(systemName: emptyIcon)
        .font(.system(size: 30, weight: .light))
        .foregroundStyle(.secondary)
      Text(emptyTitle)
        .font(.title3.weight(.semibold))
      Text(emptyDetail)
        .font(.subheadline)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
      if chat.submission == .chooseFolder || chat.submission == .load {
        Button(chat.submissionLabel) { chat.submit() }
          .buttonStyle(.glassProminent)
      }
    }
    .frame(maxWidth: .infinity, minHeight: 220)
    .accessibilityElement(children: .contain)
  }

  private var emptyIcon: String {
    switch chat.submission {
    case .chooseFolder: "folder.badge.plus"
    case .load: "bolt.circle"
    default: "bubble.left.and.text.bubble.right"
    }
  }

  private var emptyTitle: String {
    switch chat.submission {
    case .chooseFolder: "Choose a working folder"
    case .load: "Load the pack to begin"
    default: "Start a conversation"
    }
  }

  private var emptyDetail: String {
    switch chat.submission {
    case .chooseFolder: "The agent will only work in the folder you choose."
    case .load: "Your selected model will stay ready for this conversation."
    default: "Write a request below to get started."
    }
  }

  /// A stretch of the transcript drawn as one thing. Most of it is one row per block; a run of
  /// machinery nobody has opened is a single line saying how many steps it was, the way a long
  /// turn reads once it has settled.
  enum Block: Identifiable {
    case row(ChatController.Row)
    case run([ChatController.Row])
    /// What a turn came to, drawn after its last row.
    case summary(after: ChatController.Row, ChatController.TurnSummary)

    var id: String {
      switch self {
      case .row(let row): row.id
      case .run(let members): "run-" + (members.first?.id ?? "")
      case .summary(let row, _): "summary-" + row.id
      }
    }

    var first: ChatController.Row {
      switch self {
      case .row(let row): row
      case .run(let members): members[0]
      case .summary(let row, _): row
      }
    }

    var last: ChatController.Row {
      switch self {
      case .row(let row): row
      case .run(let members): members[members.count - 1]
      case .summary(let row, _): row
      }
    }
  }

  /// Runs are gathered only out of shut machinery, and never out of the tail of a turn that is
  /// still being written: folding away the row the model is working in is the one thing a live
  /// transcript must not do.
  private var blocks: [Block] {
    let rows = self.rows
    let liveTail = chat.isResponding ? rows.count - 1 : rows.count
    var out: [Block] = []
    var pending: [ChatController.Row] = []

    func flush() {
      defer { pending = [] }
      guard pending.count >= Self.runThreshold else {
        out.append(contentsOf: pending.map(Block.row))
        return
      }
      let id = "run-" + (pending.first?.id ?? "")
      if openRuns.contains(id) {
        out.append(contentsOf: pending.map(Block.row))
      } else {
        out.append(.run(pending))
      }
    }

    for (index, row) in rows.enumerated() {
      let gatherable =
        row.kind.isMachinery && index < liveTail && chat.display(for: row).expanded != true
      if gatherable {
        pending.append(row)
      } else {
        flush()
        out.append(.row(row))
      }
      if let summary = chat.summary(after: row), !summary.files.isEmpty {
        flush()
        out.append(.summary(after: row, summary))
      }
    }
    flush()
    return out
  }

  /// Two rows is one tool call and its answer, which reads fine on its own. Three is where a
  /// transcript starts to be mostly machinery.
  private static let runThreshold = 4

  /// How much air a row gets above it. Sharing a voice with the row before means the two belong
  /// to one utterance and sit almost touching; only a change of voice earns a real gap.
  private static func gap(
    after previous: ChatController.Row.Kind?, before current: ChatController.Row.Kind
  ) -> CGFloat {
    guard let previous else { return 0 }
    return previous.voice == current.voice ? 2 : 8
  }

  private static func gap(after previous: Block?, before current: Block) -> CGFloat {
    if case .summary = current { return 8 }
    if case .summary = previous { return 8 }
    return gap(after: previous?.last.kind, before: current.first.kind)
  }

  private var mono: Font {
    monoFont.isEmpty
      ? .system(size: fontSize, design: .monospaced)
      : .custom(monoFont, size: fontSize)
  }
}

/// A run of tool traffic as one line, with what it was and how long it took. Opening it puts
/// every step back exactly as it would have been drawn on its own.
@available(macOS 27.0, *)
struct ToolRunView<Content: View>: View {
  let members: [ChatController.Row]
  let mono: Font
  let size: Double
  let open: Bool
  let seconds: Double
  let onToggle: (Bool) -> Void
  @ViewBuilder let rowView: (ChatController.Row) -> Content

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      Button {
        onToggle(!open)
      } label: {
        HStack(spacing: 5) {
          Image(systemName: icon)
            .font(.footnote)
          Text(title)
            .font(.system(.footnote, design: .monospaced, weight: .medium))
          if seconds > 0 {
            Text(ChatRowView.duration(seconds))
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(.tertiary)
          }
          Image(systemName: open ? "chevron.down" : "chevron.right")
            .font(.footnote)
            .foregroundStyle(.secondary)
        }
        .foregroundStyle(.secondary)
        .frame(minHeight: Metrics.hit)
        .chipPlate(radius: 8, horizontal: 8, vertical: 1)
        .contentShape(.rect)
      }
      .buttonStyle(.plain)
      .accessibilityLabel(title)
      .accessibilityAddTraits(.isToggle)

      if open {
        VStack(alignment: .leading, spacing: 2) {
          ForEach(members) { member in
            rowView(member)
          }
        }
        .padding(.leading, 15)
        .transition(.opacity.combined(with: .move(edge: .top)))
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private var names: [String] {
    members.compactMap { if case .toolCall(let name) = $0.kind { name } else { nil } }
  }

  private var icon: String {
    let unique = Set(names)
    if unique == ["shell"] { return "terminal" }
    if unique.isSubset(of: ["read", "grep", "glob"]) { return "magnifyingglass" }
    if unique.isSubset(of: ["edit", "write"]) { return "pencil.line" }
    return "wrench.and.screwdriver"
  }

  /// Named after what the run actually did, which is nearly always one of three things.
  private var title: String {
    let count = names.count
    guard count > 0 else { return "\(members.count) steps" }
    let unique = Set(names)
    let plural = count == 1 ? "" : "s"
    if unique == ["shell"] { return "Ran \(count) command\(plural)" }
    if unique.isSubset(of: ["read", "grep", "glob"]) { return "Explored \(count) place\(plural)" }
    if unique.isSubset(of: ["edit", "write"]) { return "Changed \(count) file\(plural)" }
    return "Ran \(count) tool\(plural)"
  }
}

@available(macOS 27.0, *)
struct ChatRowView: View, Equatable {
  nonisolated static func == (a: ChatRowView, b: ChatRowView) -> Bool {
    a.row == b.row && a.mono == b.mono && a.size == b.size && a.live == b.live
      && a.caption == b.caption && a.show == b.show && a.meta == b.meta
  }

  let row: ChatController.Row
  let mono: Font
  let size: Double
  /// The row the model is writing into right now, which opens itself so the thinking can be
  /// watched rather than waited out.
  var live = false
  /// What the system model made of this, when it has had a look.
  var caption: String?
  /// What this step cost, which is what the badge beside the title is reading.
  var meta: ChatController.RowMeta?
  /// Whether this row is open, and whether a capped body has been let out in full. Held by the
  /// controller rather than here: a LazyVStack does not keep a row's own `@State` across a
  /// scroll, and a row that came back shut changed height under the scroll position.
  var show = ChatController.RowDisplay()
  var onExpand: (Bool) -> Void = { _ in }
  var onShowFull: (Bool) -> Void = { _ in }

  /// A thought stays collapsed even while live: reopening itself every time new text lands is
  /// what left one stuck open, a frame behind the row it belonged to.
  private var autoOpensLive: Bool {
    if case .reasoning = row.kind { false } else { true }
  }

  /// How many lines of an open body are shown before it is capped with a "see more".
  ///
  /// Counted in lines and applied to the text itself, rather than clamped with a height and
  /// clipped: a view clipped to a height still reports the height it wanted, and that gap
  /// between the height a row claimed and the height it drew is what let a long shell dump
  /// throw the rest of the transcript around as it scrolled in and out of sight.
  private static let bodyLineCap = 12

  private var open: Bool { show.expanded ?? (live && autoOpensLive) }

  var body: some View {
    switch row.kind {
    // Yellow for what the model was given, blue for what was asked of it: between them they
    // account for the tokens someone is waiting on before a word comes back.
    case .system:
      disclosure(
        title: "instructions", icon: "list.bullet.rectangle", tint: .instructing,
        rawBody: row.text, monospaced: false)

    case .prompt:
      VStack(alignment: .trailing, spacing: 4) {
        if !row.images.isEmpty {
          PromptImages(paths: row.images)
        }
        Text(row.text)
          .font(.system(size: size))
          .foregroundStyle(.white)
          .textSelection(.enabled)
      }
      .bubble(mine: true)
      .padding(.trailing, 10)
      .padding(.leading, 44)
      .frame(maxWidth: .infinity, alignment: .trailing)

    case .steer:
      EmptyView()

    case .answer:
      VStack(alignment: .leading, spacing: 5) {
        StreamedMarkdown(text: row.text, mono: mono, size: size, live: live)
        if let total = meta?.turnSeconds, total >= 1 {
          HStack(spacing: 3) {
            Image(systemName: "stopwatch")
              .font(.system(size: 9))
            Text(Self.duration(total))
              .font(.system(size: 10, design: .monospaced))
          }
          .foregroundStyle(.tertiary)
        }
      }
      .contextMenu {
        Button("Copy text", systemImage: "doc.on.doc") { Clipboard.copy(row.text) }
        ShareLink(item: row.text)
      }
      .plateBubble()
      .padding(.leading, 10)
      .padding(.trailing, 44)
      .frame(maxWidth: .infinity, alignment: .leading)

    case .reasoning:
      disclosure(
        title: thoughtTitle, icon: "brain", tint: .secondary,
        rawBody: row.text, monospaced: false)

    case .toolCall(let name):
      if name == "edit", let diff = Self.editDiff(from: row.text) {
        editDisclosure(
          title: verb(for: name), icon: icon(for: name), path: diff.path,
          lines: diff.lines)
      } else {
        let headline = Self.headline(tool: name, arguments: row.text)
        disclosure(
          title: headline.verb, icon: icon(for: name), tint: .reading,
          rawBody: Self.spelled(arguments: row.text), monospaced: true,
          inlineDetail: headline.detail)
      }

    case .toolOutput(let name):
      disclosure(
        title: "\(verb(for: name)) →", icon: "arrow.turn.down.right", tint: .secondary,
        rawBody: row.text, monospaced: true)

    // Where a turn stopped or came apart, drawn in the conversation at the point it happened
    // rather than in a bar under it that the next turn wipes.
    case .notice(let tone):
      HStack(alignment: .firstTextBaseline, spacing: 6) {
        Image(systemName: tone == .stopped ? "stop.circle" : "exclamationmark.triangle")
          .font(.footnote)
        Text(row.text)
          .font(.footnote)
          .textSelection(.enabled)
        Spacer(minLength: 0)
      }
      .foregroundStyle(tone == .stopped ? Color.secondary : .orange)
      .chipPlate(radius: 8, horizontal: 10, vertical: 5)
      .contextMenu {
        Button("Copy text", systemImage: "doc.on.doc") { Clipboard.copy(row.text) }
      }
      .padding(.leading, 10)
      .padding(.trailing, 44)
      .frame(maxWidth: .infinity, alignment: .leading)
    }
  }

  /// "Thought" once it has settled, and for how long when that is known — the same thing a
  /// transcript of someone else's session tells you, and the one number worth having here.
  private var thoughtTitle: String {
    if live { return "thinking…" }
    guard let seconds = meta?.elapsed, seconds >= 0.5 else { return "thought" }
    return "thought for \(Self.duration(seconds))"
  }

  /// What a step took, in whichever unit reads as a number rather than a decimal.
  static func duration(_ seconds: Double) -> String {
    if seconds < 1 { return "\(Int((seconds * 1000).rounded()))ms" }
    if seconds < 60 { return String(format: "%.1fs", seconds) }
    let minutes = Int(seconds) / 60
    return "\(minutes)m \(Int(seconds) % 60)s"
  }

  /// Collapsed by default: a coding turn is mostly tool traffic, and the answer is the part
  /// worth reading first.
  @ViewBuilder private func disclosure(
    title: String, icon: String, tint: Color, rawBody: String, monospaced: Bool,
    inlineDetail: String? = nil
  ) -> some View {
    // Command output arrives with its trailing newlines, which a Text keeps as blank lines and
    // the plate then paints around: a shell row sat on a band of empty space no other row had.
    let body = rawBody.trimmingCharacters(in: .whitespacesAndNewlines)
    VStack(alignment: .leading, spacing: 2) {
      Button {
        onExpand(!open)
      } label: {
        HStack(spacing: 5) {
          Image(systemName: icon)
            .font(.footnote)
          Text(title)
            .font(.system(.footnote, design: .monospaced, weight: .medium))
          // A tool call says what it did on the line itself, open or shut: the command is the
          // point of the row, not something to be found inside it.
          if let inlineDetail, !inlineDetail.isEmpty {
            Text(inlineDetail)
              .font(.system(.footnote, design: .monospaced))
              .foregroundStyle(.primary.opacity(0.75))
              .lineLimit(1)
              .truncationMode(.middle)
          } else if !open {
            Text(caption ?? summary(of: body))
              .font(
                caption == nil
                  ? .system(.footnote, design: .monospaced) : .footnote
              )
              .foregroundStyle(.secondary)
              .lineLimit(1)
              .truncationMode(.tail)
          }
          if let seconds = meta?.elapsed, seconds >= 0.02 {
            Text(Self.duration(seconds))
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(.tertiary)
          }
          Image(systemName: open ? "chevron.down" : "chevron.right")
            .font(.footnote)
            .foregroundStyle(.secondary)
        }
        .foregroundStyle(tint)
        .frame(minHeight: Metrics.hit)
        .chipPlate(radius: 8, horizontal: 8, vertical: 1)
        .contentShape(.rect)
      }
      .buttonStyle(.plain)
      .accessibilityLabel(title)
      .accessibilityAddTraits(.isToggle)

      if open, !body.isEmpty {
        let long = Self.isLong(body)
        // Output worth capping is worth reading from its tail: the last lines of a shell
        // dump are the ones that say how it ended, so a capped block stays pinned to the
        // bottom and "see more" sits above it, pointing at what is hidden above.
        let capped = long && !show.showFull

        VStack(alignment: .leading, spacing: 4) {
          if long {
            Button {
              onShowFull(!show.showFull)
            } label: {
              HStack(spacing: 3) {
                Image(systemName: capped ? "chevron.up" : "chevron.down")
                  .font(.footnote)
                Text(capped ? "see more" : "see less")
              }
              .frame(minHeight: Metrics.hit)
              .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .font(.system(.footnote, design: .monospaced, weight: .medium))
            .foregroundStyle(.secondary)
          }

          let shown = capped ? Self.tail(of: body, lines: Self.bodyLineCap) : body

          Group {
            if monospaced {
              Text(shown)
                .font(mono)
                .textSelection(.enabled)
            } else {
              StreamedMarkdown(text: shown, mono: mono, size: size - 1, live: live)
            }
          }
          .foregroundStyle(.primary.opacity(0.85))
          // Once there is more than one line the block takes the width rather than sizing
          // itself to whichever line happens to be longest, which left a ragged right edge
          // that moved as the text streamed in.
          .frame(maxWidth: body.contains("\n") ? .infinity : nil, alignment: .leading)
          // One line long enough to wrap past the cap would otherwise slip through it.
          .lineLimit(capped ? Self.bodyLineCap : nil)
        }
        .textPlate(radius: 8, horizontal: 9, vertical: 5)
        // Indented to sit under its own title rather than beside it.
        .padding(.leading, 15)
        .transition(.opacity.combined(with: .move(edge: .top)))
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  /// Long enough that showing it in full would push the rest of the transcript off-screen.
  private static func isLong(_ body: String) -> Bool {
    body.utf8.count > 1200
      || body.reduce(into: 0) { count, ch in if ch == "\n" { count += 1 } } >= bodyLineCap
  }

  /// The last `lines` lines: the end of a dump is the part that says how it went.
  private static func tail(of body: String, lines: Int) -> String {
    let all = body.split(separator: "\n", omittingEmptySubsequences: false)
    guard all.count > lines else { return body }
    return all.suffix(lines).joined(separator: "\n")
  }

  /// The body flattened onto one line. It is not cut to a length here: how much of it fits is
  /// the window's business, and cutting it at eighty characters meant a wide window showed
  /// exactly as little as a narrow one.
  private func summary(of body: String) -> String {
    body.replacingOccurrences(of: "\n", with: " ")
      .replacing(/\s+/, with: " ")
      .trimmingCharacters(in: .whitespaces)
  }

  /// What a call did, said as the thing it did rather than as the name of the tool that did it.
  static func headline(tool: String, arguments: String) -> (verb: String, detail: String) {
    let object =
      (try? JSONSerialization.jsonObject(
        with: Data(arguments.utf8))) as? [String: Any]

    func field(_ names: String...) -> String? {
      for name in names {
        if let value = object?[name] as? String, !value.isEmpty { return value }
      }
      return nil
    }

    switch tool {
    case "shell": return ("Ran", field("command", "cmd") ?? "")
    case "read": return ("Read", field("path", "file") ?? "")
    case "write": return ("Wrote", field("path", "file") ?? "")
    case "edit": return ("Edited", field("path", "file") ?? "")
    case "grep": return ("Searched", field("pattern", "query") ?? "")
    case "glob": return ("Listed", field("pattern", "glob", "path") ?? "")
    case "web_search", "search": return ("Searched the web", field("query", "q") ?? "")
    default:
      let spelled = spelled(arguments: arguments)
      return (tool, spelled.split(separator: "\n").first.map(String.init) ?? "")
    }
  }

  /// How a tool's output is introduced, which is the same verb its call was given.
  private func verb(for tool: String) -> String {
    Self.headline(tool: tool, arguments: "{}").verb
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
    case "web_search", "search": "globe"
    default: "wrench"
    }
  }

  private enum DiffKind { case added, removed, context }

  /// An edit call reads as the change it made, not the JSON it arrived in: the old and new
  /// strings side by side, line by line, the way a patch does.
  @ViewBuilder private func editDisclosure(
    title: String, icon: String, path: String, lines: [(text: String, kind: DiffKind)]
  ) -> some View {
    let long = lines.count > Self.bodyLineCap
    let capped = long && !show.showFull
    let shown = capped ? Array(lines.suffix(Self.bodyLineCap)) : lines
    let added = lines.filter { $0.kind == .added }.count
    let removed = lines.filter { $0.kind == .removed }.count

    VStack(alignment: .leading, spacing: 2) {
      Button {
        onExpand(!open)
      } label: {
        HStack(spacing: 5) {
          Image(systemName: icon)
            .font(.footnote)
          Text(title)
            .font(.system(.footnote, design: .monospaced, weight: .medium))
          Text(path)
            .font(.system(.footnote, design: .monospaced))
            .foregroundStyle(.primary.opacity(0.75))
            .lineLimit(1)
            .truncationMode(.middle)
          // What the patch came to, which is the part someone scanning a turn is after.
          if added > 0 {
            Text("+\(added)")
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(Color.diffAdded)
          }
          if removed > 0 {
            Text("-\(removed)")
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(Color.diffRemoved)
          }
          if let seconds = meta?.elapsed, seconds >= 0.02 {
            Text(Self.duration(seconds))
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(.tertiary)
          }
          Image(systemName: open ? "chevron.down" : "chevron.right")
            .font(.footnote)
            .foregroundStyle(.secondary)
        }
        .foregroundStyle(Color.reading)
        .frame(minHeight: Metrics.hit)
        .chipPlate(radius: 8, horizontal: 8, vertical: 1)
        .contentShape(.rect)
      }
      .buttonStyle(.plain)
      .accessibilityLabel("\(title) \(path)")
      .accessibilityAddTraits(.isToggle)

      if open, !lines.isEmpty {
        VStack(alignment: .leading, spacing: 4) {
          if long {
            Button {
              onShowFull(!show.showFull)
            } label: {
              HStack(spacing: 3) {
                Image(systemName: capped ? "chevron.up" : "chevron.down")
                  .font(.footnote)
                Text(capped ? "see more" : "see less")
              }
              .frame(minHeight: Metrics.hit)
              .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .font(.system(.footnote, design: .monospaced, weight: .medium))
            .foregroundStyle(.secondary)
          }

          VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(shown.enumerated()), id: \.offset) { _, line in
              diffRow(line)
            }
          }
          .textSelection(.enabled)
        }
        .textPlate(radius: 8, horizontal: 9, vertical: 5)
        // Indented to sit under its own title rather than beside it.
        .padding(.leading, 15)
        .transition(.opacity.combined(with: .move(edge: .top)))
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private func style(for kind: DiffKind) -> (prefix: String, color: Color, background: Color) {
    switch kind {
    case .added: ("+", .diffAdded, Color.diffAdded.opacity(0.12))
    case .removed: ("-", .diffRemoved, Color.diffRemoved.opacity(0.12))
    case .context: (" ", .primary.opacity(0.85), .clear)
    }
  }

  @ViewBuilder private func diffRow(_ line: (text: String, kind: DiffKind)) -> some View {
    let (prefix, color, background) = style(for: line.kind)
    HStack(alignment: .top, spacing: 6) {
      Text(prefix)
        .font(mono)
        .foregroundStyle(color)
      Text(line.text.isEmpty ? " " : line.text)
        .font(mono)
        .foregroundStyle(color)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .padding(.horizontal, 4)
    .padding(.vertical, 1)
    .background(background)
  }

  /// The `old`/`new` an edit call carries, turned into a line diff. `nil` when the call isn't
  /// an edit shaped like one — a malformed payload falls back to the generic disclosure.
  private static func editDiff(
    from argumentsJSON: String
  ) -> (path: String, lines: [(text: String, kind: DiffKind)])? {
    guard
      let data = argumentsJSON.data(using: .utf8),
      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let old = object["old"] as? String,
      let new = object["new"] as? String
    else { return nil }
    let path = (object["path"] as? String) ?? ""
    return (path, diffLines(old: old, new: new))
  }

  /// A line-level diff via `CollectionDifference`, walked back into order: unchanged lines
  /// appear once, a removal is shown where the old line sat, an insertion where the new one
  /// lands.
  private static func diffLines(old: String, new: String) -> [(text: String, kind: DiffKind)] {
    let oldLines = old.isEmpty ? [] : old.components(separatedBy: "\n")
    let newLines = new.isEmpty ? [] : new.components(separatedBy: "\n")
    let diff = newLines.difference(from: oldLines)

    var removedAt: [Int: String] = [:]
    var insertedAt: [Int: String] = [:]
    for change in diff {
      switch change {
      case .remove(let offset, let element, _): removedAt[offset] = element
      case .insert(let offset, let element, _): insertedAt[offset] = element
      }
    }

    var result: [(text: String, kind: DiffKind)] = []
    var oldIndex = 0
    var newIndex = 0
    while oldIndex < oldLines.count || newIndex < newLines.count {
      if let removed = removedAt[oldIndex] {
        result.append((removed, .removed))
        oldIndex += 1
      } else if let inserted = insertedAt[newIndex] {
        result.append((inserted, .added))
        newIndex += 1
      } else if oldIndex < oldLines.count, newIndex < newLines.count {
        result.append((oldLines[oldIndex], .context))
        oldIndex += 1
        newIndex += 1
      } else {
        break
      }
    }
    return result
  }
}

/// The pictures a prompt came with, drawn in the bubble they were sent from. Read from disk,
/// because that is where they still are: a transcript carries the path and nothing more.
@available(macOS 27.0, *)
struct PromptImages: View {
  let paths: [String]

  @State private var loaded: [String: Image] = [:]

  var body: some View {
    HStack(spacing: 4) {
      ForEach(paths, id: \.self) { path in
        if let image = loaded[path] {
          image
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: 54, height: 54)
            .clipShape(.rect(cornerRadius: 6))
        } else {
          RoundedRectangle(cornerRadius: 6)
            .fill(.white.opacity(0.18))
            .frame(width: 54, height: 54)
            .overlay {
              Image(systemName: "photo")
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.7))
            }
        }
      }
    }
    .task(id: paths) {
      for path in paths where loaded[path] == nil {
        let url = URL(filePath: path)
        let data = await Task.detached(priority: .utility) {
          Thumbnail.png(of: url, maxPixel: 160)
        }.value
        guard let data, let image = NSImage(data: data) else { continue }
        loaded[path] = Image(nsImage: image)
      }
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
      .font(.system(.footnote, design: .monospaced))
      .foregroundStyle(.tertiary)
    }
  }
}

/// What a turn came to, once its tool traffic has scrolled by: the files it changed and what it
/// did to each of them. Three at a time, because the point of the card is to be read at a
/// glance rather than to be the transcript over again.
@available(macOS 27.0, *)
struct TurnSummaryCard: View {
  let summary: ChatController.TurnSummary
  let workspace: URL?

  @State private var showingAll = false

  private static let shownByDefault = 3

  private var shown: [ChatController.FileChange] {
    showingAll ? summary.files : Array(summary.files.prefix(Self.shownByDefault))
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      header
      ForEach(shown) { file in
        Divider().opacity(0.25)
        row(file)
          .transition(.opacity.combined(with: .move(edge: .top)))
      }
      if summary.files.count > Self.shownByDefault {
        Divider().opacity(0.25)
        Button {
          withAnimation(.easeOut(duration: 0.2)) { showingAll.toggle() }
        } label: {
          HStack(spacing: 5) {
            Text(
              showingAll
                ? "Show fewer"
                : "Show \(summary.files.count - Self.shownByDefault) more")
            Image(systemName: showingAll ? "chevron.up" : "chevron.down")
              .font(.system(size: 9))
            Spacer(minLength: 0)
          }
          .font(.footnote)
          .foregroundStyle(.secondary)
          .padding(.horizontal, 10)
          .frame(minHeight: Metrics.hit)
          .contentShape(.rect)
        }
        .buttonStyle(.plain)
      }
    }
    .background(.thinMaterial, in: .rect(cornerRadius: 10))
    .overlay {
      RoundedRectangle(cornerRadius: 10)
        .strokeBorder(Color.hairline, lineWidth: 0.5)
    }
    .padding(.leading, 10)
    .padding(.trailing, 44)
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  @ViewBuilder private var header: some View {
    HStack(spacing: 7) {
      Image(systemName: "plusminus.circle")
        .font(.footnote)
        .foregroundStyle(.secondary)
      Text(
        summary.files.count == 1
          ? "Edited 1 file" : "Edited \(summary.files.count) files"
      )
      .font(.system(.footnote, weight: .medium))
      Spacer(minLength: 8)
      counts(added: summary.added, removed: summary.removed)
    }
    .padding(.horizontal, 10)
    .frame(minHeight: 28)
  }

  @ViewBuilder private func row(_ file: ChatController.FileChange) -> some View {
    Button {
      NSWorkspace.shared.activateFileViewerSelecting([resolved(file)])
    } label: {
      HStack(spacing: 7) {
        Image(systemName: "chevron.left.forwardslash.chevron.right")
          .font(.system(size: 9))
          .foregroundStyle(.tertiary)
        Text(file.name)
          .font(.system(.footnote, design: .monospaced))
          .lineLimit(1)
          .truncationMode(.middle)
        Spacer(minLength: 8)
        counts(added: file.added, removed: file.removed)
      }
      .padding(.horizontal, 10)
      .frame(minHeight: 26)
      .contentShape(.rect)
    }
    .buttonStyle(.plain)
    .help(file.path)
  }

  @ViewBuilder private func counts(added: Int, removed: Int) -> some View {
    HStack(spacing: 5) {
      if added > 0 {
        Text("+\(added)")
          .foregroundStyle(Color.diffAdded)
      }
      if removed > 0 {
        Text("-\(removed)")
          .foregroundStyle(Color.diffRemoved)
      }
    }
    .font(.system(.footnote, design: .monospaced))
    .monospacedDigit()
  }

  /// A tool call spells a path the way the model wrote it, which is usually relative to the
  /// folder the turn was working in.
  private func resolved(_ file: ChatController.FileChange) -> URL {
    file.path.hasPrefix("/")
      ? URL(filePath: file.path)
      : (workspace ?? URL(filePath: NSHomeDirectory())).appending(path: file.path)
  }
}
