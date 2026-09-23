// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The line over the composer: how much context is spent and on what, which branch the folder is
// on, and whether this conversation has a checkout of its own.

import IshizukiKit
import SwiftUI

/// The four things context is spent on, in the order they are laid down. Kept together so the
/// bar and the breakdown behind it can never disagree about which colour means what.
@available(macOS 27.0, *)
enum ContextPart: CaseIterable {
  case instructions
  case toolSchemas
  case conversation
  case free

  var label: String {
    switch self {
    case .instructions: "instructions"
    case .toolSchemas: "tool schemas"
    case .conversation: "conversation"
    case .free: "free"
    }
  }

  var tint: Color {
    switch self {
    case .instructions: Color.instructing
    case .toolSchemas: Color.reading
    case .conversation: Color.generating
    case .free: Color.ink.opacity(0.09)
    }
  }

  /// What this part is, in the one sentence that would make someone do something about it.
  var note: String {
    switch self {
    case .instructions: "The same every turn. Shortening them shortens every prefill."
    case .toolSchemas: "One block per tool, also the same every turn."
    case .conversation: "What has been said and done so far. A new chat clears it."
    case .free: "What is left before the window has to be trimmed."
    }
  }

  func tokens(of use: ChatController.ContextUse) -> Int {
    switch self {
    case .instructions: use.instructions
    case .toolSchemas: use.toolSchemas
    case .conversation: use.conversation
    case .free: use.free
    }
  }
}

/// Context as a bar rather than a dial: a thin band that fills left to right, banded by what
/// each stretch of it went on. Clicking it opens the same reading in full.
@available(macOS 27.0, *)
struct ContextBar: View {
  let use: ChatController.ContextUse

  @State private var showingBreakdown = false

  private var spent: [ContextPart] {
    [.instructions, .toolSchemas, .conversation]
  }

  var body: some View {
    Button {
      showingBreakdown.toggle()
    } label: {
      HStack(spacing: 8) {
        Text("context")
          .font(.system(size: 11))
          .foregroundStyle(.tertiary)
        band
          .frame(height: 6)
          .frame(maxWidth: .infinity)
        Text(ReadoutFormat.percent(use.fraction))
          .font(.system(size: 11, weight: .medium))
          .foregroundStyle(use.fraction > 0.9 ? Color.clay : .secondary)
          .monospacedDigit()
      }
      .frame(minHeight: Metrics.hit)
      .contentShape(.rect)
    }
    .buttonStyle(.plain)
    .accessibilityLabel(
      "Context, \(ReadoutFormat.percent(use.fraction)) of "
        + "\(ReadoutFormat.group(use.ceiling)) tokens"
    )
    .help("\(ReadoutFormat.group(use.used)) of \(ReadoutFormat.group(use.ceiling)) tokens held")
    .popover(isPresented: $showingBreakdown, arrowEdge: .top) {
      ContextBreakdown(use: use)
    }
  }

  /// One rectangle per part, laid end to end. Widths come from the geometry rather than from a
  /// stack of flexible frames, so a part worth less than a point still takes one and the band
  /// never comes up short of its own width.
  @ViewBuilder private var band: some View {
    GeometryReader { frame in
      let width = frame.size.width
      let total = max(use.ceiling, use.used, 1)
      HStack(spacing: 0) {
        ForEach(spent, id: \.self) { part in
          let share = Double(part.tokens(of: use)) / Double(total)
          Rectangle()
            .fill(part.tint)
            .frame(width: share > 0 ? max(1, width * share) : 0)
        }
        Rectangle()
          .fill(ContextPart.free.tint)
      }
      .clipShape(.capsule)
    }
    .animation(.easeOut(duration: 0.25), value: use)
  }
}

/// The bar opened up: the window, then every part of it with its own share. Compact on
/// purpose — the point is to see at a glance which band is the one worth doing something
/// about, not to read four paragraphs about context.
@available(macOS 27.0, *)
struct ContextBreakdown: View {
  let use: ChatController.ContextUse

  var body: some View {
    VStack(alignment: .leading, spacing: 11) {
      HStack(alignment: .firstTextBaseline, spacing: 9) {
        Text("Context window")
          .font(.system(.body, weight: .medium))
        Spacer(minLength: 12)
        Text(
          "\(Self.tokens(use.used)) / \(Self.tokens(use.ceiling)) "
            + "(\(ReadoutFormat.percent(use.fraction)))"
        )
        .font(.body.monospacedDigit())
        .foregroundStyle(.secondary)
        .monospacedDigit()
      }

      stack
        .frame(height: 9)

      VStack(spacing: 7) {
        ForEach(ContextPart.allCases, id: \.self) { part in
          row(part)
        }
      }
    }
    .padding(15)
    .frame(width: 374)
  }

  @ViewBuilder private var stack: some View {
    GeometryReader { frame in
      let width = frame.size.width
      let total = max(use.ceiling, use.used, 1)
      HStack(spacing: 1) {
        ForEach(ContextPart.allCases, id: \.self) { part in
          let share = Double(part.tokens(of: use)) / Double(total)
          Rectangle()
            .fill(part.tint)
            .frame(width: share > 0 ? max(2, width * share) : 0)
        }
      }
      .clipShape(.rect(cornerRadius: 3))
    }
  }

  @ViewBuilder private func row(_ part: ContextPart) -> some View {
    let tokens = part.tokens(of: use)
    let share = use.ceiling > 0 ? Double(tokens) / Double(use.ceiling) : 0
    HStack(spacing: 9) {
      RoundedRectangle(cornerRadius: 2)
        .fill(part.tint)
        .frame(width: 10, height: 11)
      Text(part.label)
        .font(.callout)
      Spacer(minLength: 12)
      Text(Self.tokens(tokens))
        .font(.callout.monospacedDigit())
        .foregroundStyle(.secondary)
        .monospacedDigit()
        .frame(width: 64, alignment: .trailing)
      Text(ReadoutFormat.percent(share))
        .font(.callout.monospacedDigit())
        .foregroundStyle(.secondary)
        .monospacedDigit()
        .frame(width: 48, alignment: .trailing)
    }
    .help(part.note)
  }

  /// Tokens the way a window's worth of them is spoken: 72.3k rather than 72 300.
  static func tokens(_ count: Int) -> String {
    if count >= 1_000_000 { return String(format: "%.1fM", Double(count) / 1_000_000) }
    if count >= 1_000 { return String(format: "%.1fk", Double(count) / 1_000) }
    return "\(count)"
  }
}

/// Which branch the folder is on and what is waiting on it. Present only when the folder is a
/// repository, which is the whole of what makes this automatic.
@available(macOS 27.0, *)
struct GitChip: View {
  @Bindable var chat: ChatController

  var body: some View {
    if let status = chat.git.status {
      Menu {
        Text(status.root.path(percentEncoded: false))
        Divider()
        Button("Refresh") { chat.git.refresh() }
        Button("Reveal in Finder") {
          NSWorkspace.shared.activateFileViewerSelecting([status.root])
        }
      } label: {
        HStack(spacing: 4) {
          Image(
            systemName: status.isLinkedWorktree
              ? "arrow.triangle.branch" : "point.3.filled.connected.trianglepath.dotted"
          )
          .font(.system(size: 10))
          Text(status.detached ? "detached" : status.branch)
            .font(.system(size: 11, weight: .medium))
            .lineLimit(1)
          if status.behind > 0 {
            count("arrow.down", status.behind, Color.reading)
          }
          if status.ahead > 0 {
            count("arrow.up", status.ahead, Color.generating)
          }
          if status.pendingCount > 0 {
            count("pencil", status.pendingCount, Color.instructing)
          }
        }
        .foregroundStyle(.secondary)
      }
      .menuStyle(.borderlessButton)
      .menuIndicator(.hidden)
      .fixedSize()
      .frame(minHeight: Metrics.hit)
      .accessibilityLabel("Git branch \(status.branch)")
      .help(helpText(status))
    }
  }

  private func count(_ icon: String, _ value: Int, _ tint: Color) -> some View {
    HStack(spacing: 1) {
      Image(systemName: icon)
        .font(.system(size: 9))
      Text("\(value)")
        .font(.system(size: 11))
        .monospacedDigit()
    }
    .foregroundStyle(tint)
  }

  private func helpText(_ status: GitStatus) -> String {
    var parts: [String] = []
    if !status.hasUpstream {
      parts.append("no upstream")
    } else {
      if status.behind > 0 { parts.append("\(status.behind) to pull") }
      if status.ahead > 0 { parts.append("\(status.ahead) to push") }
      if status.ahead == 0, status.behind == 0 { parts.append("up to date") }
    }
    if status.changed > 0 { parts.append("\(status.changed) changed") }
    if status.untracked > 0 { parts.append("\(status.untracked) untracked") }
    if status.conflicted > 0 { parts.append("\(status.conflicted) conflicted") }
    if status.isClean { parts.append("clean") }
    return parts.joined(separator: " · ")
  }
}

/// A checkout of its own for this conversation. Offered the moment the folder turns out to be a
/// repository, and never taken without being asked for: a worktree is a change to someone's
/// repository, not a display preference.
@available(macOS 27.0, *)
struct WorktreeChip: View {
  @Bindable var chat: ChatController

  private var current: GitWorktree? {
    guard let workspace = chat.workspace else { return nil }
    return chat.git.worktrees.first {
      $0.path.standardizedFileURL == workspace.standardizedFileURL
    }
  }

  var body: some View {
    if chat.git.isRepository {
      Menu {
        Button("New worktree for this chat") { chat.makeWorktree() }
          .disabled(chat.isRunningTurn)
        if chat.git.worktrees.count > 1 {
          Divider()
          ForEach(chat.git.worktrees) { worktree in
            Button {
              chat.use(worktree)
            } label: {
              if worktree.id == current?.id {
                Label(name(worktree), systemImage: "checkmark")
              } else {
                Text(name(worktree))
              }
            }
          }
        }
        if let current, !current.isPrimary {
          Divider()
          Button("Remove \(current.name)", role: .destructive) {
            chat.removeWorktree(current)
          }
          .disabled(chat.isRunningTurn)
        }
      } label: {
        HStack(spacing: 4) {
          Image(systemName: isDetached ? "square.on.square.dashed" : "square.on.square")
            .font(.system(size: 10))
          Text(isDetached ? current?.name ?? "worktree" : "worktree")
            .font(.system(size: 11, weight: isDetached ? .medium : .regular))
            .lineLimit(1)
        }
        .foregroundStyle(isDetached ? Color.reading : Color.secondary)
      }
      .menuStyle(.borderlessButton)
      .menuIndicator(.hidden)
      .fixedSize()
      .frame(minHeight: Metrics.hit)
      .accessibilityLabel("Worktree")
      .help(
        isDetached
          ? "This chat is working in its own checkout"
          : "Give this chat a checkout of its own")
    }
  }

  private var isDetached: Bool {
    guard let current else { return false }
    return !current.isPrimary
  }

  private func name(_ worktree: GitWorktree) -> String {
    worktree.isPrimary ? "\(worktree.name) (main checkout)" : worktree.name
  }
}

/// The whole line: context on the left, where the work is happening on the right.
@available(macOS 27.0, *)
struct ContextStrip: View {
  @Bindable var chat: ChatController

  var body: some View {
    HStack(spacing: 11) {
      ContextBar(use: chat.contextUse)
        .frame(maxWidth: 260)
      Spacer(minLength: 6)
      SandboxChip(chat: chat)
      folder
      if chat.git.isRepository {
        Divider().frame(height: 12)
        GitChip(chat: chat)
        WorktreeChip(chat: chat)
      }
    }
    .chipPlate(radius: 9, horizontal: 10, vertical: 2)
    .padding(.horizontal, 11)
  }

  /// The folder, still reachable but no longer a bar of its own: picking one is the sidebar's
  /// job now, and this is here to say which one is in force.
  @ViewBuilder private var folder: some View {
    Button {
      chat.chooseWorkspace()
    } label: {
      HStack(spacing: 4) {
        Image(systemName: "folder")
          .font(.system(size: 10))
        Text(chat.workspace?.lastPathComponent ?? "Choose a folder…")
          .font(.system(size: 11, weight: .medium))
          .lineLimit(1)
      }
      .foregroundStyle(chat.workspace == nil ? Color.instructing : .secondary)
      .frame(minHeight: Metrics.hit)
      .contentShape(.rect)
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Working folder")
    .help(chat.workspace?.path ?? "The one directory the agent may touch")
  }
}
