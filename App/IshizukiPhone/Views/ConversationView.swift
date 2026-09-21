// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A conversation on the Mac, watched from the phone: its rows as they fill, what the turn is
// doing, and a composer that steers a turn already running rather than queueing behind it.

import IshizukiKit
import IshizukiLink
import SwiftUI

struct ConversationView: View {
  @Environment(\.scenePhase) private var scenePhase
  @State var model: ConversationModel
  let chats: ChatsModel

  @FocusState private var writing: Bool

  var body: some View {
    VStack(spacing: 0) {
      transcript
      composer
    }
    .navigationTitle(model.summary?.title ?? "Conversation")
    .navigationBarTitleDisplayMode(.inline)
    .toolbar {
      ToolbarItem(placement: .topBarTrailing) {
        if model.isRunning {
          Button {
            Task { await model.stopTurn() }
          } label: {
            Image(systemName: "stop.circle")
          }
        }
      }
    }
    .task {
      model.start()
    }
    .onChange(of: scenePhase) { _, phase in
      if phase == .active { model.start() } else { model.stop() }
    }
    .onDisappear {
      model.stop()
      if let summary = model.summary { chats.absorb(summary) }
    }
  }

  private var transcript: some View {
    ScrollViewReader { scroller in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 14) {
          ForEach(model.visibleRows) { row in
            TranscriptRowView(row: row, live: isLive(row))
              .id(row.id)
          }
          if let failure = model.failure {
            Text(failure)
              .font(.callout)
              .foregroundStyle(.red)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
          Color.clear.frame(height: 1).id(Self.bottom)
        }
        .padding(16)
      }
      .scrollDismissesKeyboard(.interactively)
      .onChange(of: model.rows.last?.text) {
        withAnimation(.easeOut(duration: 0.18)) { scroller.scrollTo(Self.bottom, anchor: .bottom) }
      }
      .onChange(of: model.visibleRows.count) {
        withAnimation(.easeOut(duration: 0.18)) { scroller.scrollTo(Self.bottom, anchor: .bottom) }
      }
    }
  }

  private static let bottom = "transcript-bottom"

  private func isLive(_ row: TranscriptRow) -> Bool {
    model.isRunning && row.id == model.visibleRows.last?.id
  }

  private var composer: some View {
    VStack(spacing: 8) {
      if model.showingCached {
        Label("Saved copy · reconnecting to your Mac", systemImage: "clock.arrow.circlepath")
          .font(.caption).foregroundStyle(.secondary)
      }
      ActivityLine(activity: model.activity, status: model.status, steers: model.steers.count)

      HStack(alignment: .bottom, spacing: 8) {
        TextField(
          model.isRunning ? "Steer this turn…" : "Ask the Mac…", text: $model.draft,
          axis: .vertical
        )
        .focused($writing)
        .lineLimit(1...5)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 18))

        Button {
          Task { await model.send() }
        } label: {
          Image(systemName: model.isRunning ? "arrow.turn.down.right" : "arrow.up")
            .font(.system(size: 15, weight: .semibold))
            .frame(width: 34, height: 34)
            .background(Color.reading.opacity(model.canSend ? 1 : 0.3), in: .circle)
            .foregroundStyle(.white)
        }
        .disabled(!model.canSend)
      }
    }
    .padding(.horizontal, 12)
    .padding(.vertical, 10)
    .background(.bar)
  }
}

/// What the turn is doing, in one line: reading, writing, or writing a command, with the dials
/// behind it when there is nothing happening.
struct ActivityLine: View {
  let activity: LinkActivity
  let status: LinkStatus?
  let steers: Int

  var body: some View {
    HStack(spacing: 8) {
      switch activity.phase {
      case .idle:
        if let status {
          Text(status.model ?? "no pack")
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(.tertiary)
          if status.tokensPerSecond > 0 {
            Text(String(format: "%.0f tok/s", status.tokensPerSecond))
              .font(.system(size: 10, design: .monospaced))
              .foregroundStyle(.tertiary)
          }
        }
      case .queued:
        label("Queued")
      case .reading:
        if let fraction = activity.fraction {
          label("Reading \(ReadoutFormat.percent(fraction))")
        } else {
          label("Reading")
        }
      case .writing:
        label("Writing")
      case .command:
        label(activity.command?.isEmpty == false ? activity.command! : "Writing a command")
      case .unknown:
        label("Working")
      }
      Spacer()
      if steers > 0 {
        Text("\(steers) queued")
          .font(.system(size: 10, design: .monospaced))
          .foregroundStyle(Color.instructing)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private func label(_ text: String) -> some View {
    HStack(spacing: 6) {
      AnimatedDots(size: 3, tint: Color.reading)
      Text(text)
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .truncationMode(.middle)
    }
  }
}
