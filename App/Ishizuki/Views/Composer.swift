// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Where a turn is written: what was dropped on it, what is being typed, and the one button that
// loads the pack, sends the turn or stops it.

import IshizukiKit
import SwiftUI
import UniformTypeIdentifiers

@available(macOS 27.0, *)
struct Composer: View {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController
  let mono: Font
  /// How tall the composer may grow before it starts scrolling instead, in lines. Worked out
  /// from the transcript's own height, so a long message can be written without the window
  /// deciding how long a thought is allowed to be.
  var maxLines = 6

  /// True while a drag is over the composer, which is the only moment the drop target is worth
  /// drawing: an outline that is always there is an outline nobody reads.
  @State private var targeted = false
  /// What was just sent, held for as long as it takes to fly into the transcript. A message
  /// that simply disappears from the box reads as a message that was lost.
  @State private var flying: Flight?

  private struct Flight: Identifiable, Equatable {
    let id = UUID()
    let text: String
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      if !chat.attachments.isEmpty {
        chips
      }
      if let refused = chat.refusedDrop {
        HStack(spacing: 5) {
          Image(systemName: "exclamationmark.triangle")
          Text("\(refused) could not be attached")
          Spacer(minLength: 0)
          Button("Dismiss") { chat.clearRefusedDrop() }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .font(.footnote)
        .foregroundStyle(.orange)
      }
      field
    }
    .textPlate(radius: 10, horizontal: 12, vertical: 10)
    .overlay(alignment: .top) {
      if let flying {
        FlightBubble(text: flying.text, mono: mono)
          .id(flying.id)
      }
    }
    .animation(.easeOut(duration: 0.22), value: chat.attachments)
    .animation(.easeOut(duration: 0.2), value: chat.isResponding)
    .animation(.easeOut(duration: 0.22), value: chat.refusedDrop)
    .overlay {
      if targeted {
        RoundedRectangle(cornerRadius: 10)
          .strokeBorder(
            Color.reading, style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
      }
    }
    .dropDestination(for: URL.self) { urls, _ in
      chat.attach(urls)
      return !chat.attachments.isEmpty
    } isTargeted: {
      targeted = $0
    }
    .padding(.horizontal, 10)
    .padding(.bottom, 10)
  }

  @ViewBuilder private var field: some View {
    HStack(alignment: .bottom, spacing: 8) {
      // Built to the same geometry as the two buttons on the other end of the row, so all
      // three sit on one line however tall the box has grown.
      Button {
        chat.attach(AttachmentIntake.read(askForFiles()))
      } label: {
        ZStack {
          Circle()
            .fill(Color.primary.opacity(0.08))
            .frame(width: 27, height: 27)
          Image(systemName: "paperclip")
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(.secondary)
        }
        .frame(width: 38, height: 38)
        .contentShape(.circle)
      }
      .buttonStyle(.plain)
      .accessibilityLabel("Attach files")
      .help("Attach files, or drop them here")

      SandboxPicker(chat: chat)

      TextField(
        chat.isResponding ? "Steer the next turn…" : "What needs doing?",
        text: $chat.draft, axis: .vertical
      )
      .textFieldStyle(.plain)
      .font(mono)
      .lineLimit(1...max(2, maxLines))
      // One line sits in the middle of the row rather than on its floor. The buttons are all
      // 38 across and the row is bottom-aligned so they stay put as the box grows, which
      // means a single line of text has to claim the same height to share their centre.
      .frame(minHeight: 38)
      // Return sends, shift-return opens a line. Handled here rather than through onSubmit,
      // which cannot tell the two apart.
      .onKeyPress(phases: .down) { press in
        guard press.key == .return else { return .ignored }
        // Shift-return is left to the field itself, which opens a line at the caret rather
        // than at the end of whatever has been typed.
        guard !press.modifiers.contains(.shift) else { return .ignored }
        submit()
        return .handled
      }
      .onPasteCommand(of: [.fileURL, .png, .tiff]) { _ in chat.pasteAttachment() }

      if chat.isResponding {
        // Built to the send ring's own geometry rather than to a glyph's: the two sit side by
        // side while a turn runs, and a button half the size of its neighbour reads as an
        // afterthought — which stopping a turn is not.
        Button {
          chat.stop()
        } label: {
          ZStack {
            Circle()
              .fill(Color.primary.opacity(0.1))
              .frame(width: 27, height: 27)
            Circle()
              .stroke(.quaternary, lineWidth: 2.5)
              .frame(width: 34, height: 34)
            Image(systemName: "stop.fill")
              .font(.system(size: 11, weight: .semibold))
              .foregroundStyle(.secondary)
          }
          .frame(width: 38, height: 38)
          .contentShape(.circle)
        }
        .buttonStyle(.plain)
        .transition(.opacity.combined(with: .scale(scale: 0.8)))
        .accessibilityLabel("Stop this turn")
        .help("Stop this turn")
      }

      SendRing(chat: chat, controller: controller, onSend: submit)
    }
  }

  /// Sending, with the text kept back long enough to watch it go. Steering is not a flight:
  /// what is queued stays on screen above the composer, so it never left.
  private func submit() {
    let typed = chat.draft.trimmingCharacters(in: .whitespacesAndNewlines)
    let sending = chat.submission == .send
    chat.submit()
    guard sending, !typed.isEmpty else { return }
    let flight = Flight(text: typed)
    flying = flight
    Task {
      try? await Task.sleep(for: .milliseconds(360))
      if flying == flight { flying = nil }
    }
  }

  @ViewBuilder private var chips: some View {
    VStack(alignment: .leading, spacing: 4) {
      ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 6) {
          ForEach(chat.attachments) { attachment in
            AttachmentChip(attachment: attachment) { chat.detach(attachment) }
          }
        }
      }
      if chat.needsVision, controller.activeEntry?.hasVision == false {
        Label(
          "\(controller.activeEntry?.displayName ?? "This pack") has no vision tower — "
            + "the pictures will be named but not seen.",
          systemImage: "eye.slash"
        )
        .font(.footnote)
        .foregroundStyle(Color.instructing)
      }
    }
  }

  private func askForFiles() -> [URL] {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = false
    panel.canChooseFiles = true
    panel.allowsMultipleSelection = true
    panel.prompt = "Attach"
    panel.directoryURL = chat.workspace
    guard panel.runModal() == .OK else { return [] }
    return panel.urls
  }
}

/// One attached thing: a picture shows itself, anything else shows what it is.
@available(macOS 27.0, *)
struct AttachmentChip: View {
  let attachment: Attachment
  let onRemove: () -> Void

  @State private var thumbnail: Image?

  var body: some View {
    HStack(spacing: 6) {
      if let thumbnail {
        thumbnail
          .resizable()
          .aspectRatio(contentMode: .fill)
          .frame(width: 22, height: 22)
          .clipShape(.rect(cornerRadius: 4))
      } else {
        Image(systemName: attachment.icon)
          .font(.footnote)
          .foregroundStyle(.secondary)
          .frame(width: 22, height: 22)
      }
      VStack(alignment: .leading, spacing: 0) {
        Text(attachment.name)
          .font(.system(.footnote, weight: .medium))
          .lineLimit(1)
          .truncationMode(.middle)
        Text(ReadoutFormat.compact(attachment.byteCount))
          .font(.system(size: 9, design: .monospaced))
          .foregroundStyle(.tertiary)
      }
      .frame(maxWidth: 140, alignment: .leading)
      Button(action: onRemove) {
        Image(systemName: "xmark")
          .font(.system(size: 8, weight: .semibold))
          .foregroundStyle(.secondary)
          .hitTarget(16)
      }
      .buttonStyle(.plain)
      .accessibilityLabel("Remove \(attachment.name)")
    }
    .padding(.leading, 5)
    .padding(.trailing, 4)
    .padding(.vertical, 4)
    .background(.quaternary, in: .rect(cornerRadius: 7))
    .task(id: attachment.url) {
      guard attachment.kind == .image else { return }
      let url = attachment.url
      let data = await Task.detached(priority: .utility) { Thumbnail.png(of: url) }.value
      guard let data, let image = NSImage(data: data) else { return }
      thumbnail = Image(nsImage: image)
    }
  }
}

/// The dial, become a button. It used to sit in the corner saying how much context was spent;
/// the bar over the composer says that now, and the ring here has the thing worth watching
/// while you wait — how far this turn has got.
@available(macOS 27.0, *)
struct SendRing: View {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController
  var onSend: () -> Void = {}

  private var glyph: String {
    switch chat.submission {
    case .chooseFolder: "folder"
    case .load: "bolt.fill"
    case .loading: "hourglass"
    case .steer: "arrow.turn.down.right"
    case .answer: "arrowshape.turn.up.left.fill"
    case .switchBack: "arrow.triangle.2.circlepath"
    case .missingModel: "arrow.triangle.branch"
    case .busy: "hourglass"
    case .send, .nothingToSay: "arrow.up"
    }
  }

  private var tint: Color {
    switch chat.activity {
    case .reading: Color.reading
    case .writing, .writingCommand: .generating
    case .queued, .unknown: chat.canSubmit ? Color.mine : .secondary
    }
  }

  var body: some View {
    Button {
      onSend()
    } label: {
      ZStack {
        Circle()
          .fill(chat.canSubmit ? Color.mine : Color.primary.opacity(0.08))
          .frame(width: 27, height: 27)

        Circle()
          .stroke(.quaternary, lineWidth: 2.5)
          .frame(width: 34, height: 34)
        if let prefill = chat.prefillFraction {
          Circle()
            .trim(from: 0, to: max(0.02, min(1, prefill)))
            .stroke(Color.reading, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
            .rotationEffect(.degrees(-90))
            .frame(width: 34, height: 34)
        } else if chat.isGenerating {
          Circle()
            .stroke(Color.generating, lineWidth: 2.5)
            .frame(width: 34, height: 34)
        }

        Image(systemName: glyph)
          .font(.system(size: 12, weight: .semibold))
          .foregroundStyle(chat.canSubmit ? .white : Color.secondary)
      }
      .frame(width: 38, height: 38)
      .contentShape(.circle)
    }
    .buttonStyle(.plain)
    .disabled(!chat.canSubmit)
    .animation(.easeOut(duration: 0.2), value: chat.prefillFraction)
    .animation(.easeOut(duration: 0.18), value: chat.canSubmit)
    .contentTransition(.symbolEffect(.replace))
    .accessibilityLabel(chat.submissionLabel)
    .help(helpText)
  }

  private var helpText: String {
    switch chat.activity {
    case .reading(let fraction):
      "Reading" + (fraction.map { " — \(ReadoutFormat.percent($0))" } ?? "")
    case .writing: "Writing"
    case .writingCommand: "Writing a command"
    case .queued, .unknown: chat.submissionLabel
    }
  }
}

/// The message on its way out of the box and into the transcript: it rises, shrinks a little
/// and fades, so sending reads as the text moving rather than as the text vanishing.
@available(macOS 27.0, *)
struct FlightBubble: View {
  let text: String
  let mono: Font

  @State private var lifted = false

  var body: some View {
    Text(text)
      .font(mono)
      .lineLimit(3)
      .truncationMode(.tail)
      .foregroundStyle(.white)
      .bubble(mine: true)
      .frame(maxWidth: 340, alignment: .trailing)
      .frame(maxWidth: .infinity, alignment: .trailing)
      .offset(y: lifted ? -64 : 4)
      .scaleEffect(lifted ? 0.9 : 1, anchor: .bottomTrailing)
      .opacity(lifted ? 0 : 1)
      .allowsHitTesting(false)
      .accessibilityHidden(true)
      .onAppear {
        withAnimation(.easeOut(duration: 0.34)) { lifted = true }
      }
  }
}
