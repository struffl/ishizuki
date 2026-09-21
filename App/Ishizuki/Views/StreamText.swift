// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Streamed prose, paced: what has arrived is revealed at a steady rate with a fading edge,
// rather than in whatever bursts the decoder happened to hand over.

import IshizukiKit
import SwiftUI

/// Markdown that catches up rather than jumping. The decoder lands text in clumps — a poll's
/// worth at a time — and drawing each clump whole is what makes a reply look like it is being
/// pasted in. This reveals the backlog over a fixed short window, so the words come at the
/// rate they were generated, never slower than the model is writing: the lag it adds is the
/// drain window and nothing more.
@available(macOS 27.0, *)
struct StreamedMarkdown: View {
  let text: String
  let mono: Font
  let size: Double
  /// Off once the row settles, which shows the whole of it at once with no fade.
  var live = false

  /// The whole backlog is drained in this many ticks, so the delay stays bounded however fast
  /// the tokens come: more arriving means bigger steps, not a longer queue.
  private let ticks = 9
  private let interval = Duration.milliseconds(33)
  /// Past this much behind, catching up is not smoothing any more — a row read back from disk
  /// or a tool's output landing whole has no stream to smooth.
  private let snapAt = 400

  @State private var shown = 0

  var body: some View {
    MarkdownText(
      text: revealed, mono: mono, size: size,
      fadeTail: live && shown < text.count ? StreamFade.window : 0
    )
    .task(id: pacing) {
      guard live else {
        shown = text.count
        return
      }
      let total = text.count
      if shown > total || total - shown > snapAt { shown = total }
      while shown < total, !Task.isCancelled {
        let backlog = total - shown
        shown += min(backlog, max(1, Int((Double(backlog) / Double(ticks)).rounded(.up))))
        try? await Task.sleep(for: interval)
      }
    }
  }

  /// Restarting the reveal on every fragment keeps the target current: a task started earlier
  /// would be pacing towards the length the row had when it began.
  private var pacing: String { live ? text : "" }

  private var revealed: String {
    guard live, shown < text.count else { return text }
    return String(text.prefix(shown))
  }
}
