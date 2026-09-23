// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Thinking without the Mac: Apple's on-device model, held on the phone and kept nowhere else.

import IshizukiKit
import IshizukiLink
import SwiftUI

struct LocalChatView: View {
  @State var local: LocalSession

  var body: some View {
    VStack(spacing: 0) {
      ScrollViewReader { scroller in
        ScrollView {
          LazyVStack(alignment: .leading, spacing: 14) {
            if let blocker = local.blocker {
              Text(blocker)
                .font(.body)
                .foregroundStyle(.secondary)
            } else if local.rows.isEmpty {
              Text(
                "Write, learn, and research with this iPhone. Ask for a web search when you need sources. "
                  + "Search queries are sent to Bing; this conversation is not saved to the Mac."
              )
              .font(.body)
              .foregroundStyle(.secondary)
            }
            ForEach(local.rows) { row in
              TranscriptRowView(row: row, live: local.isAnswering && row.id == local.rows.last?.id)
                .id(row.id)
            }
            if let failure = local.failure {
              Text(failure).font(.body).foregroundStyle(.red)
            }
            Color.clear.frame(height: 1).id("local-bottom")
          }
          .padding(16)
        }
        .scrollDismissesKeyboard(.interactively)
        .onChange(of: local.rows.last?.text) {
          withAnimation(.easeOut(duration: 0.18)) {
            scroller.scrollTo("local-bottom", anchor: .bottom)
          }
        }
      }

      HStack(alignment: .bottom, spacing: 8) {
        TextField("Ask this iPhone…", text: $local.draft, axis: .vertical)
          .lineLimit(1...5)
          .padding(.horizontal, 12)
          .padding(.vertical, 8)
          .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 18))
          .disabled(!local.isAvailable)
        Button {
          local.isAnswering ? local.stop() : local.send()
        } label: {
          Image(systemName: local.isAnswering ? "stop.fill" : "arrow.up")
            .font(.system(size: 16.5, weight: .semibold))
            .frame(width: 34, height: 34)
            .background(
              Color.reading.opacity(local.canSend || local.isAnswering ? 1 : 0.3), in: .circle
            )
            .foregroundStyle(.white)
        }
        .disabled(!local.canSend && !local.isAnswering)
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 10)
      .background(.bar)
    }
    .navigationTitle("This iPhone")
    .navigationBarTitleDisplayMode(.inline)
    .toolbar {
      ToolbarItem(placement: .topBarTrailing) {
        Button("Clear") { local.clear() }
          .disabled(local.rows.isEmpty)
      }
    }
  }
}
