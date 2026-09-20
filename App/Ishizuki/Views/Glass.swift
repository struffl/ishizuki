// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The glass the panels are cut from — regular rather than clear, so text keeps its scrim.

import SwiftUI

struct GlassCard<Content: View>: View {
  var padding: CGFloat = 12
  var radius: CGFloat = 14
  @ViewBuilder var content: Content

  var body: some View {
    content
      .padding(padding)
      .frame(maxWidth: .infinity, alignment: .leading)
      .glassEffect(.regular, in: .rect(cornerRadius: radius))
  }
}

struct GlassSection<Content: View>: View {
  let title: String
  @ViewBuilder var content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(title)
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(.secondary)
        .textCase(.uppercase)
        .padding(.leading, 4)
      GlassCard { content }
    }
  }
}
