// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The photograph the window's glass sits in front of, cropped to fill rather than letterboxed,
// so every material on top of it has something with real color and shape to frost.

import SwiftUI

#if os(macOS)
  import AppKit
#else
  import UIKit
#endif

private let backdropImage: Image = {
  #if os(macOS)
    guard let url = Bundle.main.url(forResource: "Backdrop", withExtension: "jpg"),
      let image = NSImage(contentsOf: url)
    else { return Image(systemName: "photo") }
    return Image(nsImage: image)
  #else
    guard let url = Bundle.main.url(forResource: "Backdrop-iOS", withExtension: "jpg"),
      let data = try? Data(contentsOf: url), let image = UIImage(data: data)
    else { return Image(systemName: "photo") }
    return Image(uiImage: image)
  #endif
}()

/// The window's own background: the photo, filled and cropped to the frame it is given, with a
/// glassy sheen laid over it so the materials floating above have depth to catch.
struct WindowBackdrop: View {
  var body: some View {
    backdropImage
      .resizable()
      .aspectRatio(contentMode: .fill)
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      // Scaled up before the blur samples past its edges, so the blur has margin to draw from
      // instead of smearing in transparency at the frame's border.
      .scaleEffect(1.1)
      .blur(radius: 12)
      .clipped()
      .overlay { Sheen() }
      .ignoresSafeArea()
  }
}

/// A diagonal streak of light and a darkened vignette, the two cues that read as "glass" rather
/// than "photo with a blur filter on it".
private struct Sheen: View {
  var body: some View {
    LinearGradient(
      stops: [
        .init(color: .white.opacity(0.22), location: 0),
        .init(color: .white.opacity(0), location: 0.35),
        .init(color: .white.opacity(0), location: 0.7),
        .init(color: .black.opacity(0.16), location: 1),
      ],
      startPoint: .topLeading, endPoint: .bottomTrailing
    )
    .blendMode(.plusLighter)
    RadialGradient(
      colors: [.black.opacity(0), .black.opacity(0.22)],
      center: .center, startRadius: 200, endRadius: 900
    )
  }
}
