// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The companion, as a settings section and a pairing sheet: a square to point a phone at, and a
// list of the phones that have been let in.

import CoreImage.CIFilterBuiltins
import IshizukiKit
import IshizukiLink
import SwiftUI

@available(macOS 27.0, *)
struct CompanionSection: View {
  @Bindable var companion: CompanionServer
  @State private var pairing = false

  private var settings: CompanionSettings { companion.settings }

  var body: some View {
    Section("Companion") {
      Toggle("Let an iPhone use this Mac", isOn: running)
      if case .failed(let message) = companion.phase {
        Text(message).font(.caption).foregroundStyle(.red)
      }
      TextField("Name on the network", text: Bindable(settings).serviceName)
      TextField(
        "Companion port", value: Bindable(settings).port, format: .number.grouping(.never))
      Toggle("Share the shell as well as files", isOn: Bindable(settings).allowShell)
      LabeledContent("Folders shared") {
        HStack(spacing: 8) {
          Text(sharedSummary)
            .font(.system(.subheadline, design: .monospaced))
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.head)
          Button("Choose…") { chooseRoots() }
            .buttonStyle(.borderless)
        }
      }
      LabeledContent("Key") {
        Text(companion.fingerprint)
          .font(.system(.subheadline, design: .monospaced))
          .foregroundStyle(.secondary)
      }

      HStack {
        Button("Pair a phone…") {
          companion.openPairing()
          pairing = true
        }
        .disabled(!companion.phase.isRunning)
        Spacer()
        if !companion.devices.isEmpty {
          Button("Forget every phone", role: .destructive) { companion.forgetEverything() }
            .buttonStyle(.borderless)
        }
      }

      ForEach(companion.devices) { device in
        LabeledContent(device.name) {
          HStack(spacing: 8) {
            Text(device.system)
              .font(.system(.footnote, design: .monospaced))
              .foregroundStyle(.tertiary)
            Button("Forget") { companion.forget(device) }
              .buttonStyle(.borderless)
          }
        }
      }
    }
    .sheet(isPresented: $pairing) {
      PairingSheet(companion: companion) {
        companion.closePairing()
        pairing = false
      }
    }
  }

  private var running: Binding<Bool> {
    Binding(
      get: { companion.phase.isRunning },
      set: { wanted in wanted ? companion.start() : companion.stop() })
  }

  private var sharedSummary: String {
    let roots = settings.allowedRoots
    guard let first = roots.first else { return "none" }
    return roots.count > 1
      ? "\(first.lastPathComponent) and \(roots.count - 1) more" : first.path
  }

  private func chooseRoots() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = true
    panel.prompt = "Share"
    panel.directoryURL = FileManager.default.homeDirectoryForCurrentUser
    guard panel.runModal() == .OK, !panel.urls.isEmpty else { return }
    settings.roots = panel.urls.map(\.path)
  }
}

@available(macOS 27.0, *)
struct PairingSheet: View {
  let companion: CompanionServer
  let done: () -> Void

  var body: some View {
    VStack(spacing: 16) {
      Text("Point the phone's camera at this")
        .font(.headline)

      if let ticket = companion.ticket, let url = ticket.url {
        if let image = QRCode.image(for: url.absoluteString, side: 260) {
          Image(nsImage: image)
            .interpolation(.none)
            .resizable()
            .frame(width: 260, height: 260)
            .background(.white)
            .clipShape(.rect(cornerRadius: 12))
        }
        VStack(spacing: 4) {
          Text(ticket.hosts.first ?? "—")
            .font(.system(.callout, design: .monospaced))
          Text("port \(String(ticket.port)) · key \(companion.fingerprint)")
            .font(.system(.subheadline, design: .monospaced))
            .foregroundStyle(.secondary)
        }
        if let closes = companion.pairingCloses {
          Text("Pairing closes \(closes, style: .relative) from now")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        Button("Copy the link instead") {
          NSPasteboard.general.clearContents()
          NSPasteboard.general.setString(url.absoluteString, forType: .string)
        }
        .buttonStyle(.borderless)
      } else {
        Text("Pairing has closed.")
          .foregroundStyle(.secondary)
      }

      Button("Done", action: done)
        .keyboardShortcut(.defaultAction)
    }
    .padding(24)
    .frame(width: 340)
  }
}

/// A square drawn from a string, at the size it will be shown rather than scaled up from the
/// handful of points Core Image hands back.
enum QRCode {
  static func image(for text: String, side: CGFloat) -> NSImage? {
    let filter = CIFilter.qrCodeGenerator()
    filter.message = Data(text.utf8)
    filter.correctionLevel = "M"
    guard let output = filter.outputImage else { return nil }
    let scale = side / output.extent.width
    let scaled = output.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    let context = CIContext()
    guard let cgImage = context.createCGImage(scaled, from: scaled.extent) else { return nil }
    return NSImage(cgImage: cgImage, size: NSSize(width: side, height: side))
  }
}
