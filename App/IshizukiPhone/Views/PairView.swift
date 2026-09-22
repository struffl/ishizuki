// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Pairing: point the camera at the square the Mac is showing, pick a Mac the network is
// advertising, or paste the link when neither works.

import AVFoundation
import IshizukiKit
import IshizukiLink
import SwiftUI

struct PairView: View {
  let store: LinkStore
  let local: LocalSession

  @State private var scanning = false
  @State private var typed = ""
  @State private var failure: String?
  @State private var pairing = false

  var body: some View {
    NavigationStack {
      Form {
        Section {
          NavigationLink {
            LocalChatView(local: local)
          } label: {
            Label("Assistant on this iPhone", systemImage: "iphone.gen3")
          }
          Text("Use the on-device model without pairing a Mac. Web search needs internet.")
            .font(.caption).foregroundStyle(.secondary)
        }
        Section {
          Text(
            "Open Ishizuki on your Mac, then Settings → Companion → Pair a phone. "
              + "Point the camera at the square it shows."
          )
          .font(.callout)
          .foregroundStyle(.secondary)
          Button {
            scanning = true
          } label: {
            Label("Scan the pairing square", systemImage: "qrcode.viewfinder")
          }
        }

        if !store.browser.found.isEmpty {
          Section("On this network") {
            ForEach(store.browser.found) { found in
              LabeledContent(found.name) {
                Text(found.model ?? "—")
                  .font(.system(size: 11, design: .monospaced))
                  .foregroundStyle(.secondary)
              }
            }
            Text(
              "A Mac found here still needs its pairing square once, so the key can be handed over."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
          }
        }

        Section("Or paste the link") {
          TextField("ishizuki://pair/…", text: $typed, axis: .vertical)
            .font(.system(size: 12, design: .monospaced))
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
          Button("Pair") { pair(with: typed) }
            .disabled(typed.isEmpty || pairing)
        }

        if let failure {
          Section {
            Text(failure).font(.callout).foregroundStyle(.red)
          }
        }
      }
      .glassList()
      .navigationTitle("Find your Mac")
      .task {
        store.browser.start()
      }
      .onDisappear { store.browser.stop() }
      .sheet(isPresented: $scanning) {
        QRScannerSheet { text in
          scanning = false
          pair(with: text)
        }
      }
      .overlay {
        if pairing {
          ProgressView("Pairing…")
            .padding(24)
            .background(.regularMaterial, in: .rect(cornerRadius: 14))
        }
      }
    }
  }

  private func pair(with text: String) {
    guard let ticket = LinkTicket(string: text) else {
      failure = "That is not an Ishizuki pairing link."
      return
    }
    pairing = true
    failure = nil
    Task {
      do {
        try await store.pair(with: ticket)
      } catch {
        failure = error.localizedDescription
      }
      pairing = false
    }
  }
}

/// The camera, looking for one square. Handed back as a string, because whether it is a ticket is
/// the pairing view's business rather than the scanner's.
struct QRScannerSheet: View {
  let found: (String) -> Void

  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      QRScanner(found: found)
        .ignoresSafeArea()
        .navigationTitle("Pairing square")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .topBarTrailing) {
            Button("Cancel") { dismiss() }
          }
        }
    }
  }
}

struct QRScanner: UIViewControllerRepresentable {
  let found: (String) -> Void

  func makeUIViewController(context: Context) -> ScannerController {
    let controller = ScannerController()
    controller.found = found
    return controller
  }

  func updateUIViewController(_ controller: ScannerController, context: Context) {
    controller.found = found
  }

  final class ScannerController: UIViewController,
    @preconcurrency AVCaptureMetadataOutputObjectsDelegate
  {
    var found: ((String) -> Void)?

    private let session = AVCaptureSession()
    private var preview: AVCaptureVideoPreviewLayer?
    private var reported = false

    override func viewDidLoad() {
      super.viewDidLoad()
      view.backgroundColor = .black
      guard let device = AVCaptureDevice.default(for: .video),
        let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input)
      else { return }
      session.addInput(input)

      let output = AVCaptureMetadataOutput()
      guard session.canAddOutput(output) else { return }
      session.addOutput(output)
      output.setMetadataObjectsDelegate(self, queue: .main)
      output.metadataObjectTypes = [.qr]

      let layer = AVCaptureVideoPreviewLayer(session: session)
      layer.videoGravity = .resizeAspectFill
      layer.frame = view.bounds
      view.layer.addSublayer(layer)
      preview = layer
    }

    override func viewDidLayoutSubviews() {
      super.viewDidLayoutSubviews()
      preview?.frame = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
      super.viewWillAppear(animated)
      guard !session.isRunning else { return }
      let handle = Handle(session: session)
      DispatchQueue.global(qos: .userInitiated).async { handle.session.startRunning() }
    }

    /// Starting a capture session blocks for long enough to matter, and the framework asks for
    /// it off the main queue. The session itself is not Sendable, so it crosses in a box.
    private struct Handle: @unchecked Sendable {
      let session: AVCaptureSession
    }

    override func viewWillDisappear(_ animated: Bool) {
      super.viewWillDisappear(animated)
      session.stopRunning()
    }

    func metadataOutput(
      _ output: AVCaptureMetadataOutput,
      didOutput objects: [AVMetadataObject],
      from connection: AVCaptureConnection
    ) {
      guard !reported,
        let object = objects.first as? AVMetadataMachineReadableCodeObject,
        let text = object.stringValue
      else { return }
      reported = true
      session.stopRunning()
      found?(text)
    }
  }
}
