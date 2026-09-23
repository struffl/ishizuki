// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Where this conversation's commands run, chosen beside the composer's own buttons: this Mac,
// a Linux VM with the folder shared in, or a pod on a cluster. A VM is a real purchase, so
// what it costs is picked here and what it is doing is shown over the box.

import IshizukiKit
import SwiftUI

@available(macOS 27.0, *)
struct SandboxPicker: View {
  @Bindable var chat: ChatController

  @State private var open = false
  @State private var typedImage = ""

  private var choice: SandboxChoice { chat.sandboxChoice }

  var body: some View {
    Button {
      typedImage = choice.image
      open.toggle()
    } label: {
      ZStack {
        Circle()
          .fill(Color.primary.opacity(choice.isSandboxed ? 0.14 : 0.08))
          .frame(width: 27, height: 30)
        Image(systemName: choice.kind.glyph)
          .font(.system(size: 13, weight: .medium))
          .foregroundStyle(choice.isSandboxed ? Color.reading : .secondary)
      }
      .frame(width: 38, height: 42)
      .contentShape(.circle)
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Where commands run")
    .help("Commands run in \(choice.summary)")
    .popover(isPresented: $open, arrowEdge: .bottom) {
      form
        .frame(width: 352)
        .padding(15)
    }
  }

  @ViewBuilder private var form: some View {
    VStack(alignment: .leading, spacing: 13) {
      Picker("Run commands in", selection: kind) {
        ForEach(SandboxChoice.Kind.allCases, id: \.self) { kind in
          Label(kind.label, systemImage: kind.glyph).tag(kind)
        }
      }
      .pickerStyle(.inline)
      .labelsHidden()

      switch choice.kind {
      case .native:
        Text("This Mac, this user, no boundary. Fast, and nothing is fenced off.")
          .font(.subheadline)
          .foregroundStyle(.secondary)
      case .container:
        container
      case .cluster:
        cluster
      }
    }
  }

  @ViewBuilder private var container: some View {
    VStack(alignment: .leading, spacing: 11) {
      Text(
        "A Linux VM on this Mac. The folder is shared into it at /workspace; the rest of the Mac is not."
      )
      .font(.subheadline)
      .foregroundStyle(.secondary)

      HStack(spacing: 7) {
        TextField("image", text: $typedImage)
          .font(.system(size: 12))
          .onSubmit { commitImage() }
        Menu {
          ForEach(chat.sandboxes.settings.recentImages, id: \.self) { image in
            Button(image) {
              typedImage = image
              commitImage()
            }
          }
        } label: {
          Image(systemName: "clock.arrow.circlepath")
        }
        .menuStyle(.borderlessButton)
        .frame(width: 29)
      }

      Picker("Architecture", selection: architecture) {
        ForEach(SandboxChoice.Architecture.allCases, id: \.self) { arch in
          Text(arch.label).tag(arch)
        }
      }
      .pickerStyle(.segmented)

      slider(
        title: "CPUs", value: cpus, range: SandboxLimits.cpuRange,
        caption: "\(choice.cpus) of \(SandboxLimits.cores), two left for the host")

      slider(
        title: "Memory", value: memoryGigabytes, range: memoryRange,
        caption: "\(choice.memoryBytes / (1024 * 1024 * 1024)) GB, "
          + "\(memoryRange.upperBound) GB spare after the pack and the system")

      if !SandboxArtifacts.isReady, chat.sandboxes.settings.kernelPath.isEmpty {
        Text("First boot fetches a Linux kernel automatically; after that it's cached.")
          .font(.subheadline)
          .foregroundStyle(.secondary)
      }
    }
  }

  @ViewBuilder private var cluster: some View {
    VStack(alignment: .leading, spacing: 11) {
      Text(
        "A pod on your cluster, through kubectl. The folder is copied in; the pod outlives this app, so a build keeps going."
      )
      .font(.subheadline)
      .foregroundStyle(.secondary)

      TextField("image", text: $typedImage)
        .font(.system(size: 12))
        .onSubmit { commitImage() }
      TextField("context", text: context)
        .font(.system(size: 12))
      TextField("namespace", text: namespace)
        .font(.system(size: 12))

      slider(title: "CPUs", value: cpus, range: 1...32, caption: "\(choice.cpus) requested")
      slider(
        title: "Memory", value: memoryGigabytes, range: 1...128,
        caption: "\(choice.memoryBytes / (1024 * 1024 * 1024)) GB requested")
    }
  }

  @ViewBuilder private func slider(
    title: String, value: Binding<Double>, range: ClosedRange<Int>, caption: String
  ) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      HStack {
        Text(title).font(.subheadline)
        Spacer(minLength: 6)
        Text(caption)
          .font(.system(size: 11).monospacedDigit())
          .foregroundStyle(.tertiary)
      }
      Slider(
        value: value, in: Double(range.lowerBound)...Double(range.upperBound),
        step: 1)
    }
  }

  private var memoryRange: ClosedRange<Int> {
    let range = SandboxLimits.memoryRange(residentBytes: chat.residentBytes)
    let gigabyte = 1024 * 1024 * 1024
    return
      (range.lowerBound / gigabyte)...max(
        range.lowerBound / gigabyte + 1, range.upperBound / gigabyte)
  }

  private func commitImage() {
    var next = choice
    next.image = typedImage.trimmingCharacters(in: .whitespaces)
    guard !next.image.isEmpty else { return }
    chat.sandboxes.settings.remember(image: next.image)
    chat.sandboxChoice = next
  }

  private var kind: Binding<SandboxChoice.Kind> {
    Binding(
      get: { choice.kind },
      set: {
        var next = choice
        next.kind = $0
        chat.sandboxChoice = next
      })
  }

  private var architecture: Binding<SandboxChoice.Architecture> {
    Binding(
      get: { choice.architecture },
      set: {
        var next = choice
        next.architecture = $0
        chat.sandboxChoice = next
      })
  }

  private var cpus: Binding<Double> {
    Binding(
      get: { Double(choice.cpus) },
      set: {
        var next = choice
        next.cpus = Int($0)
        chat.sandboxChoice = next
      })
  }

  private var memoryGigabytes: Binding<Double> {
    Binding(
      get: { Double(choice.memoryBytes / (1024 * 1024 * 1024)) },
      set: {
        var next = choice
        next.memoryBytes = Int($0) * 1024 * 1024 * 1024
        chat.sandboxChoice = next
      })
  }

  private var context: Binding<String> {
    Binding(
      get: { choice.context ?? "" },
      set: {
        var next = choice
        next.context = $0.isEmpty ? nil : $0
        chat.sandboxChoice = next
      })
  }

  private var namespace: Binding<String> {
    Binding(
      get: { choice.namespace ?? "" },
      set: {
        var next = choice
        next.namespace = $0.isEmpty ? nil : $0
        chat.sandboxChoice = next
      })
  }
}

/// What the sandbox is doing, on the line over the composer. A booted VM has taken cores and
/// gigabytes from the machine; it does not get to do that quietly.
@available(macOS 27.0, *)
struct SandboxChip: View {
  @Bindable var chat: ChatController

  var body: some View {
    let phase = chat.sandboxPhase
    let choice = chat.sandboxChoice
    if choice.isSandboxed || phase.isUp || phase.isBusy {
      Button {
        chat.stopSandbox()
      } label: {
        HStack(spacing: 4) {
          if phase.isBusy {
            AnimatedDots(size: 3, tint: Color.reading)
          } else {
            Image(systemName: choice.kind.glyph)
              .font(.system(size: 10))
          }
          Text(label(phase, choice))
            .font(.system(size: 11, weight: .medium))
            .lineLimit(1)
        }
        .foregroundStyle(tint(phase))
        .frame(minHeight: Metrics.hit)
        .contentShape(.rect)
      }
      .buttonStyle(.plain)
      .accessibilityLabel("Sandbox")
      .help(
        phase.isUp
          ? "Running in \(choice.summary). Click to give the machine back."
          : "Commands run in \(choice.summary)")
    }
  }

  private func label(_ phase: SandboxPhase, _ choice: SandboxChoice) -> String {
    switch phase {
    case .off: choice.kind == .native ? "this Mac" : "\(choice.kind.label.lowercased()) · idle"
    case .starting(let what): what
    case .running(let what): what
    case .failed: "sandbox failed"
    }
  }

  private func tint(_ phase: SandboxPhase) -> Color {
    switch phase {
    case .off: .secondary
    case .starting: Color.reading
    case .running: Color.generating
    case .failed: Color.clay
    }
  }
}
