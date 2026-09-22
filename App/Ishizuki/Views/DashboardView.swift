// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The terminal dashboard as a window: the same fields, in the same order.

import IshizukiKit
import SwiftUI

struct DashboardView: View {
  enum Tab: String { case agent, server, models, tools }

  @Bindable var controller: ServerController
  @State private var runner = JobRunner()
  @State private var quantize = QuantizeController()
  @State private var split = ExpertSplitController()
  @State private var bench = BenchController()
  @State private var tab: Tab
  var chat: ChatController
  var companion: CompanionServer

  init(controller: ServerController, chat: ChatController, companion: CompanionServer) {
    self.controller = controller
    self.chat = chat
    self.companion = companion
    _tab = State(initialValue: controller.catalog.entries.isEmpty ? .models : .agent)
  }

  var body: some View {
    TabView(selection: $tab) {
      ChatView(chat: chat, controller: controller)
        .tabItem { Label("Agent", systemImage: "bubble.left.and.text.bubble.right") }
        .tag(Tab.agent)
      ScrollView { readout.padding(16) }
        .tabItem { Label("Server", systemImage: "gauge.with.dots.needle.33percent") }
        .tag(Tab.server)
      ModelsView(controller: controller)
        .tabItem { Label("Models", systemImage: "shippingbox") }
        .tag(Tab.models)
      ToolsView(
        controller: controller, runner: runner, quantize: quantize, bench: bench,
        split: split
      )
      .tabItem { Label("Tools", systemImage: "wrench.and.screwdriver") }
      .tag(Tab.tools)
    }
    .frame(minWidth: 680, minHeight: 560)
    .scrollContentBackground(.hidden)
    .windowBackdrop()
  }

  @ViewBuilder private var readout: some View {
    VStack(alignment: .leading, spacing: 14) {
      GlassCard { EndpointHeader(controller: controller) }

      if let readout = controller.readout {
        GlassSection(title: "In flight") { InFlightSection(readout: readout) }
        GlassSection(title: "Session") { SessionSection(readout: readout) }
        GlassSection(title: "Load") {
          VStack(alignment: .leading, spacing: 3) {
            LoadSection(readout: readout)
            if let prefix = readout.prefix {
              PrefixSection(prefix: prefix)
            }
            if let experts = readout.experts {
              ExpertSection(experts: experts)
            }
            StateSection(state: readout.state)
          }
        }
      } else {
        GlassCard {
          Text(idleMessage)
            .font(.system(.subheadline, design: .monospaced))
            .foregroundStyle(.secondary)
        }
      }

      if !controller.log.isEmpty {
        GlassSection(title: "Log") { LogSection(lines: controller.log) }
      }
    }
  }

  private var idleMessage: String {
    switch controller.phase {
    case .stopped: "Server stopped."
    case .starting(let name): "Loading \(name)…"
    case .running: "Waiting for the first reading…"
    case .failed(let message): message
    }
  }
}

private struct EndpointHeader: View {
  @Bindable var controller: ServerController

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(spacing: 8) {
        StatusDot(phase: controller.phase)
        Text(controller.readout?.modelName ?? controller.activeEntry?.displayName ?? "no model")
          .font(.system(.body, design: .monospaced, weight: .semibold))
        Spacer()
        Button(controller.phase.isRunning ? "Stop" : "Start") {
          controller.phase.isRunning ? controller.stop() : controller.start()
        }
        .buttonStyle(.glassProminent)
        .disabled(controller.phase.isBusy)
      }

      if controller.phase.isRunning {
        Field(label: "OpenAI") {
          CopyableURL(text: "\(controller.baseURL)/v1")
        }
        Field(label: "Anthropic") {
          CopyableURL(text: controller.baseURL)
        }
      }
    }
  }
}

/// The server's state as a shape as well as a colour, so it still reads for anyone who can't
/// tell the green from the red.
private struct StatusDot: View {
  let phase: ServerController.Phase

  private var mark: (symbol: String, color: Color, label: String) {
    switch phase {
    case .running: ("circle.fill", .green, "Running")
    case .starting: ("circle.dotted", .orange, "Starting")
    case .failed: ("exclamationmark.triangle.fill", .red, "Failed")
    case .stopped: ("circle", .secondary, "Stopped")
    }
  }

  var body: some View {
    Image(systemName: mark.symbol)
      .font(.footnote)
      .foregroundStyle(mark.color)
      .accessibilityLabel(mark.label)
      .help(mark.label)
  }
}

private struct CopyableURL: View {
  let text: String
  @State private var copied = false

  var body: some View {
    HStack(spacing: 6) {
      Text(text).foregroundStyle(.secondary)
      Button {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copied = true
        Task {
          try? await Task.sleep(for: .seconds(1.2))
          copied = false
        }
      } label: {
        Image(systemName: copied ? "checkmark" : "doc.on.doc")
          .hitTarget()
      }
      .buttonStyle(.borderless)
      .accessibilityLabel(copied ? "Copied" : "Copy address")
      .help("Copy this address")
    }
  }
}

private struct InFlightSection: View {
  let readout: ServeReadout

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack(spacing: 6) {
        Text("\(readout.running)").fontWeight(.semibold)
        Text("running").foregroundStyle(.secondary)
        if readout.queued > 0 {
          Text("·").foregroundStyle(.tertiary)
          Text("\(readout.queued)").foregroundStyle(.orange)
          Text("queued").foregroundStyle(.secondary)
        }
      }
      .font(.system(.subheadline, design: .monospaced))

      if readout.inFlight.isEmpty {
        Text("idle — waiting for requests")
          .font(.system(.subheadline, design: .monospaced))
          .foregroundStyle(.tertiary)
          .padding(.leading, 2)
      } else {
        ForEach(readout.inFlight.prefix(10), id: \.id) { RequestRow(request: $0) }
        if readout.inFlight.count > 10 {
          Text("+\(readout.inFlight.count - 10) more")
            .font(.system(.subheadline, design: .monospaced))
            .foregroundStyle(.tertiary)
        }
      }
    }
  }
}

private struct SessionSection: View {
  let readout: ServeReadout

  var body: some View {
    let totals = readout.totals
    VStack(alignment: .leading, spacing: 3) {
      Field(label: "prefill") {
        HStack(spacing: 6) {
          Text(String(format: "%.1f", totals.prefillRate)).fontWeight(.semibold)
          Text("tok/s").foregroundStyle(.secondary)
          Text(String(format: "last %.1f", totals.lastPrefillRate)).foregroundStyle(.tertiary)
        }
      }
      Field(label: "decode") {
        HStack(spacing: 6) {
          Text(String(format: "%.1f", totals.decodeRate)).fontWeight(.semibold)
          Text("tok/s").foregroundStyle(.secondary)
          Text(String(format: "last %.1f", totals.lastDecodeRate)).foregroundStyle(.tertiary)
        }
      }
      Field(label: "requests") {
        HStack(spacing: 6) {
          Text("\(totals.completed)").foregroundStyle(Color.accentColor)
          Text("done").foregroundStyle(.secondary)
          Text("·").foregroundStyle(.tertiary)
          Text("\(totals.failed)").foregroundStyle(totals.failed > 0 ? .red : .secondary)
          Text("failed").foregroundStyle(.secondary)
          if totals.cancelled > 0 {
            Text("·").foregroundStyle(.tertiary)
            Text("\(totals.cancelled)").foregroundStyle(.orange)
            Text("cancelled").foregroundStyle(.secondary)
          }
          Text("· \(totals.arrived) seen").foregroundStyle(.tertiary)
        }
      }
      Field(label: "tokens") {
        HStack(spacing: 6) {
          Text(ReadoutFormat.group(totals.promptTokens)).foregroundStyle(Color.accentColor)
          Text("in").foregroundStyle(.secondary)
          Text("·").foregroundStyle(.tertiary)
          Text(ReadoutFormat.group(totals.generatedTokens)).foregroundStyle(Color.accentColor)
          Text("out").foregroundStyle(.secondary)
        }
      }
      Field(label: "cache") {
        HStack(spacing: 6) {
          Text(ReadoutFormat.percent(totals.cacheRatio)).foregroundStyle(Color.accentColor)
          Text(
            "\(totals.cacheHits) hit · \(totals.cacheMisses) miss · "
              + "\(ReadoutFormat.group(totals.cachedTokens)) tok reused"
          )
          .foregroundStyle(.tertiary)
        }
      }
    }
  }
}

private struct LoadSection: View {
  let readout: ServeReadout

  var body: some View {
    let load = readout.load
    let context = readout.context
    VStack(alignment: .leading, spacing: 3) {
      Field(label: "memory") {
        HStack(spacing: 8) {
          Bar(fraction: load.fraction)
          Text(ReadoutFormat.percent(load.fraction)).fontWeight(.semibold)
          Text("\(ReadoutFormat.gigabytes(load.held)) / \(ReadoutFormat.gigabytes(load.ceiling))")
            .foregroundStyle(.secondary)
          Text(
            "weights \(ReadoutFormat.gigabytes(load.weights)) · "
              + "peak \(ReadoutFormat.gigabytes(load.peak))"
          )
          .foregroundStyle(.tertiary)
        }
      }
      if let gpu = load.gpu {
        Field(label: "gpu") {
          HStack(spacing: 8) {
            Bar(fraction: gpu)
            Text(ReadoutFormat.percent(gpu)).fontWeight(.semibold)
            Text("busy").foregroundStyle(.secondary)
          }
        }
      }
      Field(label: "budget") {
        HStack(spacing: 6) {
          Text(readout.budgetSummary).foregroundStyle(Color.accentColor)
          Text("· \(ReadoutFormat.gigabytes(readout.headroom)) spare").foregroundStyle(.tertiary)
        }
      }
      Field(label: "context") {
        HStack(spacing: 6) {
          Text(MemoryBudget.tokens(context.peakTokens) + " peak")
            .foregroundStyle(.secondary)
          Text(
            "· \(MemoryBudget.tokens(context.reservedTokens)) reserved"
              + " · \(MemoryBudget.tokens(context.ceilingTokens)) ceiling"
              + " · \(ReadoutFormat.compact(context.kvHeldBytes)) kv held"
          )
          .foregroundStyle(.tertiary)
        }
      }
    }
  }
}

private struct PrefixSection: View {
  let prefix: ServeReadout.Prefix

  private var occupancy: String {
    var parts: [String] = []
    parts.append(
      prefix.ramLimit > 0
        ? "\(ReadoutFormat.compact(prefix.ramBytes)) / \(ReadoutFormat.compact(prefix.ramLimit)) ram"
        : "\(ReadoutFormat.compact(prefix.ramBytes)) ram")
    if let bytes = prefix.diskBytes, let limit = prefix.diskLimit {
      parts.append("\(ReadoutFormat.compact(bytes)) / \(ReadoutFormat.compact(limit)) disk")
    }
    return parts.joined(separator: " · ")
  }

  private var detail: String {
    var parts: [String] = []
    if prefix.lookups > 0 { parts.append("\(prefix.hits) of \(prefix.lookups) reused") }
    if prefix.branches > 0 { parts.append("\(prefix.branches) branched") }
    if prefix.diskHits > 0 { parts.append("\(prefix.diskHits) from disk") }
    if prefix.evictions > 0 { parts.append("\(prefix.evictions) evicted") }
    parts.append("\(prefix.slots) slot\(prefix.slots == 1 ? "" : "s")")
    return parts.joined(separator: " · ")
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 3) {
      Field(label: "prefix") {
        HStack(spacing: 6) {
          if prefix.lookups > 0 {
            Text(ReadoutFormat.percent(prefix.hitRate) + " hit")
              .foregroundStyle(.secondary)
            Text("·").foregroundStyle(.tertiary)
          }
          Text(occupancy).foregroundStyle(.secondary)
        }
      }
      Field(label: "") {
        Text(detail).foregroundStyle(.tertiary)
      }
    }
  }
}

/// The one dial a streamed pack has, and whether it is buying anything. A slot budget at or
/// below what a token routes to cannot hit, however large the bank is.
private struct ExpertSection: View {
  let experts: ExpertStore.Summary

  private var detail: String {
    var parts = [
      "\(experts.slots) of \(experts.expertCount) held",
      "\(experts.layers) layer\(experts.layers == 1 ? "" : "s")",
      "\(ReadoutFormat.compact(experts.heldBytes)) resident",
    ]
    if experts.reads > 0 {
      parts.append("\(experts.misses) read from disk")
    }
    return parts.joined(separator: " · ")
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 3) {
      Field(label: "experts") {
        HStack(spacing: 6) {
          if experts.reads > 0 {
            Text(ReadoutFormat.percent(experts.hitRate) + " hit")
              .foregroundStyle(.secondary)
            Text("·").foregroundStyle(.tertiary)
          }
          Text("streamed from disk").foregroundStyle(.secondary)
        }
      }
      Field(label: "") {
        Text(detail).foregroundStyle(.tertiary)
      }
    }
  }
}

private struct StateSection: View {
  let state: ServeReadout.State

  private var conditions: String {
    var parts = [state.politeness.rawValue, "thermal \(state.thermal)"]
    if state.lowPower { parts.append("low power") }
    if state.idleSeconds > 0 { parts.append("pool freed at \(Int(state.idleSeconds))s idle") }
    if state.evictSeconds > 0 { parts.append("unload at \(Int(state.evictSeconds))s idle") }
    parts.append("up \(ReadoutFormat.duration(state.uptime))")
    return parts.joined(separator: " · ")
  }

  var body: some View {
    Field(label: "state") {
      Text(conditions).foregroundStyle(.secondary)
    }
  }
}

private struct LogSection: View {
  let lines: [String]

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      ForEach(Array(lines.suffix(8).enumerated()), id: \.offset) { _, line in
        Text(line)
          .font(.system(.footnote, design: .monospaced))
          .foregroundStyle(.tertiary)
          .lineLimit(1)
      }
    }
  }
}
