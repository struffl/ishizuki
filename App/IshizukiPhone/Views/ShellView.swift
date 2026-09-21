// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A command line on the Mac, typed on the phone. One command at a time, with its output kept
// above the next one.

import IshizukiKit
import IshizukiLink
import SwiftUI

struct ShellView: View {
  let store: LinkStore
  let chats: ChatsModel

  @State private var command = ""
  @State private var cwd: String?
  @State private var history: [Entry] = []
  @State private var running = false

  struct Entry: Identifiable {
    let id = UUID()
    var command: String
    var outcome: ShellOutcome?
    var failure: String?
  }

  var body: some View {
    NavigationStack {
      VStack(spacing: 0) {
        ScrollViewReader { scroller in
          ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
              ForEach(history) { entry in
                VStack(alignment: .leading, spacing: 4) {
                  Text("$ \(entry.command)")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.reading)
                  if let outcome = entry.outcome {
                    if !outcome.stdout.isEmpty {
                      Text(outcome.stdout)
                        .font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                    }
                    if !outcome.stderr.isEmpty {
                      Text(outcome.stderr)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Color.instructing)
                        .textSelection(.enabled)
                    }
                    HStack(spacing: 8) {
                      Text(outcome.exitCode == 0 ? "ok" : "exit \(outcome.exitCode)")
                        .foregroundStyle(outcome.exitCode == 0 ? Color.generating : .red)
                      Text(String(format: "%.1fs", outcome.seconds))
                        .foregroundStyle(.tertiary)
                      if outcome.truncated {
                        Text("output capped").foregroundStyle(.tertiary)
                      }
                    }
                    .font(.system(size: 10, design: .monospaced))
                  } else if let failure = entry.failure {
                    Text(failure)
                      .font(.system(size: 11, design: .monospaced))
                      .foregroundStyle(.red)
                  } else {
                    AnimatedDots(size: 3, tint: Color.reading)
                  }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .id(entry.id)
              }
            }
            .padding(14)
          }
          .onChange(of: history.count) {
            guard let last = history.last?.id else { return }
            withAnimation { scroller.scrollTo(last, anchor: .bottom) }
          }
          .onChange(of: history.last?.outcome?.stdout) {
            guard let last = history.last?.id else { return }
            withAnimation { scroller.scrollTo(last, anchor: .bottom) }
          }
        }

        VStack(spacing: 8) {
          Picker("Folder", selection: folder) {
            ForEach(chats.roots) { root in
              Text(root.name).tag(Optional(root.path))
            }
          }
          .pickerStyle(.menu)
          .frame(maxWidth: .infinity, alignment: .leading)

          HStack(spacing: 8) {
            TextField("command", text: $command)
              .font(.system(size: 13, design: .monospaced))
              .textInputAutocapitalization(.never)
              .autocorrectionDisabled()
              .padding(.horizontal, 12)
              .padding(.vertical, 8)
              .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 18))
              .onSubmit { run() }
            Button {
              run()
            } label: {
              Image(systemName: "return")
                .frame(width: 34, height: 34)
                .background(Color.reading.opacity(canRun ? 1 : 0.3), in: .circle)
                .foregroundStyle(.white)
            }
            .disabled(!canRun)
          }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(.bar)
      }
      .navigationTitle("Shell")
      .toolbar {
        ToolbarItem(placement: .topBarTrailing) {
          Button("Clear") { history.removeAll() }
            .disabled(history.isEmpty)
        }
      }
      .task { if chats.roots.isEmpty { await chats.refresh() } }
    }
  }

  private var folder: Binding<String?> {
    Binding(
      get: { cwd ?? chats.roots.first?.path },
      set: { cwd = $0 })
  }

  private var canRun: Bool {
    !running && !command.trimmingCharacters(in: .whitespaces).isEmpty && store.phase.isReady
  }

  private func run() {
    guard canRun, let client = store.client else { return }
    let text = command
    command = ""
    running = true
    let entry = Entry(command: text)
    history.append(entry)

    Task {
      do {
        let outcome = try await client.shell(
          ShellRequest(command: text, cwd: folder.wrappedValue, timeout: 300))
        update(entry.id) { $0.outcome = outcome }
      } catch {
        update(entry.id) { $0.failure = error.localizedDescription }
      }
      running = false
    }
  }

  private func update(_ id: UUID, _ change: (inout Entry) -> Void) {
    guard let index = history.firstIndex(where: { $0.id == id }) else { return }
    change(&history[index])
  }
}
