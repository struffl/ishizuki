// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A command line on the Mac, typed on the phone. A command that runs long keeps running: its
// output arrives as it is written, and it can be stopped from here.

import IshizukiKit
import IshizukiLink
import SwiftUI

struct ShellView: View {
  let store: LinkStore
  let chats: ChatsModel

  @State private var command = ""
  @State private var cwd: String?
  @State private var history: [Entry] = []

  struct Entry: Identifiable {
    let id = UUID()
    var command: String
    var job: ShellJob?
    var stdout = ""
    var stderr = ""
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
                  if !entry.stdout.isEmpty {
                    Text(entry.stdout)
                      .font(.system(size: 11, design: .monospaced))
                      .textSelection(.enabled)
                  }
                  if !entry.stderr.isEmpty {
                    Text(entry.stderr)
                      .font(.system(size: 11, design: .monospaced))
                      .foregroundStyle(Color.instructing)
                      .textSelection(.enabled)
                  }
                  if let failure = entry.failure {
                    Text(failure)
                      .font(.system(size: 11, design: .monospaced))
                      .foregroundStyle(.red)
                  } else if let job = entry.job, !job.isRunning {
                    HStack(spacing: 8) {
                      Text(job.state == .killed ? "stopped" : statusText(job))
                        .foregroundStyle(
                          job.state == .exited && job.exitCode == 0
                            ? Color.generating : .red)
                      Text(String(format: "%.1fs", job.seconds))
                        .foregroundStyle(.tertiary)
                    }
                    .font(.system(size: 10, design: .monospaced))
                  } else {
                    HStack(spacing: 10) {
                      AnimatedDots(size: 3, tint: Color.reading)
                      if let job = entry.job {
                        Text("\(job.id) · \(Int(job.seconds))s")
                          .font(.system(size: 10, design: .monospaced))
                          .foregroundStyle(.tertiary)
                        Button("stop") { stop(job) }
                          .font(.system(size: 10, design: .monospaced))
                          .buttonStyle(.plain)
                          .foregroundStyle(Color.instructing)
                      }
                    }
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
          .onChange(of: history.last?.stdout) {
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
    !command.trimmingCharacters(in: .whitespaces).isEmpty && store.phase.isReady
  }

  private func statusText(_ job: ShellJob) -> String {
    (job.exitCode ?? 0) == 0 ? "ok" : "exit \(job.exitCode ?? 0)"
  }

  /// Starts the command on the Mac and then reads it as it writes, so a build can be watched
  /// rather than waited on, and the next command can be typed over the top of it.
  private func run() {
    guard canRun, let client = store.client else { return }
    let text = command
    command = ""
    let entry = Entry(command: text)
    history.append(entry)

    Task {
      do {
        let started = try await client.start(
          ShellRequest(command: text, cwd: folder.wrappedValue))
        update(entry.id) { $0.job = started }
        while true {
          let seen = try await client.jobOutput(started.id, wait: 5, limit: 32 * 1024)
          update(entry.id) {
            $0.stdout += seen.stdout
            $0.stderr += seen.stderr
            $0.job = seen.job
          }
          if !seen.job.isRunning, seen.remaining == 0 { break }
        }
      } catch {
        update(entry.id) { $0.failure = error.localizedDescription }
      }
    }
  }

  private func stop(_ job: ShellJob) {
    guard let client = store.client else { return }
    Task { _ = try? await client.stopJob(job.id) }
  }

  private func update(_ id: UUID, _ change: (inout Entry) -> Void) {
    guard let index = history.firstIndex(where: { $0.id == id }) else { return }
    change(&history[index])
  }
}
