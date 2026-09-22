// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The sandboxes this app has going, one per conversation that asked for one. Holds them so a
// second turn in the same chat reaches the VM the first turn booted, and so the line over the
// composer has something to report.

import Foundation
import IshizukiKit
import Observation

@available(macOS 27.0, *)
@MainActor
@Observable
final class Sandboxes {
  private(set) var phases: [UUID: SandboxPhase] = [:]
  private(set) var choices: [UUID: SandboxChoice] = [:]

  private var containers: [UUID: ContainerSandbox] = [:]
  private var clusters: [UUID: ClusterSandbox] = [:]
  let settings = SandboxSettings.shared

  func phase(of chat: UUID) -> SandboxPhase { phases[chat] ?? .off }

  func choice(of chat: UUID) -> SandboxChoice { choices[chat] ?? .native }

  /// The shell a conversation's tools should run through. Booting is not done here: it happens
  /// on the first command, so choosing a container costs nothing until something is run.
  func host(for chat: UUID, choice: SandboxChoice, workspace folder: URL) -> any ShellHost {
    choices[chat] = choice

    switch choice.kind {
    case .native:
      phases[chat] = .off
      return LocalShellHost(workspace: folder)

    case .container:
      let sandbox =
        containers[chat]
        ?? ContainerSandbox(
          id: "ishizuki-\(chat.uuidString.prefix(8).lowercased())",
          choice: choice,
          hostWorkspace: folder,
          artifacts: settings.artifacts,
          report: { [weak self] phase in
            Task { @MainActor in self?.phases[chat] = phase }
          })
      containers[chat] = sandbox
      phases[chat] = phases[chat] ?? .off
      return SpooledShellHost(
        workspace: SandboxChoice.guestWorkspace, transport: sandbox)

    case .cluster:
      let sandbox =
        clusters[chat]
        ?? ClusterSandbox(
          pod: "ishizuki-\(chat.uuidString.prefix(8).lowercased())",
          choice: choice,
          hostWorkspace: folder,
          report: { [weak self] phase in
            Task { @MainActor in self?.phases[chat] = phase }
          })
      clusters[chat] = sandbox
      phases[chat] = phases[chat] ?? .off
      return SpooledShellHost(
        workspace: SandboxChoice.guestWorkspace, transport: sandbox)
    }
  }

  /// Gives back the VM or the pod. A conversation being closed is not reason enough; being
  /// deleted, or the app quitting, is.
  func shutdown(_ chat: UUID) {
    let container = containers.removeValue(forKey: chat)
    let cluster = clusters.removeValue(forKey: chat)
    phases[chat] = .off
    Task {
      await container?.teardown()
      await cluster?.teardown()
    }
  }

  func shutdownAll() {
    for chat in Array(containers.keys) + Array(clusters.keys) { shutdown(chat) }
  }
}
