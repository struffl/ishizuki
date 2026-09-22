// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A pod on somebody else's cluster, presented as the same shell. The pod outlives the app, so
// a build started here is still running when the window opens again and asks after it.

import Foundation
import IshizukiKit

actor ClusterSandbox: ExecTransport {
  let pod: String
  let choice: SandboxChoice
  let hostWorkspace: URL
  private let report: @Sendable (SandboxPhase) -> Void
  private var ready = false
  private var seeding: Task<Void, Error>?

  init(
    pod: String, choice: SandboxChoice, hostWorkspace: URL,
    report: @escaping @Sendable (SandboxPhase) -> Void
  ) {
    self.pod = pod
    self.choice = choice
    self.hostWorkspace = hostWorkspace
    self.report = report
  }

  nonisolated var describes: String { "pod \(pod)" }

  nonisolated var isAvailable: Bool { Tooling.locate("kubectl") != nil }

  func prepare() async throws {
    if ready { return }
    if let seeding {
      try await seeding.value
      return
    }
    let work = Task { try await bring() }
    seeding = work
    do {
      try await work.value
      ready = true
    } catch {
      seeding = nil
      report(.failed(error.localizedDescription))
      throw error
    }
  }

  private func bring() async throws {
    guard let kubectl = Tooling.locate("kubectl") else {
      throw SandboxError.missingTool("kubectl")
    }
    report(.starting("asking the cluster for a pod"))

    let existing = try await ProcessRunner.run(
      argv: base(kubectl) + ["get", "pod", pod, "-o", "jsonpath={.status.phase}"],
      stdin: nil, timeout: 60, byteLimit: 4096)

    if existing.stdout.trimmingCharacters(in: .whitespacesAndNewlines) != "Running" {
      let applied = try await ProcessRunner.run(
        argv: base(kubectl) + ["apply", "-f", "-"], stdin: Data(manifest().utf8),
        timeout: 120, byteLimit: 8192)
      guard applied.succeeded else {
        throw SandboxError.failed("creating \(pod)", applied.stderr)
      }
      report(.starting("waiting for the pod to be ready"))
      let waited = try await ProcessRunner.run(
        argv: base(kubectl)
          + ["wait", "--for=condition=Ready", "pod/\(pod)", "--timeout=180s"],
        stdin: nil, timeout: 200, byteLimit: 8192)
      guard waited.succeeded else {
        throw SandboxError.failed("waiting for \(pod)", waited.stderr)
      }
      try await seed(kubectl)
    }
    report(.running(choice.summary))
  }

  /// The folder, copied in once. A cluster has no view of this Mac's disk, so the workspace
  /// travels as a tar and what happens to it afterwards happens there.
  private func seed(_ kubectl: String) async throws {
    report(.starting("copying the folder in"))
    guard let tar = Tooling.locate("tar") else { throw SandboxError.missingTool("tar") }
    let archive = try await ProcessRunner.capture(
      argv: [tar, "-c", "-C", hostWorkspace.path, "."], stdin: nil, timeout: 300)
    guard archive.exitCode == 0 else {
      throw SandboxError.failed("reading the folder", "tar exited \(archive.exitCode)")
    }
    let limit = 512 * 1024 * 1024
    guard archive.data.count <= limit else {
      throw SandboxError.failed(
        "copying the folder in", "it is larger than 512 MB; clone it in the pod instead")
    }
    let sent = try await ProcessRunner.run(
      argv: base(kubectl)
        + [
          "exec", "-i", pod, "--", "sh", "-c",
          "mkdir -p \(SandboxChoice.guestWorkspace.path) && tar -x -C \(SandboxChoice.guestWorkspace.path)",
        ],
      stdin: archive.data, timeout: 600, byteLimit: 8192)
    guard sent.succeeded else {
      throw SandboxError.failed("copying the folder in", sent.stderr)
    }
  }

  func exec(_ script: String, stdin: Data?, timeout: Double, byteLimit: Int) async throws
    -> ShellResult
  {
    try await prepare()
    return try await transport().exec(
      script, stdin: stdin, timeout: timeout, byteLimit: byteLimit)
  }

  func capture(_ script: String, stdin: Data?, timeout: Double) async throws -> (
    data: Data, exitCode: Int32
  ) {
    try await prepare()
    return try await transport().capture(script, stdin: stdin, timeout: timeout)
  }

  /// Deletes the pod, which takes its jobs and its copy of the folder with it.
  func teardown() async {
    guard let kubectl = Tooling.locate("kubectl") else { return }
    _ = try? await ProcessRunner.run(
      argv: base(kubectl) + ["delete", "pod", pod, "--wait=false"], stdin: nil, timeout: 60,
      byteLimit: 4096)
    ready = false
    seeding = nil
    report(.off)
  }

  private func transport() throws -> CommandTransport {
    try CommandTransport.kubectl(
      pod: pod, namespace: choice.namespace, context: choice.context, container: "agent")
  }

  private func base(_ kubectl: String) -> [String] {
    var argv = [kubectl]
    if let context = choice.context, !context.isEmpty { argv += ["--context", context] }
    if let namespace = choice.namespace, !namespace.isEmpty {
      argv += ["--namespace", namespace]
    }
    return argv
  }

  private func manifest() -> String {
    let cpu = max(1, choice.cpus)
    let memory = max(1, choice.memoryBytes / (1024 * 1024 * 1024))
    return """
      {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": { "name": "\(pod)", "labels": { "app": "ishizuki" } },
        "spec": {
          "restartPolicy": "Never",
          "terminationGracePeriodSeconds": 5,
          "containers": [{
            "name": "agent",
            "image": "\(choice.image)",
            "command": ["sh", "-c", "sleep 2147483647"],
            "workingDir": "\(SandboxChoice.guestWorkspace.path)",
            "resources": {
              "requests": { "cpu": "\(cpu)", "memory": "\(memory)Gi" },
              "limits": { "cpu": "\(cpu)", "memory": "\(memory)Gi" }
            },
            "volumeMounts": [{ "name": "workspace", "mountPath": "\(SandboxChoice.guestWorkspace.path)" }]
          }],
          "volumes": [{ "name": "workspace", "emptyDir": {} }]
        }
      }
      """
  }
}
