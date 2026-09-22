// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Where a conversation's commands run. Carried with the conversation rather than set once for
// the app, because the answer is different for a repository you trust and a patch you do not.

import Foundation

public struct SandboxChoice: Codable, Sendable, Equatable, Hashable {
  public enum Kind: String, Codable, Sendable, CaseIterable {
    /// This Mac, this user, no boundary. What a coding session has always been.
    case native
    /// A Linux VM of this Mac's own, with the folder shared into it.
    case container
    /// A pod on a cluster, with the folder copied in.
    case cluster

    public var label: String {
      switch self {
      case .native: "This Mac"
      case .container: "Local container"
      case .cluster: "Cluster"
      }
    }

    public var glyph: String {
      switch self {
      case .native: "laptopcomputer"
      case .container: "cube"
      case .cluster: "cloud"
      }
    }
  }

  public enum Architecture: String, Codable, Sendable, CaseIterable {
    case arm64
    case amd64

    public var label: String {
      switch self {
      case .arm64: "arm64"
      case .amd64: "amd64 (Rosetta)"
      }
    }
  }

  public var kind: Kind
  public var image: String
  public var cpus: Int
  public var memoryBytes: Int
  public var architecture: Architecture
  /// Where a cluster sandbox goes. Ignored by the other two.
  public var context: String?
  public var namespace: String?

  public init(
    kind: Kind = .native, image: String = "docker.io/library/debian:bookworm-slim",
    cpus: Int = 4, memoryBytes: Int = 8 * 1024 * 1024 * 1024,
    architecture: Architecture = .arm64, context: String? = nil, namespace: String? = nil
  ) {
    self.kind = kind
    self.image = image
    self.cpus = cpus
    self.memoryBytes = memoryBytes
    self.architecture = architecture
    self.context = context
    self.namespace = namespace
  }

  public static let native = SandboxChoice()

  /// The workspace's path inside a sandbox. The same everywhere, so a transcript does not
  /// carry one machine's home directory into another machine's shell.
  public static let guestWorkspace = URL(filePath: "/workspace")

  public var isSandboxed: Bool { kind != .native }

  /// What the status line says when this is what a turn is running in.
  public var summary: String {
    switch kind {
    case .native: "this Mac"
    case .container:
      "\(image) · \(cpus) CPU · \(memoryBytes / (1024 * 1024 * 1024)) GB"
        + (architecture == .amd64 ? " · amd64" : "")
    case .cluster:
      [context, namespace, image].compactMap { $0 }.filter { !$0.isEmpty }
        .joined(separator: " · ")
    }
  }
}
