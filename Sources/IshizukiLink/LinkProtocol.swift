// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The wire between a Mac running the pack and a phone borrowing it. Every type here is Codable
// and carries no platform in it, so the same declarations serve both ends of the link.

import Foundation
import IshizukiKit

public enum Link {
  /// Bumped when a field stops meaning what it did. A phone that disagrees says so rather than
  /// guessing at a transcript.
  public static let protocolVersion = 1
  public static let defaultPort: UInt16 = 8129
}

public struct ServerInfo: Codable, Sendable, Equatable {
  public var name: String
  public var version: String
  public var protocolVersion: Int
  public var model: String?
  public var modelLoaded: Bool
  public var chats: Int
  public var pairingOpen: Bool

  public init(
    name: String, version: String, protocolVersion: Int = Link.protocolVersion,
    model: String?, modelLoaded: Bool, chats: Int, pairingOpen: Bool
  ) {
    self.name = name
    self.version = version
    self.protocolVersion = protocolVersion
    self.model = model
    self.modelLoaded = modelLoaded
    self.chats = chats
    self.pairingOpen = pairingOpen
  }
}

public struct DeviceDescriptor: Codable, Sendable, Equatable, Identifiable {
  public var id: String
  public var name: String
  public var system: String
  public var pairedAt: Date
  public var lastSeen: Date?

  public init(
    id: String, name: String, system: String, pairedAt: Date = Date(), lastSeen: Date? = nil
  ) {
    self.id = id
    self.name = name
    self.system = system
    self.pairedAt = pairedAt
    self.lastSeen = lastSeen
  }
}

/// What a device gets for completing a handshake while pairing is open. The token is what
/// distinguishes one paired phone from another, and revoking it is how one is forgotten.
public struct PairGrant: Codable, Sendable, Equatable {
  public var token: String
  public var serverName: String
  public var info: ServerInfo

  public init(token: String, serverName: String, info: ServerInfo) {
    self.token = token
    self.serverName = serverName
    self.info = info
  }
}

public struct ChatSummary: Codable, Sendable, Hashable, Identifiable {
  public var id: UUID
  public var title: String
  public var created: Date
  public var updated: Date
  public var workspace: String?
  public var model: String?
  public var effort: String
  public var rows: Int
  public var isRunning: Bool
  public var preview: String?

  public init(
    id: UUID, title: String, created: Date, updated: Date, workspace: String?,
    model: String?, effort: String, rows: Int, isRunning: Bool, preview: String?
  ) {
    self.id = id
    self.title = title
    self.created = created
    self.updated = updated
    self.workspace = workspace
    self.model = model
    self.effort = effort
    self.rows = rows
    self.isRunning = isRunning
    self.preview = preview
  }
}

/// One row of a transcript as the phone draws it. The kinds are the desktop window's own, so a
/// conversation reads the same on both screens.
public struct TranscriptRow: Codable, Sendable, Equatable, Identifiable {
  public enum Kind: String, Codable, Sendable {
    case system, prompt, steer, reasoning, answer, toolCall, toolOutput
    /// What became of a turn rather than what was in it: stopped, or failed.
    case notice
  }

  public var id: String
  public var kind: Kind
  public var tool: String?
  public var text: String
  public var at: Date?
  public var seconds: Double?
  public var tokens: Int?
  public var wasRead: Bool

  public init(
    id: String, kind: Kind, tool: String? = nil, text: String, at: Date? = nil,
    seconds: Double? = nil, tokens: Int? = nil, wasRead: Bool = false
  ) {
    self.id = id
    self.kind = kind
    self.tool = tool
    self.text = text
    self.at = at
    self.seconds = seconds
    self.tokens = tokens
    self.wasRead = wasRead
  }
}

public struct ChatDetail: Codable, Sendable, Equatable {
  public var summary: ChatSummary
  public var rows: [TranscriptRow]
  public var pendingSteers: [TranscriptRow]
  public var failure: String?

  public init(
    summary: ChatSummary, rows: [TranscriptRow], pendingSteers: [TranscriptRow] = [],
    failure: String? = nil
  ) {
    self.summary = summary
    self.rows = rows
    self.pendingSteers = pendingSteers
    self.failure = failure
  }
}

/// What the turn is doing, flattened out of the window's own activity so it can cross a wire.
public struct LinkActivity: Codable, Sendable, Equatable {
  public enum Phase: String, Codable, Sendable {
    case idle, queued, reading, writing, command, unknown
  }

  public var phase: Phase
  public var fraction: Double?
  public var command: String?
  public var isRunning: Bool

  public init(phase: Phase, fraction: Double? = nil, command: String? = nil, isRunning: Bool) {
    self.phase = phase
    self.fraction = fraction
    self.command = command
    self.isRunning = isRunning
  }

  public static let idle = LinkActivity(phase: .idle, isRunning: false)
}

/// The dials, condensed. Enough for a phone to show what the Mac is spending without carrying
/// the whole readout across on every tick.
public struct LinkStatus: Codable, Sendable, Equatable {
  public var model: String?
  public var loaded: Bool
  public var contextTokens: Int
  public var contextCeiling: Int
  public var generatedTokens: Int
  public var tokensPerSecond: Double
  public var cachedTokens: Int
  public var heldBytes: Int
  public var ceilingBytes: Int
  public var gpu: Double?
  public var thermal: String?

  public init(
    model: String?, loaded: Bool, contextTokens: Int, contextCeiling: Int,
    generatedTokens: Int, tokensPerSecond: Double, cachedTokens: Int, heldBytes: Int,
    ceilingBytes: Int, gpu: Double?, thermal: String?
  ) {
    self.model = model
    self.loaded = loaded
    self.contextTokens = contextTokens
    self.contextCeiling = contextCeiling
    self.generatedTokens = generatedTokens
    self.tokensPerSecond = tokensPerSecond
    self.cachedTokens = cachedTokens
    self.heldBytes = heldBytes
    self.ceilingBytes = ceilingBytes
    self.gpu = gpu
    self.thermal = thermal
  }
}

/// One frame of a conversation's live stream. Rows arrive whole rather than as deltas, which is
/// what the transcript already hands over on the Mac and what a text view wants anyway.
public struct TurnEvent: Codable, Sendable, Equatable {
  public enum Kind: String, Codable, Sendable {
    case snapshot, rows, activity, status, failure, done, ping
  }

  public var kind: Kind
  public var rows: [TranscriptRow]?
  public var pendingSteers: [TranscriptRow]?
  public var activity: LinkActivity?
  public var status: LinkStatus?
  public var summary: ChatSummary?
  public var failure: String?

  public init(
    kind: Kind, rows: [TranscriptRow]? = nil, pendingSteers: [TranscriptRow]? = nil,
    activity: LinkActivity? = nil, status: LinkStatus? = nil, summary: ChatSummary? = nil,
    failure: String? = nil
  ) {
    self.kind = kind
    self.rows = rows
    self.pendingSteers = pendingSteers
    self.activity = activity
    self.status = status
    self.summary = summary
    self.failure = failure
  }
}

public struct NewChat: Codable, Sendable, Equatable {
  public var workspace: String?
  public var model: String?
  public var effort: String?

  public init(
    workspace: String? = nil, model: String? = nil, effort: String? = nil
  ) {
    self.workspace = workspace
    self.model = model
    self.effort = effort
  }
}

public struct ChatChange: Codable, Sendable, Equatable {
  public var title: String?
  public var workspace: String?
  public var model: String?
  public var effort: String?

  public init(
    title: String? = nil, workspace: String? = nil, model: String? = nil, effort: String? = nil
  ) {
    self.title = title
    self.workspace = workspace
    self.model = model
    self.effort = effort
  }
}

public struct SendText: Codable, Sendable, Equatable {
  public var text: String

  public init(text: String) {
    self.text = text
  }
}

public struct ModelEntry: Codable, Sendable, Equatable, Identifiable {
  public var id: String
  public var name: String
  public var quantization: String?
  public var contextTokens: Int
  public var sizeBytes: Int
  public var loaded: Bool

  public init(
    id: String, name: String, quantization: String?, contextTokens: Int, sizeBytes: Int,
    loaded: Bool
  ) {
    self.id = id
    self.name = name
    self.quantization = quantization
    self.contextTokens = contextTokens
    self.sizeBytes = sizeBytes
    self.loaded = loaded
  }
}

public struct ModelList: Codable, Sendable, Equatable {
  public var active: String?
  public var entries: [ModelEntry]

  public init(active: String?, entries: [ModelEntry]) {
    self.active = active
    self.entries = entries
  }
}

/// A folder the Mac is willing to open a conversation in. The phone never names a path the Mac
/// has not offered, so browsing a machine remotely cannot wander out of what was allowed.
public struct WorkspaceRoot: Codable, Sendable, Equatable, Identifiable {
  public var id: String { path }
  public var path: String
  public var name: String
  public var isRecent: Bool

  public init(path: String, name: String, isRecent: Bool) {
    self.path = path
    self.name = name
    self.isRecent = isRecent
  }
}

public struct RootList: Codable, Sendable, Equatable {
  public var roots: [WorkspaceRoot]

  public init(roots: [WorkspaceRoot]) {
    self.roots = roots
  }
}

public struct DirectoryEntry: Codable, Sendable, Equatable, Identifiable {
  public var id: String { path }
  public var name: String
  public var path: String
  public var isDirectory: Bool
  public var size: Int
  public var modified: Date?

  public init(name: String, path: String, isDirectory: Bool, size: Int, modified: Date?) {
    self.name = name
    self.path = path
    self.isDirectory = isDirectory
    self.size = size
    self.modified = modified
  }
}

public struct DirectoryListing: Codable, Sendable, Equatable {
  public var path: String
  public var parent: String?
  public var entries: [DirectoryEntry]

  public init(path: String, parent: String?, entries: [DirectoryEntry]) {
    self.path = path
    self.parent = parent
    self.entries = entries
  }
}

public struct FileSlice: Codable, Sendable, Equatable {
  public var path: String
  public var start: Int
  public var end: Int
  public var total: Int
  public var lines: [String]
  public var isBinary: Bool

  public init(path: String, start: Int, end: Int, total: Int, lines: [String], isBinary: Bool) {
    self.path = path
    self.start = start
    self.end = end
    self.total = total
    self.lines = lines
    self.isBinary = isBinary
  }
}

public struct FileWrite: Codable, Sendable, Equatable {
  public var path: String
  public var contents: String

  public init(path: String, contents: String) {
    self.path = path
    self.contents = contents
  }
}

public struct ShellRequest: Codable, Sendable, Equatable {
  public var command: String
  public var cwd: String?
  public var timeout: Double?

  public init(command: String, cwd: String? = nil, timeout: Double? = nil) {
    self.command = command
    self.cwd = cwd
    self.timeout = timeout
  }
}

/// A job list on the wire. An envelope rather than a bare array, so the Mac can say how much
/// of its shell it is sharing alongside it later without breaking a phone.
public struct JobList: Codable, Sendable, Equatable {
  public var jobs: [ShellJob]

  public init(jobs: [ShellJob]) {
    self.jobs = jobs
  }
}

public struct ShellOutcome: Codable, Sendable, Equatable {
  public var stdout: String
  public var stderr: String
  public var exitCode: Int32
  public var truncated: Bool
  public var seconds: Double

  public init(stdout: String, stderr: String, exitCode: Int32, truncated: Bool, seconds: Double) {
    self.stdout = stdout
    self.stderr = stderr
    self.exitCode = exitCode
    self.truncated = truncated
    self.seconds = seconds
  }
}

public struct LinkFailure: Codable, Sendable, Equatable, Error {
  public var code: String
  public var message: String

  public init(code: String, message: String) {
    self.code = code
    self.message = message
  }
}

extension JSONEncoder {
  public static var link: JSONEncoder {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = [.withoutEscapingSlashes]
    return encoder
  }
}

extension JSONDecoder {
  public static var link: JSONDecoder {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return decoder
  }
}
