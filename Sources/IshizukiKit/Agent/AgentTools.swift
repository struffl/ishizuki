// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The workspace operations, presented to a session as tools. Nothing but argument shapes and
// the sentences that teach a small model when to reach for each one.

import Foundation
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct ReadFileTool: Tool {
  public let name = "read"
  public let description = """
    Read a slice of a text file. Returns numbered lines and says how many were withheld; \
    call again with offset to see more. Read before you write or edit.
    """

  @Generable
  public struct Arguments {
    @Guide(description: "Path, relative to the workspace")
    public var path: String
    @Guide(description: "First line to return, 1-based. Omit for the start of the file")
    public var offset: Int?
    @Guide(description: "How many lines to return. Omit for the default slice")
    public var limit: Int?
  }

  let workspace: Workspace

  public init(workspace: Workspace) {
    self.workspace = workspace
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.readSlice(
        path: arguments.path, offset: arguments.offset, limit: arguments.limit)
    }
  }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct WriteFileTool: Tool {
  public let name = "write"
  public let description = """
    Write a file whole, creating it if it does not exist. Overwriting a file you have not read \
    in full is refused; prefer edit for a change to part of a file.
    """

  @Generable
  public struct Arguments {
    @Guide(description: "Path, relative to the workspace")
    public var path: String
    @Guide(description: "The file's entire new contents")
    public var contents: String
  }

  let workspace: Workspace

  public init(workspace: Workspace) {
    self.workspace = workspace
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.writeWhole(path: arguments.path, contents: arguments.contents)
    }
  }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct EditFileTool: Tool {
  public let name = "edit"
  public let description = """
    Replace an exact stretch of text in a file. You must have read the lines you are changing. \
    The old text must appear exactly once unless you pass all.
    """

  @Generable
  public struct Arguments {
    @Guide(description: "Path, relative to the workspace")
    public var path: String
    @Guide(description: "The exact text to replace, copied from a read")
    public var old: String
    @Guide(description: "What to put in its place")
    public var new: String
    @Guide(description: "Replace every occurrence rather than requiring exactly one")
    public var all: Bool?
  }

  let workspace: Workspace

  public init(workspace: Workspace) {
    self.workspace = workspace
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.edit(
        path: arguments.path, old: arguments.old, new: arguments.new,
        all: arguments.all ?? false)
    }
  }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct GrepTool: Tool {
  public let name = "grep"
  public let description = """
    Search the workspace for a regular expression. Returns path:line:text, capped. Use this to \
    find where to read rather than reading whole files.
    """

  @Generable
  public struct Arguments {
    @Guide(description: "Regular expression to search for")
    public var pattern: String
    @Guide(description: "Limit to paths matching this glob, e.g. *.swift")
    public var glob: String?
    @Guide(description: "Directory or file to search under. Omit for the whole workspace")
    public var path: String?
    @Guide(description: "Ignore case")
    public var ignoreCase: Bool?
  }

  let workspace: Workspace
  let limit: Int

  public init(workspace: Workspace, limit: Int = 40) {
    self.workspace = workspace
    self.limit = limit
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.grep(
        pattern: arguments.pattern, glob: arguments.glob, path: arguments.path,
        ignoreCase: arguments.ignoreCase ?? false, limit: limit)
    }
  }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct GlobTool: Tool {
  public let name = "glob"
  public let description = "List workspace files matching a glob, capped. Respects .gitignore."

  @Generable
  public struct Arguments {
    @Guide(description: "Glob to match, e.g. Sources/**/*.swift")
    public var pattern: String
  }

  let workspace: Workspace
  let limit: Int

  public init(workspace: Workspace, limit: Int = 60) {
    self.workspace = workspace
    self.limit = limit
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.glob(pattern: arguments.pattern, limit: limit)
    }
  }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct ShellTool: Tool {
  public let name = "shell"
  public let description = """
    Run a command in the workspace and return its output, capped. Use read, write, edit, grep \
    and glob for files; use this for builds, tests and git. A command still running after the \
    wait is left in the background with a job id rather than killed.
    """

  @Generable
  public struct Arguments {
    @Guide(description: "The command line to run")
    public var command: String
    @Guide(description: "Seconds to wait before it goes to the background. Omit for 15")
    public var timeout: Int?
    @Guide(description: "Send it to the background at once, for a server or a long build")
    public var background: Bool?
  }

  let workspace: Workspace
  let byteLimit: Int

  public init(workspace: Workspace, byteLimit: Int = 8 * 1024) {
    self.workspace = workspace
    self.byteLimit = byteLimit
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.shell(
        command: arguments.command, timeout: Double(arguments.timeout ?? 15),
        byteLimit: byteLimit, background: arguments.background ?? false)
    }
  }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct JobsTool: Tool {
  public let name = "jobs"
  public let description = """
    List the commands still running in the background, with how long they have been going and \
    how much output is waiting to be read.
    """

  @Generable
  public struct Arguments {}

  let workspace: Workspace

  public init(workspace: Workspace) {
    self.workspace = workspace
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.jobs()
    }
  }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct JobOutputTool: Tool {
  public let name = "output"
  public let description = """
    Read what a background job has written since you last read it. Pass wait to give it that \
    many seconds to finish first; a job that has finished reports its exit code here.
    """

  @Generable
  public struct Arguments {
    @Guide(description: "The job id, as jobs and shell spell it")
    public var job: String
    @Guide(description: "Seconds to wait for it to finish before answering. Omit for none")
    public var wait: Int?
  }

  let workspace: Workspace
  let byteLimit: Int

  public init(workspace: Workspace, byteLimit: Int = 8 * 1024) {
    self.workspace = workspace
    self.byteLimit = byteLimit
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.jobOutput(
        arguments.job, wait: Double(arguments.wait ?? 0), byteLimit: byteLimit)
    }
  }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct KillJobTool: Tool {
  public let name = "kill"
  public let description = """
    Stop a background job. It is asked to stop first and killed if it stays up; pass force to \
    kill it outright.
    """

  @Generable
  public struct Arguments {
    @Guide(description: "The job id, as jobs and shell spell it")
    public var job: String
    @Guide(description: "Kill it outright rather than asking it to stop")
    public var force: Bool?
  }

  let workspace: Workspace

  public init(workspace: Workspace) {
    self.workspace = workspace
  }

  public func call(arguments: Arguments) async throws -> String {
    try await recovered {
      try await workspace.killJob(arguments.job, force: arguments.force ?? false)
    }
  }
}

/// The set a coding turn is given, in the order the model should reach for them.
@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public func codingTools(for workspace: Workspace) -> [any Tool] {
  [
    ReadFileTool(workspace: workspace),
    GrepTool(workspace: workspace),
    GlobTool(workspace: workspace),
    EditFileTool(workspace: workspace),
    WriteFileTool(workspace: workspace),
    ShellTool(workspace: workspace),
    JobsTool(workspace: workspace),
    JobOutputTool(workspace: workspace),
    KillJobTool(workspace: workspace),
  ]
}

/// A tool that fails hands the failure back as its output rather than throwing it.
///
/// A thrown tool error ends the whole turn: the session unwinds, the answer in flight is lost,
/// and the transcript is left holding a call with nothing under it. Almost none of what these
/// tools refuse is worth that — a missing path or a stale edit is a step the model can take
/// again, and the refusals are already written to be read. Only cancellation still throws,
/// because stopping a turn is the one failure that is meant to end it.
@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
func recovered(_ work: () async throws -> String) async throws -> String {
  do {
    return try await work()
  } catch is CancellationError {
    throw CancellationError()
  } catch let refusal as LedgerRefusal {
    return refusal.message
  } catch {
    if Task.isCancelled { throw CancellationError() }
    return "error: \(error.localizedDescription)"
  }
}
