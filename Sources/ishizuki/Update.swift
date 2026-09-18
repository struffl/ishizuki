// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit

struct Update: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "update",
    abstract: "Update ishizuki in place from its GitHub releases.",
    discussion: """
      Fetches the release tarball for this machine's architecture and swaps the running
      executable and the metallib beside it, wherever ishizuki was installed from — the
      .pkg, the tarball, or `just install`.

      Nothing is installed until the download's sha256 matches the digest GitHub
      publishes for the asset and the new executable verifies against the same
      Developer ID team as the one it replaces. Models, configuration files, and the
      launchd agent are left alone, and a server that is already running keeps the code
      it started with until you restart it.

        ishizuki update                # install the latest release, if it is newer
        ishizuki update --check        # report what is available, change nothing
        ishizuki update --tag v0.1.7   # install a specific release
        ishizuki update --force        # reinstall the current version

      serve and launch mention a newer release in their header, from a check refreshed
      in the background once a day. Set \(SelfUpdate.optOutVariable)=1 to turn that off.
      """)

  @Flag(name: .long, help: "Report the available release and exit.")
  var check = false

  @Option(name: .long, help: "Release tag to install instead of the latest.")
  var tag: String?

  @Option(name: .long, help: "GitHub repository to update from.")
  var repo: String = SelfUpdate.defaultRepo

  @Flag(name: .long, help: "Install even when the running version is the same or newer.")
  var force = false

  @Flag(name: .long, help: "Disable coloured output.")
  var noColor = false

  func run() throws {
    if noColor { Style.disable() }

    let executable = SelfUpdate.runningExecutable()
    let running = SelfUpdate.Version(BuildInfo.version)
    let release = try SelfUpdate.release(repo: repo, tag: tag)

    print(Style.banner("update"))
    print("")
    print(
      "  "
        + Style.field(
          "installed",
          Style.accent(BuildInfo.version)
            + (running == nil ? Style.faint("  a build from source") : "")))
    print(
      "  "
        + Style.field(
          "available",
          Style.accent(release.tag)
            + Style.faint(
              release.releaseDate.isEmpty ? "" : "  released \(release.releaseDate)")))
    print("  " + Style.field("binary", Style.faint(shortPath(executable.path))))
    print("")
    fflush(stdout)

    guard let latest = release.version else {
      throw BonsaiError.missingComponent("cannot read a version out of the tag '\(release.tag)'")
    }

    if let running, latest <= running, !force {
      print(
        Style.good(
          latest == running
            ? "Already on \(release.tag)."
            : "Running \(BuildInfo.version), newer than \(release.tag)."))
      if !check { print(Style.faint("Pass --force to install it anyway.")) }
      return
    }

    if check {
      print(Style.warn("\(release.tag) is available.") + Style.faint("  Run: ishizuki update"))
      return
    }

    try SelfUpdate.install(release, replacing: executable) { line in
      FileHandle.standardError.write(Data(("  " + Style.faint(line) + "\n").utf8))
    }

    print("")
    print(Style.good("Updated to \(release.tag).") + Style.faint("  Restart any running server."))
  }

  private func shortPath(_ path: String) -> String {
    path.hasPrefix(NSHomeDirectory()) ? "~" + path.dropFirst(NSHomeDirectory().count) : path
  }
}
