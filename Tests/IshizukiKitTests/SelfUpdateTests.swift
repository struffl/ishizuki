// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import IshizukiKit

@Suite("Self-update version")
struct SelfUpdateTests {
  @Test("release tags and git describe stamps both parse")
  func parsing() throws {
    #expect(SelfUpdate.Version("v0.2.0")?.description == "v0.2.0")
    #expect(SelfUpdate.Version("0.2.0")?.description == "v0.2.0")
    #expect(SelfUpdate.Version("v1.10.3")?.description == "v1.10.3")
    #expect(SelfUpdate.Version("v0.2")?.description == "v0.2.0")
    #expect(SelfUpdate.Version("v0.2.0-4-gdeadbee")?.description == "v0.2.0")
    #expect(SelfUpdate.Version("v0.2.0-4-gdeadbee-dirty")?.description == "v0.2.0")
  }

  @Test("a build with no release tag has no version, so it is never called up to date")
  func unstamped() {
    #expect(SelfUpdate.Version("dev") == nil)
    #expect(SelfUpdate.Version("") == nil)
    #expect(SelfUpdate.Version("gdeadbee") == nil)
  }

  @Test("ordering is by component, not lexicographic")
  func ordering() throws {
    let patchRelease = try #require(SelfUpdate.Version("v0.1.7"))
    let minorRelease = try #require(SelfUpdate.Version("v0.2.0"))
    let tenthMinor = try #require(SelfUpdate.Version("v0.10.0"))

    #expect(patchRelease < minorRelease)
    #expect(minorRelease < tenthMinor)
    #expect(patchRelease < tenthMinor)
    #expect(minorRelease == SelfUpdate.Version("v0.2.0"))
    #expect(!(minorRelease < minorRelease))
  }
}
