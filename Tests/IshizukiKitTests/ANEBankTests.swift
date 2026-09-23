// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The Neural Engine bank stops at its byte budget, and turning the offload off lets go of it.

import Foundation
import Testing

@testable import IshizukiKit

@Suite("ANE bank", .serialized)
struct ANEBankTests {
  private func pack(sliceBytes: Int) throws -> URL {
    let root = FileManager.default.temporaryDirectory
      .appending(path: "ane-bank-\(UUID().uuidString)")
    for name in ["0.mlp.gate_proj", "0.mlp.up_proj"] {
      let slice = root.appending(path: "ane-2048/\(name).mlmodelc")
      try FileManager.default.createDirectory(at: slice, withIntermediateDirectories: true)
      try Data(count: sliceBytes).write(to: slice.appending(path: "weight.bin"))
    }
    return root
  }

  @Test func refusesPastBudget() throws {
    let root = try pack(sliceBytes: 1024)
    defer { try? FileManager.default.removeItem(at: root) }
    let bank = try #require(ANEBank(pack: root, budgetBytes: 1500))

    #expect(bank.slices(["0.mlp.gate_proj", "0.mlp.up_proj"]) == nil)
    #expect(bank.residentBytes == 0)
  }

  @Test func missingSliceSpendsNothing() throws {
    let root = try pack(sliceBytes: 1024)
    defer { try? FileManager.default.removeItem(at: root) }
    let bank = try #require(ANEBank(pack: root, budgetBytes: 1 << 20))

    #expect(bank.slices(["0.mlp.gate_proj", "0.mlp.down_proj"]) == nil)
    #expect(bank.residentBytes == 0)
  }

  @Test func offReleasesTheBank() throws {
    let root = try pack(sliceBytes: 16)
    defer { try? FileManager.default.removeItem(at: root) }

    try ANEOffload.apply(.automatic, pack: root)
    #expect(BonsaiRuntime.aneBank != nil)
    try ANEOffload.apply(nil, pack: root)
    #expect(BonsaiRuntime.aneBank == nil)
    #expect(BonsaiRuntime.aneOffload == nil)
  }
}
