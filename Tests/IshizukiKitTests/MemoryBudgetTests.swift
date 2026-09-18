// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

private let gigabyte = 1_073_741_824
private let megabyte = 1_048_576

private func budget(
  ceiling: Int = 54 * 1_073_741_824,
  weights: Int = 8 * 1_073_741_824,
  maxContextTokens: Int = 262_144,
  slots: Int? = nil,
  bufferCache: Int? = nil
) -> MemoryBudget {
  MemoryBudget(
    kvBits: 3.5, maxContextTokens: maxContextTokens, weights: weights, ceiling: ceiling,
    slots: slots, bufferCache: bufferCache)
}

@Suite("Memory budget ladder")
struct MemoryBudgetTests {
  @Test("a cold server commits to one short conversation, whatever the machine")
  func coldStartIsSmall() {
    for ceiling in [12, 54, 400].map({ $0 * gigabyte }) {
      let tier = budget(ceiling: ceiling).tier
      #expect(tier.slots == 1)
      #expect(tier.contextTokens == MemoryBudget.contextFloor)
      #expect(tier.bufferCache == MemoryBudget.bufferFloor)
    }
  }

  @Test("the context reserve doubles until it covers the prompt")
  func contextDoublesToDemand() {
    let ladder = budget()
    #expect(ladder.observe(contextTokens: 8_000) == nil)
    #expect(ladder.observe(contextTokens: 20_000) != nil)
    #expect(ladder.tier.contextTokens == 32_768)
    #expect(ladder.observe(contextTokens: 32_768) == nil)
    #expect(ladder.observe(contextTokens: 300_000) != nil)
    #expect(ladder.tier.contextTokens == 262_144)
    #expect(ladder.observe(contextTokens: 900_000) == nil)
  }

  @Test("an evicted prefix doubles the slots, up to the ceiling")
  func slotsDoubleUnderPressure() {
    let ladder = budget()
    for expected in [2, 4, 8] {
      #expect(ladder.notePrefixEviction() != nil)
      #expect(ladder.tier.slots == expected)
    }
    #expect(ladder.notePrefixEviction() == nil)
  }

  @Test("the buffer pool doubles only on repeated saturation")
  func poolDoublesOnSustainedPressure() {
    let ladder = budget()
    #expect(ladder.notePoolPressure(cacheMemory: 32 * megabyte) == nil)
    #expect(ladder.notePoolPressure(cacheMemory: 512 * megabyte) == nil)
    #expect(ladder.notePoolPressure(cacheMemory: 32 * megabyte) == nil)
    #expect(ladder.tier.bufferCache == MemoryBudget.bufferFloor)

    #expect(ladder.notePoolPressure(cacheMemory: 512 * megabyte) == nil)
    #expect(ladder.notePoolPressure(cacheMemory: 512 * megabyte) != nil)
    #expect(ladder.tier.bufferCache == gigabyte)
  }

  @Test("a longer context sheds slots rather than overcommitting")
  func contextGrowthShedsSlots() {
    let ladder = budget(ceiling: 12 * gigabyte)
    #expect(ladder.notePrefixEviction() != nil)
    #expect(ladder.notePrefixEviction() != nil)
    #expect(ladder.tier.slots == 4)

    #expect(ladder.observe(contextTokens: 100_000) != nil)
    #expect(ladder.tier.slots == 1)
    #expect(committed(ladder) <= 12 * gigabyte)
  }

  @Test("no ladder step plans past the ceiling, except one context that cannot be shed")
  func staysUnderCeiling() {
    for ceiling in [10, 12, 18, 54, 128].map({ $0 * gigabyte }) {
      let ladder = budget(ceiling: ceiling)
      for demand in [8_000, 40_000, 262_144] {
        _ = ladder.observe(contextTokens: demand)
        for _ in 0..<4 { _ = ladder.notePrefixEviction() }
        for _ in 0..<40 { _ = ladder.notePoolPressure(cacheMemory: ladder.tier.bufferCache) }
      }
      #expect(ladder.tier.slots >= 1)
      if committed(ladder) > ceiling {
        #expect(ladder.tier.slots == 1)
        #expect(ladder.fitsFullContext == false)
      }
    }
  }

  @Test("pinned dimensions do not move")
  func pinsHold() {
    let ladder = budget(slots: 3, bufferCache: 2 * gigabyte)
    #expect(ladder.notePrefixEviction() == nil)
    #expect(ladder.notePoolPressure(cacheMemory: 2 * gigabyte) == nil)
    #expect(ladder.notePoolPressure(cacheMemory: 2 * gigabyte) == nil)
    #expect(ladder.tier.slots == 3)
    #expect(ladder.tier.bufferCache == 2 * gigabyte)

    #expect(ladder.observe(contextTokens: 262_144) != nil)
    #expect(ladder.tier.slots == 3)
  }

  @Test("an unloaded model returns the plan to the floor")
  func resetGoesBackToFloor() {
    let ladder = budget()
    _ = ladder.observe(contextTokens: 200_000)
    _ = ladder.notePrefixEviction()
    #expect(ladder.reset() != nil)
    #expect(ladder.tier.slots == 1)
    #expect(ladder.tier.contextTokens == MemoryBudget.contextFloor)
    #expect(ladder.tier.bufferCache == MemoryBudget.bufferFloor)
    #expect(ladder.reset() == nil)
  }

  private func committed(_ ladder: MemoryBudget) -> Int {
    let tier = ladder.tier
    return 8 * gigabyte + MemoryBudget.workingReserve
      + tier.slots * tier.contextTokens * ladder.bytesPerToken + tier.bufferCache
  }
}
