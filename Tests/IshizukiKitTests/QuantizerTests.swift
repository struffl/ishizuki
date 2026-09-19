// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("Quantizer")
struct QuantizerTests {
  @Test("a narrower width loses more of the weight than a wider one")
  func errorOrders() {
    let w = MLXRandom.normal([256, 512]).asType(.float16)
    let measured = ModuleSurvey.measure(w, path: "m", widths: [2, 3, 4, 5], groupSize: 64)
    #expect(measured.elements == 256 * 512)
    #expect(measured.error(2) > measured.error(3))
    #expect(measured.error(3) > measured.error(4))
    #expect(measured.error(4) > measured.error(5))
    #expect(measured.error(5) > 0)
  }

  @Test("the group's scale and bias are counted, not just the nominal width")
  func bpwIncludesOverhead() {
    // A "4-bit" group-64 module occupies 4.5 bits per weight, which is why oQ4e lands near 4.6.
    #expect(ModuleMeasurement.bpw(bits: 4, groupSize: 64) == 4.5)
    #expect(ModuleMeasurement.bpw(bits: 3, groupSize: 128) == 3.25)
  }

  private func measurement(
    _ path: String, elements: Int, errors: [Int: Double]
  ) -> ModuleMeasurement {
    ModuleMeasurement(path: path, elements: elements, errorAt: errors)
  }

  @Test("the budget is spent where it buys the most, and is never overspent")
  func allocatesByValue() {
    // Three equal modules: one suffers badly at the base width, one moderately, one barely.
    let modules = [
      measurement("bad", elements: 1 << 20, errors: [3: 0.30, 4: 0.10, 5: 0.05]),
      measurement("middling", elements: 1 << 20, errors: [3: 0.12, 4: 0.09, 5: 0.08]),
      measurement("fine", elements: 1 << 20, errors: [3: 0.02, 4: 0.019, 5: 0.018]),
    ]
    // Base 3-bit is 3.5 bpw with group 64, so 3.9 leaves room to lift roughly one module.
    let result = BitAllocator(profile: .quality).allocate(modules)

    #expect(result.achievedBpw <= 3.9)
    #expect(result.bits["bad"]! > result.bits["fine"]!)
    #expect(result.bits["fine"] == 3)
    #expect(result.boosted >= 1)
  }

  @Test("a budget with nothing to spend leaves every module at the base width")
  func tightBudget() {
    let modules = [
      measurement("a", elements: 1 << 20, errors: [3: 0.3, 4: 0.1, 5: 0.05]),
      measurement("b", elements: 1 << 20, errors: [3: 0.3, 4: 0.1, 5: 0.05]),
    ]
    // 3-bit at group 64 is exactly 3.5 bpw: the base costs the whole budget.
    let profile = QuantProfile(
      name: "exact", baseBits: 3, boostBits: [4, 5], targetBpw: 3.5, summary: "")
    let result = BitAllocator(profile: profile).allocate(modules)
    #expect(result.boosted == 0)
    #expect(result.achievedBpw == 3.5)
    #expect(result.histogram == [3: 2])
  }

  @Test("the steeper improvement wins, whatever the modules weigh")
  func valueIsPerByte() {
    // Same size, so only the shape of the improvement separates them.
    let modules = [
      measurement("steep", elements: 1 << 20, errors: [3: 0.30, 4: 0.05]),
      measurement("shallow", elements: 1 << 20, errors: [3: 0.30, 4: 0.28]),
    ]
    // Base 3-bit is 3.5 bpw at group 64, so 4.0 leaves room for exactly one of the two.
    let profile = QuantProfile(
      name: "one", baseBits: 3, boostBits: [4], targetBpw: 4.0, summary: "")
    let result = BitAllocator(profile: profile).allocate(modules)
    #expect(result.bits["steep"] == 4)
    #expect(result.bits["shallow"] == 3)
  }

  @Test("a lift that does not fit is skipped rather than shrinking the model past its target")
  func neverOverspends() {
    // The large module cannot be lifted inside the budget; the small one can.
    let modules = [
      measurement("large", elements: 1 << 22, errors: [3: 0.30, 4: 0.05]),
      measurement("small", elements: 1 << 16, errors: [3: 0.20, 4: 0.10]),
    ]
    let profile = QuantProfile(
      name: "tight", baseBits: 3, boostBits: [4], targetBpw: 3.6, summary: "")
    let result = BitAllocator(profile: profile).allocate(modules)
    #expect(result.bits["large"] == 3)
    #expect(result.bits["small"] == 4)
    #expect(result.achievedBpw <= 3.6)
  }
}
