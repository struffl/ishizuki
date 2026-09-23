// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

// Probe, not a test: speculative decoding with the few-row verify kernel off and on.
@Suite("SpecGainProbe", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_PACK"] != nil))
struct SpecGainProbe {
  @Test("probe")
  func probe() throws {
    let path = ProcessInfo.processInfo.environment["ISHIZUKI_PACK"]!
    let drafter = ProcessInfo.processInfo.environment["ISHIZUKI_DRAFTER"] ?? "ngram"
    for draft in [4, 7] {
      for enabled in [false, true] {
        BonsaiRuntime.useVerifyMatmul = enabled
        print("== draft \(draft), verify kernel \(enabled ? "on" : "off")")
        var options = SpecBench.Options(model: URL(filePath: path))
        options.draftLength = draft
        options.drafter = drafter
        options.maxTokens = 160
        try SpecBench.run(options) { print($0) }
      }
    }
    BonsaiRuntime.useVerifyMatmul = true
  }
}
