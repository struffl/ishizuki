// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The checks and sweeps, pointed at whichever pack is selected.

import Foundation
import IshizukiKit
import Observation

enum BenchKind: String, CaseIterable, Identifiable {
  case kernelCheck
  case batchCheck
  case kvBench
  case contextBench
  case specBench
  case prefillBench
  case ggmlBench

  var id: String { rawValue }

  var title: String {
    switch self {
    case .kernelCheck: "Kernel check"
    case .batchCheck: "Batch check"
    case .kvBench: "KV cache sweep"
    case .contextBench: "Context sweep"
    case .specBench: "Speculative decoding"
    case .prefillBench: "Prefill"
    case .ggmlBench: "GGML kernels"
    }
  }

  var summary: String {
    switch self {
    case .kernelCheck: "The wide matvec kernel against MLX, for agreement and for speed."
    case .batchCheck: "Batched decoding against sequential, for the same tokens and the rate."
    case .kvBench: "What each KV bit budget costs in memory and in divergence."
    case .contextBench: "Memory and rate as the context grows, extrapolated past what fits."
    case .specBench: "Draft acceptance and the rate it buys over greedy."
    case .prefillBench: "Per-module prefill timing, a layer at a time."
    case .ggmlBench: "One tensor per GGML type, dequantized and timed. Needs a GGUF."
    }
  }

  /// GGUF kernels are timed on the file itself, so a pack directory cannot stand in.
  var needsGGUF: Bool { self == .ggmlBench }
}

@MainActor
@Observable
final class BenchController {
  var kind: BenchKind = .kernelCheck

  func canRun(_ entry: ModelCatalog.Entry?) -> Bool {
    guard let entry else { return false }
    return kind.needsGGUF ? entry.format == .gguf : true
  }

  func start(on runner: JobRunner, entry: ModelCatalog.Entry, neuralEngine: Bool) {
    let url = entry.url
    let kind = kind

    runner.run(kind.title) { log in
      let emit: @Sendable (String) -> Void = { log.line($0) }
      switch kind {
      case .kernelCheck:
        try KernelCheck.run(KernelCheck.Options(model: url), log: emit)
      case .batchCheck:
        try BatchCheck.run(BatchCheck.Options(model: url), log: emit)
      case .kvBench:
        try KVBench.run(KVBench.Options(model: url), log: emit)
      case .contextBench:
        try ContextBench.run(ContextBench.Options(model: url), log: emit)
      case .specBench:
        try SpecBench.run(SpecBench.Options(model: url), log: emit)
      case .prefillBench:
        var options = PrefillBench.Options(model: url)
        options.ane = neuralEngine ? .automatic : nil
        try PrefillBench.run(options, log: emit)
      case .ggmlBench:
        try GGMLBench.run(GGMLBench.Options(gguf: url), log: emit)
      }
    }
  }
}
