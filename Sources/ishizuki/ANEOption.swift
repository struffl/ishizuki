// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit

struct ANEOption: ParsableArguments {
  @Option(
    name: .long,
    help: ArgumentHelp(
      "Share of the wide projections to prefill on the Neural Engine: a fraction, or auto.",
      discussion: """
        Off unless asked for; bare --ane means auto. The two halves run concurrently, so auto \
        times both and picks the split where they finish together — that balance moves with the \
        machine's Metal-to-ANE ratio, so it is measured rather than assumed. Offloaded channels \
        are held as INT8 beside the 2-bit weights, costing roughly 0.72 bytes per offloaded \
        parameter, so a smaller fraction trades speed back for memory.
        """,
      valueName: "fraction|auto"))
  var ane: String?

  // ArgumentParser has no optional-valued option, so a bare --ane is given its "auto" here,
  // before parsing. A value already attached, by space or by =, is left alone.
  static func normalize(_ arguments: [String]) -> [String] {
    var out: [String] = []
    var index = arguments.startIndex
    while index < arguments.endIndex {
      let argument = arguments[index]
      out.append(argument)
      if argument == "--ane" {
        let next = arguments.index(after: index)
        let value = next < arguments.endIndex ? arguments[next] : nil
        // Anything that is not the next option is left for the parser to accept or reject, so a
        // bad value still earns its own message rather than becoming a stray argument.
        let carries = value.map { Double($0) != nil || !$0.hasPrefix("-") } ?? false
        if !carries { out.append("auto") }
      }
      index = arguments.index(after: index)
    }
    return out
  }

  var setting: ANEOffload? {
    guard let ane else { return nil }
    if ane == "auto" { return .automatic }
    return Double(ane).map { .fraction($0) }
  }

  func validate() throws {
    guard let ane else { return }
    guard let setting else {
      throw ValidationError("--ane takes a fraction in (0, 1] or auto; got \(ane)")
    }
    if case .fraction(let value) = setting, !(value > 0 && value <= 1) {
      throw ValidationError("--ane takes a fraction in (0, 1] or auto; got \(ane)")
    }
  }

  // The channel split is settled when the slices are cut, so the pack decides it and the flag
  // only says whether to use them — and refuses a fraction the pack was not cut at.
  func apply(pack: URL) throws {
    BonsaiRuntime.aneOffload = setting
    guard let setting else { return }

    guard let bank = ANEBank(pack: pack) else {
      throw ValidationError(
        "this pack carries no Neural Engine slices. Cut them with "
          + "`uv run --with coremltools --with numpy Scripts/ane-export.py`.")
    }
    if case .fraction(let wanted) = setting, let cut = bank.fraction, abs(cut - wanted) > 0.02 {
      throw ValidationError(
        String(
          format: "this pack's slices are cut at %.2f, not %.2f — re-export to change the split",
          cut, wanted))
    }
    BonsaiRuntime.aneBank = bank
  }
}
