// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
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

  // The offload itself still needs the INT8 slices in the pack and the merge on the Metal side;
  // until those land, say so rather than accepting the flag and ignoring it.
  func apply() throws {
    BonsaiRuntime.aneOffload = setting
    guard setting == nil else {
      throw ValidationError(
        "--ane is not wired up yet: this pack carries no INT8 Neural Engine slices. "
          + "Measure a split with `ishizuki ane-check` in the meantime.")
    }
  }
}
