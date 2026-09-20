// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What this Mac can hold, so a pack that cannot run here is not offered as if it could.

import Foundation
import IshizukiKit
import Metal

enum Machine {
  static let physicalMemory = Int(ProcessInfo.processInfo.physicalMemory)

  /// What Metal says it will let one process keep resident, which is the real ceiling on
  /// weights — smaller than the installed memory, and what the pack has to fit inside.
  static let workingSet: Int = {
    guard let device = MTLCreateSystemDefaultDevice() else { return physicalMemory }
    return Int(device.recommendedMaxWorkingSetSize)
  }()

  static let deviceName = MTLCreateSystemDefaultDevice()?.name ?? "no Metal device"

  /// Weights plus the headroom a run needs for activations, cache and the rest of the system.
  static func fits(weightBytes: Int) -> Bool {
    weightBytes + MemoryBudget.workingReserve <= workingSet
  }

  static var summary: String {
    "\(ReadoutFormat.gigabytes(physicalMemory)) unified · "
      + "\(ReadoutFormat.gigabytes(workingSet)) addressable · \(deviceName)"
  }
}
