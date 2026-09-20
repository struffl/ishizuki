// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import IOKit

/// Device utilization, read from the accelerator's own IOKit performance statistics.
public enum GPUMeter {
  private static let interval: TimeInterval = 0.5
  private static let lock = NSLock()
  private nonisolated(unsafe) static var cached: (value: Double?, taken: Date)?

  public static func utilization() -> Double? {
    lock.lock()
    if let cached, -cached.taken.timeIntervalSinceNow < interval {
      defer { lock.unlock() }
      return cached.value
    }
    lock.unlock()

    let fresh = read()
    lock.lock()
    cached = (fresh, Date())
    lock.unlock()
    return fresh
  }

  private static func read() -> Double? {
    var iterator: io_iterator_t = 0
    guard
      IOServiceGetMatchingServices(
        kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS
    else { return nil }
    defer { IOObjectRelease(iterator) }

    while case let service = IOIteratorNext(iterator), service != 0 {
      defer { IOObjectRelease(service) }
      var properties: Unmanaged<CFMutableDictionary>?
      guard
        IORegistryEntryCreateCFProperties(service, &properties, kCFAllocatorDefault, 0)
          == KERN_SUCCESS,
        let dictionary = properties?.takeRetainedValue() as? [String: Any],
        let statistics = dictionary["PerformanceStatistics"] as? [String: Any]
      else { continue }
      for key in ["Device Utilization %", "GPU Activity(%)", "Renderer Utilization %"] {
        if let value = statistics[key] as? NSNumber {
          return min(max(Double(value.doubleValue) / 100, 0), 1)
        }
      }
    }
    return nil
  }
}
