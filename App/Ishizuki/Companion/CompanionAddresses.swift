// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Every way this Mac can be reached, best first. A Tailscale address is preferred to a LAN one
// because it keeps working from somewhere else; the Bonjour name comes last because it only
// resolves on the network the phone is standing on.

import Foundation

enum CompanionAddresses {
  static func hosts() -> [String] {
    var tailscale: [String] = []
    var lan: [String] = []

    var pointer: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&pointer) == 0, let first = pointer else { return named() }
    defer { freeifaddrs(pointer) }

    for interface in sequence(first: first, next: { $0.pointee.ifa_next }) {
      let flags = Int32(interface.pointee.ifa_flags)
      guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }
      guard let address = interface.pointee.ifa_addr,
        address.pointee.sa_family == UInt8(AF_INET)
      else { continue }

      var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
      guard
        getnameinfo(
          address, socklen_t(address.pointee.sa_len), &host, socklen_t(host.count), nil, 0,
          NI_NUMERICHOST) == 0
      else { continue }
      let text = String(cString: host)
      guard !text.isEmpty, !text.hasPrefix("169.254.") else { continue }
      if isTailscale(text) {
        tailscale.append(text)
      } else {
        lan.append(text)
      }
    }
    return tailscale + lan + named()
  }

  /// Tailscale hands out addresses from the carrier-grade NAT range, which is how one is told
  /// apart from an address the local router gave out.
  private static func isTailscale(_ address: String) -> Bool {
    let parts = address.split(separator: ".").compactMap { Int($0) }
    guard parts.count == 4, parts[0] == 100 else { return false }
    return (64...127).contains(parts[1])
  }

  private static func named() -> [String] {
    var names: [String] = []
    let hostName = ProcessInfo.processInfo.hostName
    if !hostName.isEmpty, hostName != "localhost" { names.append(hostName) }
    return names
  }
}
