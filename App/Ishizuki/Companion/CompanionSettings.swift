// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What the companion listener is allowed to do, kept between launches. Nothing here is a secret;
// the key and the device tokens live in the keychain.

import Foundation
import IshizukiKit
import IshizukiLink
import Observation

@MainActor
@Observable
final class CompanionSettings {
  var enabled: Bool { didSet { write(enabled, "companion.enabled") } }
  var port: Int { didSet { write(port, "companion.port") } }
  var serviceName: String { didSet { write(serviceName, "companion.serviceName") } }
  /// Whether a paired phone may run commands, as against reading files and having conversations.
  var allowShell: Bool { didSet { write(allowShell, "companion.allowShell") } }
  /// The folders a phone may see. Empty means the home directory and nothing above it.
  var roots: [String] { didSet { write(roots, "companion.roots") } }
  var devices: [DeviceDescriptor] {
    didSet {
      guard let data = try? JSONEncoder.link.encode(devices) else { return }
      defaults.set(data, forKey: "companion.devices")
    }
  }

  private let defaults = UserDefaults.standard

  init() {
    let defaults = UserDefaults.standard
    enabled = defaults.object(forKey: "companion.enabled") as? Bool ?? false
    port = defaults.object(forKey: "companion.port") as? Int ?? Int(Link.defaultPort)
    serviceName =
      defaults.object(forKey: "companion.serviceName") as? String
      ?? (Host.current().localizedName ?? "Ishizuki")
    allowShell = defaults.object(forKey: "companion.allowShell") as? Bool ?? true
    roots = defaults.object(forKey: "companion.roots") as? [String] ?? []
    devices =
      (defaults.data(forKey: "companion.devices")
        .flatMap { try? JSONDecoder.link.decode([DeviceDescriptor].self, from: $0) }) ?? []
  }

  /// The folders a phone is allowed to reach, as URLs. The home directory stands in when nothing
  /// has been chosen, which is what makes the companion useful the moment it is switched on.
  var allowedRoots: [URL] {
    let paths = roots.isEmpty ? [FileManager.default.homeDirectoryForCurrentUser.path] : roots
    return paths.map { URL(filePath: $0).standardizedFileURL }
  }

  private func write(_ value: Any, _ key: String) {
    defaults.set(value, forKey: key)
  }
}
