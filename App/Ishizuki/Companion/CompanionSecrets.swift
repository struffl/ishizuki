// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The companion's key and its per-device tokens. A key that opens a shell on this Mac belongs in
// the keychain rather than in defaults beside the window's font size.

import Foundation
import IshizukiKit
import Security

struct CompanionSecrets: Codable, Sendable {
  var psk: String
  var tokens: [String: String]

  init(psk: String, tokens: [String: String] = [:]) {
    self.psk = psk
    self.tokens = tokens
  }

  var key: PreSharedKey? { PreSharedKey(base64: psk) }
}

/// One keychain item, read and written whole. There is little enough of it that a merge would
/// cost more than it saved.
enum CompanionVault {
  private static let service = "studio.ishizuki.companion"
  private static let account = "link"

  static func load() -> CompanionSecrets {
    if let data = read(), let secrets = try? JSONDecoder().decode(CompanionSecrets.self, from: data)
    {
      return secrets
    }
    let fresh = CompanionSecrets(psk: PreSharedKey.generate().base64)
    save(fresh)
    return fresh
  }

  static func save(_ secrets: CompanionSecrets) {
    guard let data = try? JSONEncoder().encode(secrets) else { return }
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
    ]
    let attributes: [String: Any] = [kSecValueData as String: data]
    if SecItemUpdate(query as CFDictionary, attributes as CFDictionary) == errSecSuccess { return }
    var insert = query
    insert[kSecValueData as String] = data
    insert[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked
    SecItemAdd(insert as CFDictionary, nil)
  }

  /// Forgets the key itself, which is the only way to shut out a device that was once paired and
  /// still remembers what it was told.
  static func rotate() -> CompanionSecrets {
    let fresh = CompanionSecrets(psk: PreSharedKey.generate().base64)
    save(fresh)
    return fresh
  }

  private static func read() -> Data? {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: account,
      kSecReturnData as String: true,
      kSecMatchLimit as String: kSecMatchLimitOne,
    ]
    var item: CFTypeRef?
    guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess else { return nil }
    return item as? Data
  }

  static func token() -> String {
    var bytes = [UInt8](repeating: 0, count: 24)
    for index in bytes.indices { bytes[index] = UInt8.random(in: .min ... .max) }
    return Data(bytes).base64EncodedString()
  }
}
