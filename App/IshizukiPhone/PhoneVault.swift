// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What the phone keeps about a Mac it has paired with: the key it was handed and the token that
// says which device it is. Both belong in the keychain, so neither survives a backup in the clear.

import Foundation
import IshizukiKit
import Security

struct ServerCredentials: Codable, Sendable {
  var psk: String
  var token: String

  var key: PreSharedKey? { PreSharedKey(base64: psk) }
}

enum PhoneVault {
  private static let service = "studio.ishizuki.phone"

  static func credentials(for id: String) -> ServerCredentials? {
    guard let data = read(id) else { return nil }
    return try? JSONDecoder().decode(ServerCredentials.self, from: data)
  }

  static func save(_ credentials: ServerCredentials, for id: String) {
    guard let data = try? JSONEncoder().encode(credentials) else { return }
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: id,
    ]
    if SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
      == errSecSuccess
    {
      return
    }
    var insert = query
    insert[kSecValueData as String] = data
    insert[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
    SecItemAdd(insert as CFDictionary, nil)
  }

  static func forget(_ id: String) {
    SecItemDelete(
      [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: service,
        kSecAttrAccount as String: id,
      ] as CFDictionary)
  }

  private static func read(_ id: String) -> Data? {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: id,
      kSecReturnData as String: true,
      kSecMatchLimit as String: kSecMatchLimitOne,
    ]
    var item: CFTypeRef?
    guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess else { return nil }
    return item as? Data
  }
}
