// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What a listener needs to be reachable and private: a pre-shared key both ends were told once,
// and a Bonjour name so the other end can find it without being told an address.

import Foundation
import Network

/// A key handed to one device at pairing time and held in its keychain afterwards. Both ends
/// build the same TLS options from it, so completing a handshake is itself the proof of pairing.
public struct PreSharedKey: Sendable, Equatable {
  public static let defaultIdentity = "ishizuki"

  public let key: Data
  public let identity: String

  public init(key: Data, identity: String = PreSharedKey.defaultIdentity) {
    self.key = key
    self.identity = identity
  }

  /// A fresh 256-bit key, which is what a Mac generates the first time the companion is switched on.
  public static func generate(identity: String = PreSharedKey.defaultIdentity) -> PreSharedKey {
    var bytes = [UInt8](repeating: 0, count: 32)
    for index in bytes.indices { bytes[index] = UInt8.random(in: .min ... .max) }
    return PreSharedKey(key: Data(bytes), identity: identity)
  }

  public var base64: String { key.base64EncodedString() }

  public init?(base64: String, identity: String = PreSharedKey.defaultIdentity) {
    guard let data = Data(base64Encoded: base64), !data.isEmpty else { return nil }
    self.init(key: data, identity: identity)
  }

  /// The first four bytes, spelled out, so a pairing sheet and a phone can be seen to agree.
  public var fingerprint: String {
    key.prefix(4).map { String(format: "%02X", $0) }.joined()
  }

  public func tlsOptions() -> NWProtocolTLS.Options {
    let options = NWProtocolTLS.Options()
    let keyData = key.withUnsafeBytes { DispatchData(bytes: $0) }
    let identityData = Data(identity.utf8).withUnsafeBytes { DispatchData(bytes: $0) }
    sec_protocol_options_add_pre_shared_key(
      options.securityProtocolOptions,
      keyData as __DispatchData,
      identityData as __DispatchData)
    if let suite = tls_ciphersuite_t(rawValue: UInt16(TLS_PSK_WITH_AES_128_GCM_SHA256)) {
      sec_protocol_options_append_tls_ciphersuite(options.securityProtocolOptions, suite)
    }
    return options
  }

  public func parameters() -> NWParameters {
    let parameters = NWParameters(tls: tlsOptions())
    parameters.allowLocalEndpointReuse = true
    return parameters
  }
}

/// How a listener names itself on the local network. Type is a Bonjour service type, which is
/// what both mDNSResponder and avahi answer browse requests for.
public struct BonjourService: Sendable, Equatable {
  public static let linkType = "_ishizuki._tcp"

  public var name: String
  public var type: String
  public var txt: [String: String]

  public init(name: String, type: String = BonjourService.linkType, txt: [String: String] = [:]) {
    self.name = name
    self.type = type
    self.txt = txt
  }

  var service: NWListener.Service {
    NWListener.Service(name: name, type: type, domain: nil, txtRecord: NWTXTRecord(txt).data)
  }
}
