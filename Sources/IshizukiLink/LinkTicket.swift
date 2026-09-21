// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What a pairing QR carries: where the Mac is, and the key that makes a handshake with it mean
// something. Small on purpose — it has to survive being a square on a screen.

import Foundation
import IshizukiKit

public struct LinkTicket: Codable, Sendable, Equatable {
  public var version: Int
  public var serverName: String
  /// Every address the Mac believes it can be reached at, best first. A Tailscale name is one of
  /// these, which is how the link works from somewhere Bonjour cannot reach.
  public var hosts: [String]
  public var port: UInt16
  public var psk: String
  public var service: String?

  public init(
    version: Int = Link.protocolVersion, serverName: String, hosts: [String], port: UInt16,
    psk: String, service: String? = nil
  ) {
    self.version = version
    self.serverName = serverName
    self.hosts = hosts
    self.port = port
    self.psk = psk
    self.service = service
  }

  public var key: PreSharedKey? { PreSharedKey(base64: psk) }

  /// The one string a QR encodes, and the one a phone can be told to type if a camera will not
  /// cooperate.
  public var url: URL? {
    guard let data = try? JSONEncoder.link.encode(self) else { return nil }
    return URL(string: "ishizuki://pair/" + Self.encode(data))
  }

  public init?(url: URL) {
    guard url.scheme == "ishizuki", url.host == "pair" else { return nil }
    let encoded = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    guard let data = Self.decode(encoded),
      let ticket = try? JSONDecoder.link.decode(LinkTicket.self, from: data)
    else { return nil }
    self = ticket
  }

  public init?(string: String) {
    let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
    if let url = URL(string: trimmed), url.scheme == "ishizuki" {
      self.init(url: url)
      return
    }
    guard let data = Self.decode(trimmed),
      let ticket = try? JSONDecoder.link.decode(LinkTicket.self, from: data)
    else { return nil }
    self = ticket
  }

  private static func encode(_ data: Data) -> String {
    data.base64EncodedString()
      .replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "=", with: "")
  }

  private static func decode(_ text: String) -> Data? {
    var padded =
      text
      .replacingOccurrences(of: "-", with: "+")
      .replacingOccurrences(of: "_", with: "/")
    while padded.count % 4 != 0 { padded += "=" }
    return Data(base64Encoded: padded)
  }
}

/// A Mac the phone has been paired with, as the phone remembers it. The key and the token live
/// in the keychain; this is the part that is safe to keep in defaults beside it.
public struct KnownServer: Codable, Sendable, Equatable, Identifiable {
  public var id: String
  public var name: String
  public var hosts: [String]
  public var port: UInt16
  public var pairedAt: Date
  public var lastConnected: Date?

  public init(
    id: String, name: String, hosts: [String], port: UInt16, pairedAt: Date = Date(),
    lastConnected: Date? = nil
  ) {
    self.id = id
    self.name = name
    self.hosts = hosts
    self.port = port
    self.pairedAt = pairedAt
    self.lastConnected = lastConnected
  }
}
