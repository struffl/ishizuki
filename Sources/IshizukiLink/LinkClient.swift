// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Everything the phone can ask a paired Mac for, in one place. The route names are the whole of
// the companion API; a Mac serving them is answering this file.

import Foundation
import IshizukiKit

public struct LinkClient: Sendable {
  public var transport: LinkTransport

  public init(transport: LinkTransport) {
    self.transport = transport
  }

  public init(
    endpoint: LinkEndpoint, psk: PreSharedKey, token: String? = nil, connectTimeout: Double = 6
  ) {
    self.transport = LinkTransport(
      endpoint: endpoint, psk: psk, token: token, connectTimeout: connectTimeout)
  }

  /// The first of several addresses to answer, all tried at once. A pairing ticket carries a
  /// tailnet address, a LAN address and a Bonjour name, and which of them is reachable is not
  /// something the phone can know before it asks.
  public static func firstToAnswer(
    _ endpoints: [LinkEndpoint], psk: PreSharedKey, token: String? = nil
  ) async throws -> (client: LinkClient, info: ServerInfo) {
    guard !endpoints.isEmpty else {
      throw LinkTransportError.unreachable("no address to try")
    }
    return try await withThrowingTaskGroup(of: (LinkClient, ServerInfo).self) { group in
      for endpoint in endpoints {
        group.addTask {
          let client = LinkClient(endpoint: endpoint, psk: psk, token: token, connectTimeout: 5)
          return (client, try await client.info())
        }
      }
      var failure: Error = LinkTransportError.unreachable("no address answered")
      while !group.isEmpty {
        do {
          guard let answer = try await group.next() else { break }
          group.cancelAll()
          return answer
        } catch {
          failure = error
        }
      }
      throw failure
    }
  }

  public var endpoint: LinkEndpoint { transport.endpoint }

  public func withToken(_ token: String?) -> LinkClient {
    var copy = self
    copy.transport.token = token
    return copy
  }

  public func withEndpoint(_ endpoint: LinkEndpoint) -> LinkClient {
    var copy = self
    copy.transport.endpoint = endpoint
    return copy
  }

  // MARK: - Pairing

  public func info() async throws -> ServerInfo {
    try await transport.call("GET", "/link/info")
  }

  public func pair(as device: DeviceDescriptor) async throws -> PairGrant {
    try await transport.call("POST", "/link/pair", body: device)
  }

  public func unpair() async throws {
    try await transport.callVoid("POST", "/link/unpair", body: Empty())
  }

  // MARK: - Conversations

  public func chats() async throws -> [ChatSummary] {
    try await transport.call("GET", "/link/chats")
  }

  public func chat(_ id: UUID) async throws -> ChatDetail {
    try await transport.call("GET", "/link/chats/\(id.uuidString)")
  }

  public func create(_ chat: NewChat) async throws -> ChatSummary {
    try await transport.call("POST", "/link/chats", body: chat)
  }

  public func delete(_ id: UUID) async throws {
    try await transport.callVoid("DELETE", "/link/chats/\(id.uuidString)", body: Empty())
  }

  public func change(_ id: UUID, _ change: ChatChange) async throws -> ChatSummary {
    try await transport.call("PATCH", "/link/chats/\(id.uuidString)", body: change)
  }

  public func send(_ id: UUID, text: String) async throws {
    try await transport.callVoid(
      "POST", "/link/chats/\(id.uuidString)/send", body: SendText(text: text))
  }

  public func steer(_ id: UUID, text: String) async throws {
    try await transport.callVoid(
      "POST", "/link/chats/\(id.uuidString)/steer", body: SendText(text: text))
  }

  public func stop(_ id: UUID) async throws {
    try await transport.callVoid("POST", "/link/chats/\(id.uuidString)/stop", body: Empty())
  }

  /// The conversation as it fills: rows, what the turn is doing, and the dials. Held open until
  /// the reading task is cancelled.
  public func events(_ id: UUID) -> AsyncThrowingStream<TurnEvent, Error> {
    transport.events("/link/chats/\(id.uuidString)/events")
  }

  // MARK: - The machine

  public func status() async throws -> LinkStatus {
    try await transport.call("GET", "/link/status")
  }

  public func models() async throws -> ModelList {
    try await transport.call("GET", "/link/models")
  }

  public func activate(model id: String) async throws {
    try await transport.callVoid("POST", "/link/models/activate", body: SendText(text: id))
  }

  public func roots() async throws -> RootList {
    try await transport.call("GET", "/link/roots")
  }

  // MARK: - Files and shell

  public func list(_ path: String) async throws -> DirectoryListing {
    try await transport.call("GET", "/link/files?path=\(Self.escape(path))")
  }

  public func read(_ path: String, offset: Int = 1, limit: Int = 400) async throws -> FileSlice {
    try await transport.call(
      "GET", "/link/file?path=\(Self.escape(path))&offset=\(offset)&limit=\(limit)")
  }

  public func write(_ write: FileWrite) async throws {
    try await transport.callVoid("POST", "/link/file", body: write)
  }

  public func shell(_ request: ShellRequest) async throws -> ShellOutcome {
    try await transport.call(
      "POST", "/link/shell", body: request,
      timeout: max(30, min(900, request.timeout ?? 120) + 10))
  }

  static func escape(_ text: String) -> String {
    text.addingPercentEncoding(
      withAllowedCharacters: .alphanumerics.union(CharacterSet(charactersIn: "-._~/"))) ?? text
  }
}

public struct Empty: Codable, Sendable {
  public init() {}
}
