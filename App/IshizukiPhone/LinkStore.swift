// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The phone's one connection to a Mac: how it was paired, which of its addresses answered last,
// and what to try next when that one stops answering.

import Foundation
import IshizukiKit
import IshizukiLink
import Observation
import UIKit

@MainActor
@Observable
final class LinkStore {
  enum Phase: Equatable {
    case unpaired
    case connecting
    case ready
    case offline(String)

    var isReady: Bool { self == .ready }
  }

  private(set) var phase: Phase = .unpaired
  private(set) var info: ServerInfo?
  private(set) var known: KnownServer?
  private(set) var client: LinkClient?
  private(set) var cache: LinkCache?
  private var connectionTask: Task<Void, Never>?
  private var connectionID = UUID()

  let browser = LinkBrowser()

  private let defaults = UserDefaults.standard

  init() {
    if let data = defaults.data(forKey: "link.server"),
      let server = try? JSONDecoder.link.decode(KnownServer.self, from: data)
    {
      known = server
      cache = LinkCache(serverID: server.id)
      phase = .connecting
    }
  }

  var deviceID: String {
    if let saved = defaults.string(forKey: "link.deviceID") { return saved }
    let made = UIDevice.current.identifierForVendor?.uuidString ?? UUID().uuidString
    defaults.set(made, forKey: "link.deviceID")
    return made
  }

  private var descriptor: DeviceDescriptor {
    DeviceDescriptor(
      id: deviceID,
      name: UIDevice.current.name,
      system: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)")
  }

  // MARK: - Pairing

  /// Takes a ticket read from a square on the Mac's screen and tries every address in it until
  /// one answers. What comes back is a token; the key was in the ticket.
  func pair(with ticket: LinkTicket) async throws {
    guard let key = ticket.key else {
      throw LinkFailure(code: "bad_ticket", message: "that pairing code was not readable")
    }
    connectionTask?.cancel()
    connectionTask = nil
    let attempt = UUID()
    connectionID = attempt
    phase = .connecting

    let endpoints = Self.endpoints(hosts: ticket.hosts, port: ticket.port, service: ticket.service)
    do {
      let answer = try await LinkClient.firstToAnswer(endpoints, psk: key)
      let grant = try await answer.client.pair(as: descriptor)
      guard connectionID == attempt else { throw CancellationError() }
      let id = key.fingerprint
      cache = LinkCache(serverID: id)
      PhoneVault.save(ServerCredentials(psk: ticket.psk, token: grant.token), for: id)
      remember(
        KnownServer(
          id: id, name: grant.serverName,
          hosts: Self.hosts(of: answer.client.endpoint, hosts: ticket.hosts), port: ticket.port,
          pairedAt: Date(), lastConnected: Date()))
      client = answer.client.withToken(grant.token)
      info = grant.info
      phase = .ready
    } catch {
      if connectionID == attempt { phase = .offline(error.localizedDescription) }
      throw error
    }
  }

  func forget() {
    connectionID = UUID()
    connectionTask?.cancel()
    connectionTask = nil
    if let cache { Task { await cache.clear() } }
    cache = nil
    if let known {
      let previous = client
      Task { try? await previous?.unpair() }
      PhoneVault.forget(known.id)
    }
    defaults.removeObject(forKey: "link.server")
    known = nil
    client = nil
    info = nil
    phase = .unpaired
  }

  // MARK: - Connecting

  /// Tries the address that worked last, then the rest, then the Bonjour name. A Mac that moved
  /// between a LAN and Tailscale is found again without being paired twice.
  func connect() async {
    if let connectionTask {
      await connectionTask.value
      return
    }
    let attempt = UUID()
    connectionID = attempt
    let task = Task { await establish(attempt) }
    connectionTask = task
    await task.value
    if connectionID == attempt { connectionTask = nil }
  }

  private func establish(_ attempt: UUID) async {
    guard let known, let credentials = PhoneVault.credentials(for: known.id),
      let key = credentials.key
    else {
      phase = known == nil ? .unpaired : .offline("the key for this Mac is missing")
      return
    }
    phase = .connecting

    let endpoints = Self.endpoints(hosts: known.hosts, port: known.port, service: known.name)
    do {
      let answer = try await LinkClient.firstToAnswer(
        endpoints, psk: key, token: credentials.token)
      guard connectionID == attempt, !Task.isCancelled else { return }
      info = answer.info
      client = answer.client
      phase = .ready
      var updated = known
      updated.hosts = Self.hosts(of: answer.client.endpoint, hosts: known.hosts)
      updated.lastConnected = Date()
      remember(updated)
    } catch {
      guard connectionID == attempt, !Task.isCancelled else { return }
      client = nil
      phase = .offline(error.localizedDescription)
    }
  }

  func refresh() async {
    guard let client else {
      await connect()
      return
    }
    let serverID = known?.id
    do {
      let fresh = try await client.info()
      guard known?.id == serverID, !Task.isCancelled else { return }
      info = fresh
      phase = .ready
    } catch {
      guard known?.id == serverID, !Task.isCancelled else { return }
      await connect()
    }
  }

  private func remember(_ server: KnownServer) {
    known = server
    guard let data = try? JSONEncoder.link.encode(server) else { return }
    defaults.set(data, forKey: "link.server")
  }

  private static func endpoints(hosts: [String], port: UInt16, service: String?) -> [LinkEndpoint] {
    var endpoints = hosts.map { LinkEndpoint(host: $0, port: port) }
    if let service, !service.isEmpty {
      endpoints.append(.bonjour(name: service))
    }
    return endpoints
  }

  /// Whichever address answered goes to the front, so the next launch starts where the last one
  /// left off rather than racing the whole list again.
  private static func hosts(of endpoint: LinkEndpoint, hosts: [String]) -> [String] {
    guard case .address(let host, _) = endpoint.kind else { return hosts }
    return [host] + hosts.filter { $0 != host }
  }
}
