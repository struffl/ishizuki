// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The listener a phone talks to: Bonjour-advertised, TLS with a key handed over once by QR, and
// a token per device so one can be forgotten without the others noticing.

import Foundation
import IshizukiKit
import IshizukiLink
import Observation

@available(macOS 27.0, *)
@MainActor
@Observable
final class CompanionServer {
  enum Phase: Equatable {
    case stopped
    case running(UInt16)
    case failed(String)

    var isRunning: Bool { if case .running = self { true } else { false } }
  }

  private(set) var phase: Phase = .stopped
  /// The ticket a QR is drawn from, present only while the pairing window is open.
  private(set) var ticket: LinkTicket?
  private(set) var pairingCloses: Date?
  private(set) var log: [String] = []

  let settings = CompanionSettings()

  private var secrets = CompanionVault.load()
  private var server: HTTPServer?
  private var bridge: CompanionBridge?
  private var pairingTimer: Task<Void, Never>?
  private let logLimit = 120

  var key: PreSharedKey? { secrets.key }
  var fingerprint: String { secrets.key?.fingerprint ?? "—" }
  var devices: [DeviceDescriptor] { settings.devices }
  var isPairing: Bool { ticket != nil }

  func attach(chat: ChatController, server controller: ServerController) {
    bridge = CompanionBridge(chat: chat, server: controller, settings: settings)
    if settings.enabled, !phase.isRunning { start() }
  }

  func start() {
    guard !phase.isRunning else { return }
    guard let key = secrets.key else {
      phase = .failed("no key in the keychain")
      return
    }
    let port = UInt16(clamping: settings.port)
    do {
      let listener = try HTTPServer(
        port: port,
        psk: key,
        advertise: BonjourService(
          name: settings.serviceName,
          txt: [
            "version": String(Link.protocolVersion),
            "model": bridge?.info(pairingOpen: false).model ?? "",
            "fingerprint": key.fingerprint,
          ])
      ) { [weak self] request, writer in
        Task { @MainActor in self?.route(request, writer) }
      }
      listener.start()
      server = listener
      phase = .running(port)
      settings.enabled = true
      append("listening on \(port), advertised as \(settings.serviceName)")
    } catch {
      phase = .failed(String(describing: error))
    }
  }

  func stop() {
    server?.stop()
    server = nil
    closePairing()
    phase = .stopped
    settings.enabled = false
    append("stopped")
  }

  // MARK: - Pairing

  /// Opens a window during which a phone that can complete the handshake may ask for a token.
  /// Anyone who has the key can already reach the port; what this gates is becoming known to it.
  func openPairing(minutes: Double = 3) {
    guard let key = secrets.key else { return }
    if !phase.isRunning { start() }
    guard case .running(let port) = phase else { return }
    ticket = LinkTicket(
      serverName: settings.serviceName,
      hosts: CompanionAddresses.hosts(),
      port: port,
      psk: key.base64,
      service: settings.serviceName)
    pairingCloses = Date().addingTimeInterval(minutes * 60)
    pairingTimer?.cancel()
    pairingTimer = Task { [weak self] in
      try? await Task.sleep(for: .seconds(minutes * 60))
      guard !Task.isCancelled else { return }
      self?.closePairing()
    }
    append("pairing open for \(Int(minutes)) minutes")
  }

  func closePairing() {
    pairingTimer?.cancel()
    pairingTimer = nil
    ticket = nil
    pairingCloses = nil
  }

  func forget(_ device: DeviceDescriptor) {
    settings.devices.removeAll { $0.id == device.id }
    secrets.tokens.removeValue(forKey: device.id)
    CompanionVault.save(secrets)
    append("forgot \(device.name)")
  }

  /// Rotates the key, which is the only thing that shuts out a device that still remembers it.
  func forgetEverything() {
    settings.devices = []
    secrets = CompanionVault.rotate()
    closePairing()
    if phase.isRunning {
      stop()
      start()
    }
    append("rotated the key; every device must pair again")
  }

  // MARK: - Routing

  private func route(_ request: HTTPRequest, _ writer: ResponseWriter) {
    guard let bridge else {
      writer.sendError(status: 503, type: "not_ready", message: "this Mac is still starting up")
      return
    }
    let parts = request.path.split(separator: "?", maxSplits: 1)
    let path = String(parts.first ?? "")
    let query = Self.query(parts.count > 1 ? String(parts[1]) : "")

    if request.method == "OPTIONS" {
      writer.send(json: [:])
      return
    }

    switch (request.method, path) {
    case ("GET", "/link/info"):
      respond(writer, bridge.info(pairingOpen: isPairing))
      return
    case ("POST", "/link/pair"):
      pair(request, writer, bridge)
      return
    default:
      break
    }

    guard let device = authenticate(request) else {
      writer.sendError(
        status: 401, type: "unpaired", message: "this device is not paired with this Mac")
      return
    }
    touch(device)

    Task { @MainActor in
      do {
        try await self.dispatch(request, writer, bridge, path: path, query: query)
      } catch let failure as LinkFailure {
        self.fail(
          writer, status: ["no_chat", "no_job"].contains(failure.code) ? 404 : 409, failure)
      } catch {
        self.fail(
          writer, status: 500,
          LinkFailure(code: "failed", message: String(describing: error)))
      }
    }
  }

  private func dispatch(
    _ request: HTTPRequest, _ writer: ResponseWriter, _ bridge: CompanionBridge,
    path: String, query: [String: String]
  ) async throws {
    let segments = path.split(separator: "/").map(String.init)
    guard segments.first == "link" else {
      notFound(writer, path)
      return
    }
    let rest = Array(segments.dropFirst())
    let method = request.method

    if rest.count >= 2, rest[0] == "jobs" {
      try await jobRoute(
        writer, bridge, id: rest[1], verb: rest.count > 2 ? rest[2] : nil, method: method,
        query: query)
      return
    }

    if rest.count >= 2, rest[0] == "chats" {
      guard let id = UUID(uuidString: rest[1]) else { throw badBody() }
      try await chatRoute(request, writer, bridge, id: id, verb: rest.count > 2 ? rest[2] : nil)
      return
    }

    switch (method, rest.joined(separator: "/")) {
    case ("GET", "chats"):
      respond(writer, bridge.summaries())
    case ("POST", "chats"):
      respond(writer, try bridge.create(decode(request) ?? NewChat()))
    case ("GET", "status"):
      respond(writer, bridge.status())
    case ("GET", "models"):
      respond(writer, bridge.models())
    case ("POST", "models/activate"):
      guard let wanted: SendText = decode(request) else { throw badBody() }
      try bridge.activate(model: wanted.text)
      respond(writer, bridge.models())
    case ("GET", "roots"):
      respond(writer, bridge.roots())
    case ("GET", "files"):
      respond(writer, try bridge.list(query["path"] ?? ""))
    case ("GET", "file"):
      respond(
        writer,
        try bridge.read(
          query["path"] ?? "", offset: Int(query["offset"] ?? "1") ?? 1,
          limit: Int(query["limit"] ?? "400") ?? 400))
    case ("POST", "file"):
      guard let write: FileWrite = decode(request) else { throw badBody() }
      try bridge.write(write)
      respond(writer, Empty())
    case ("POST", "shell"):
      guard let wanted: ShellRequest = decode(request) else { throw badBody() }
      respond(writer, try await bridge.shell(wanted))
    case ("POST", "shell/start"):
      guard let wanted: ShellRequest = decode(request) else { throw badBody() }
      respond(writer, try await bridge.startJob(wanted))
    case ("GET", "jobs"):
      respond(writer, try await bridge.jobs())
    case ("POST", "unpair"):
      if let device = authenticate(request) { forget(device) }
      respond(writer, Empty())
    default:
      notFound(writer, path)
    }
  }

  private func jobRoute(
    _ writer: ResponseWriter, _ bridge: CompanionBridge, id: String, verb: String?,
    method: String, query: [String: String]
  ) async throws {
    switch (method, verb) {
    case ("GET", "output"):
      respond(
        writer,
        try await bridge.jobOutput(
          id, wait: Double(query["wait"] ?? "0") ?? 0,
          limit: Int(query["limit"] ?? "8192") ?? 8192))
    case ("POST", "stop"):
      respond(writer, try await bridge.stopJob(id, force: query["force"] == "1"))
    default:
      notFound(writer, "jobs/\(id)")
    }
  }

  private func chatRoute(
    _ request: HTTPRequest, _ writer: ResponseWriter, _ bridge: CompanionBridge,
    id: UUID, verb: String?
  ) async throws {
    switch (request.method, verb) {
    case ("GET", nil):
      respond(writer, try bridge.detail(id))
    case ("DELETE", nil):
      try bridge.delete(id)
      respond(writer, Empty())
    case ("PATCH", nil):
      guard let change: ChatChange = decode(request) else { throw badBody() }
      respond(writer, try bridge.change(id, change))
    case ("POST", "send"):
      guard let text: SendText = decode(request) else { throw badBody() }
      try bridge.send(id, text: text.text)
      respond(writer, try bridge.frame(id))
    case ("POST", "steer"):
      guard let text: SendText = decode(request) else { throw badBody() }
      try bridge.steer(id, text: text.text)
      respond(writer, Empty())
    case ("POST", "stop"):
      try bridge.stop(id)
      respond(writer, Empty())
    case ("GET", "events"):
      _ = try bridge.detail(id)
      stream(id, writer, bridge)
    default:
      notFound(writer, "/link/chats/\(id.uuidString)/\(verb ?? "")")
    }
  }

  private func pair(_ request: HTTPRequest, _ writer: ResponseWriter, _ bridge: CompanionBridge) {
    guard isPairing else {
      writer.sendError(
        status: 403, type: "pairing_closed",
        message: "pairing is not open on this Mac; open it from the companion settings")
      return
    }
    guard let device: DeviceDescriptor = decode(request) else {
      fail(writer, status: 400, LinkFailure(code: "bad_body", message: "no device in the request"))
      return
    }
    let token = CompanionVault.token()
    secrets.tokens[device.id] = token
    CompanionVault.save(secrets)
    settings.devices.removeAll { $0.id == device.id }
    settings.devices.append(
      DeviceDescriptor(
        id: device.id, name: device.name, system: device.system, pairedAt: Date(),
        lastSeen: Date()))
    closePairing()
    append("paired \(device.name)")
    respond(
      writer,
      PairGrant(
        token: token, serverName: settings.serviceName,
        info: bridge.info(pairingOpen: false)))
  }

  private func authenticate(_ request: HTTPRequest) -> DeviceDescriptor? {
    guard let header = request.headers["authorization"],
      header.lowercased().hasPrefix("bearer ")
    else { return nil }
    let token = String(header.dropFirst(7)).trimmingCharacters(in: .whitespaces)
    guard let id = secrets.tokens.first(where: { $0.value == token })?.key else { return nil }
    return settings.devices.first { $0.id == id }
  }

  private func touch(_ device: DeviceDescriptor) {
    guard let index = settings.devices.firstIndex(where: { $0.id == device.id }) else { return }
    var devices = settings.devices
    devices[index].lastSeen = Date()
    settings.devices = devices
  }

  /// A conversation's live stream. Only what changed goes out: a phone merges rows by id, so a
  /// turn writing into one row costs one row per tick rather than the whole transcript.
  private func stream(_ id: UUID, _ writer: ResponseWriter, _ bridge: CompanionBridge) {
    writer.beginEventStream()
    Task { @MainActor in
      var sentRows: [String: String] = [:]
      var lastActivity: LinkActivity?
      var lastStatus: LinkStatus?
      var lastStatusSent = Date.distantPast
      var lastPing = Date()

      if let frame = try? bridge.frame(id) {
        send(writer, frame)
        sentRows = Dictionary(
          (frame.rows ?? []).map { ($0.id, $0.text) }, uniquingKeysWith: { _, last in last })
        lastActivity = frame.activity
        lastStatus = frame.status
      }

      while !writer.isCancelled {
        // A conversation nobody is answering is polled less often: folding a long transcript
        // six times a second to find nothing changed is the Mac's main thread being spent.
        try? await Task.sleep(
          for: .milliseconds(lastActivity?.isRunning == true ? 150 : 750))
        guard !writer.isCancelled, let frame = try? bridge.frame(id) else { break }

        let rows = frame.rows ?? []
        let changed = rows.filter { sentRows[$0.id] != $0.text }
        if !changed.isEmpty {
          for row in changed { sentRows[row.id] = row.text }
          send(
            writer,
            TurnEvent(
              kind: .rows, rows: changed, pendingSteers: frame.pendingSteers,
              summary: frame.summary, failure: frame.failure))
        }
        if frame.activity != lastActivity {
          lastActivity = frame.activity
          send(writer, TurnEvent(kind: .activity, activity: frame.activity))
        }
        if frame.status != lastStatus, -lastStatusSent.timeIntervalSinceNow > 0.5 {
          lastStatus = frame.status
          lastStatusSent = Date()
          send(writer, TurnEvent(kind: .status, status: frame.status))
        }
        if -lastPing.timeIntervalSinceNow > 15 {
          lastPing = Date()
          send(writer, TurnEvent(kind: .ping))
        }
      }
      writer.finish()
    }
  }

  // MARK: - Writing replies

  private func respond(_ writer: ResponseWriter, _ value: some Encodable) {
    guard let data = try? JSONEncoder.link.encode(value),
      let json = try? JSONSerialization.jsonObject(with: data)
    else {
      writer.sendError(status: 500, type: "encode_failed", message: "could not encode the reply")
      return
    }
    writer.send(json: json)
  }

  private func send(_ writer: ResponseWriter, _ event: TurnEvent) {
    guard let data = try? JSONEncoder.link.encode(event),
      let json = try? JSONSerialization.jsonObject(with: data)
    else { return }
    writer.sendEvent(data: json)
  }

  private func fail(_ writer: ResponseWriter, status: Int, _ failure: LinkFailure) {
    guard let data = try? JSONEncoder.link.encode(failure),
      let json = try? JSONSerialization.jsonObject(with: data)
    else {
      writer.sendError(status: status, type: failure.code, message: failure.message)
      return
    }
    writer.send(status: status, json: json)
  }

  private func notFound(_ writer: ResponseWriter, _ path: String) {
    fail(
      writer, status: 404,
      LinkFailure(code: "no_route", message: "the companion has no route for \(path)"))
  }

  private func badBody() -> LinkFailure {
    LinkFailure(code: "bad_body", message: "the request body was not understood")
  }

  private func decode<Body: Decodable>(_ request: HTTPRequest) -> Body? {
    try? JSONDecoder.link.decode(Body.self, from: request.body)
  }

  private static func query(_ text: String) -> [String: String] {
    var pairs: [String: String] = [:]
    for field in text.split(separator: "&") {
      let parts = field.split(separator: "=", maxSplits: 1)
      guard let name = parts.first else { continue }
      let value = parts.count > 1 ? String(parts[1]) : ""
      pairs[String(name)] =
        value.replacingOccurrences(of: "+", with: " ")
        .removingPercentEncoding ?? value
    }
    return pairs
  }

  private func append(_ message: String) {
    log.append(message)
    if log.count > logLimit { log.removeFirst(log.count - logLimit) }
  }
}
