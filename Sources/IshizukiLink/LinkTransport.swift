// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One request, one connection, spoken to a listener that answers TLS with a pre-shared key.
// URLSession has no way to present such a key, so the HTTP is written out by hand here.

import Foundation
import IshizukiKit
import Network

/// Where a Mac is, however it came to be known: an address and port typed in or read from a
/// pairing ticket, or a Bonjour name the network resolves at connect time.
public struct LinkEndpoint: Codable, Sendable, Equatable {
  public enum Kind: Codable, Sendable, Equatable {
    case address(String, UInt16)
    case bonjour(name: String, type: String, domain: String)
  }

  public var kind: Kind

  public init(kind: Kind) {
    self.kind = kind
  }

  public init(host: String, port: UInt16 = Link.defaultPort) {
    self.kind = .address(host, port)
  }

  public static func bonjour(
    name: String, type: String = BonjourService.linkType, domain: String = "local."
  ) -> LinkEndpoint {
    LinkEndpoint(kind: .bonjour(name: name, type: type, domain: domain))
  }

  /// What goes in the Host header, which for a Bonjour name is the name itself.
  public var host: String {
    switch kind {
    case .address(let host, _): host
    case .bonjour(let name, _, _): name
    }
  }

  public var port: UInt16? {
    switch kind {
    case .address(_, let port): port
    case .bonjour: nil
    }
  }

  public var description: String {
    switch kind {
    case .address(let host, let port): "\(host):\(port)"
    case .bonjour(let name, _, _): name
    }
  }

  var nwEndpoint: NWEndpoint {
    switch kind {
    case .address(let host, let port):
      .hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port) ?? .any)
    case .bonjour(let name, let type, let domain):
      .service(name: name, type: type, domain: domain, interface: nil)
    }
  }
}

public enum LinkTransportError: Error, LocalizedError, Sendable {
  case unreachable(String)
  case handshakeRefused
  case closed
  case malformedResponse
  case status(Int, String)
  case timedOut

  public var errorDescription: String? {
    switch self {
    case .unreachable(let why): "cannot reach this Mac: \(why)"
    case .handshakeRefused: "this Mac refused the pairing key; pair again"
    case .closed: "the connection closed"
    case .malformedResponse: "the reply was not understood"
    case .status(let code, let message): message.isEmpty ? "HTTP \(code)" : message
    case .timedOut: "the Mac did not answer"
    }
  }
}

final class LinkConnection: @unchecked Sendable {
  private let connection: NWConnection
  private let queue = DispatchQueue(label: "ishizuki.link.connection")
  private let lock = NSLock()
  private var opening: CheckedContinuation<Void, Error>?
  private var settled = false
  private var cancelled = false

  init(endpoint: LinkEndpoint, psk: PreSharedKey) {
    connection = NWConnection(to: endpoint.nwEndpoint, using: psk.parameters())
  }

  func open(timeout: Double = 8) async throws {
    try await withThrowingTaskGroup(of: Void.self) { group in
      group.addTask { try await self.waitForReady() }
      group.addTask {
        try await Task.sleep(for: .seconds(timeout))
        throw LinkTransportError.timedOut
      }
      defer { group.cancelAll() }
      try await group.next()
    }
  }

  private func waitForReady() async throws {
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, Error>) in
        lock.lock()
        if cancelled {
          lock.unlock()
          continuation.resume(throwing: CancellationError())
          return
        }
        opening = continuation
        lock.unlock()
        connection.stateUpdateHandler = { [weak self] state in
          guard let self else { return }
          switch state {
          case .ready:
            self.settle(nil)
          case .waiting(let error):
            if case .tls = error { self.settle(LinkTransportError.handshakeRefused) }
          case .failed(let error):
            self.settle(Self.translate(error))
          case .cancelled:
            self.settle(LinkTransportError.closed)
          default:
            break
          }
        }
        connection.start(queue: queue)
      }
    } onCancel: {
      self.cancel()
    }
  }

  private func settle(_ error: Error?) {
    lock.lock()
    guard !settled, let continuation = opening else {
      lock.unlock()
      return
    }
    settled = true
    opening = nil
    lock.unlock()
    if let error {
      continuation.resume(throwing: error)
    } else {
      continuation.resume()
    }
  }

  private static func translate(_ error: NWError) -> Error {
    switch error {
    case .tls: LinkTransportError.handshakeRefused
    default: LinkTransportError.unreachable(String(describing: error))
    }
  }

  func write(_ data: Data) async throws {
    try await withTaskCancellationHandler {
      try Task.checkCancellation()
      try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, Error>) in
        connection.send(
          content: data,
          completion: .contentProcessed { error in
            if let error {
              continuation.resume(throwing: LinkConnection.translate(error))
            } else {
              continuation.resume()
            }
          })
      }
    } onCancel: {
      self.cancel()
    }
  }

  /// The next stretch of bytes, or nil once the other end is finished.
  func read(timeout: Double? = nil) async throws -> Data? {
    guard let timeout else { return try await receive() }
    return try await withThrowingTaskGroup(of: Data?.self) { group in
      group.addTask { try await self.receive() }
      group.addTask {
        try await Task.sleep(for: .seconds(timeout))
        throw LinkTransportError.timedOut
      }
      defer { group.cancelAll() }
      return try await group.next() ?? nil
    }
  }

  private func receive() async throws -> Data? {
    try await withTaskCancellationHandler {
      try Task.checkCancellation()
      return try await withCheckedThrowingContinuation { continuation in
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 18) {
          data, _, isComplete, error in
          if let error {
            continuation.resume(throwing: LinkConnection.translate(error))
            return
          }
          if let data, !data.isEmpty {
            continuation.resume(returning: data)
            return
          }
          continuation.resume(returning: isComplete ? nil : Data())
        }
      }
    } onCancel: {
      self.cancel()
    }
  }

  func cancel() {
    lock.withLock { cancelled = true }
    settle(CancellationError())
    connection.cancel()
  }
}

struct HTTPResponseHead {
  var status: Int
  var headers: [String: String]
  var bodyStart: Int

  var contentLength: Int? { headers["content-length"].flatMap(Int.init) }
  var isChunked: Bool { headers["transfer-encoding"]?.contains("chunked") == true }

  static func parse(_ buffer: Data) -> HTTPResponseHead? {
    guard let end = buffer.range(of: Data("\r\n\r\n".utf8)),
      let text = String(data: buffer[..<end.lowerBound], encoding: .utf8)
    else { return nil }
    var lines = text.components(separatedBy: "\r\n")
    guard let statusLine = lines.first else { return nil }
    lines.removeFirst()
    let parts = statusLine.components(separatedBy: " ")
    guard parts.count >= 2, let status = Int(parts[1]) else { return nil }
    var headers: [String: String] = [:]
    for line in lines {
      guard let colon = line.firstIndex(of: ":") else { continue }
      headers[String(line[line.startIndex..<colon]).lowercased()] =
        String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
    }
    return HTTPResponseHead(
      status: status, headers: headers,
      bodyStart: buffer.distance(from: buffer.startIndex, to: end.upperBound))
  }
}

/// Undoes chunked transfer encoding as the pieces land, which is what a stream of events arrives
/// wrapped in.
struct ChunkedDecoder {
  private var buffer = Data()
  private var finished = false

  mutating func push(_ data: Data) -> Data {
    buffer.append(data)
    var out = Data()
    while !finished {
      guard let lineEnd = buffer.range(of: Data("\r\n".utf8)),
        let sizeText = String(data: buffer[buffer.startIndex..<lineEnd.lowerBound], encoding: .utf8)
      else { break }
      let size = Int(
        sizeText.components(separatedBy: ";")[0].trimmingCharacters(in: .whitespaces), radix: 16)
      guard let size else { break }
      let bodyStart = lineEnd.upperBound
      guard buffer.distance(from: bodyStart, to: buffer.endIndex) >= size + 2 else { break }
      if size == 0 {
        finished = true
        break
      }
      let bodyEnd = buffer.index(bodyStart, offsetBy: size)
      out.append(buffer[bodyStart..<bodyEnd])
      buffer = Data(buffer[buffer.index(bodyEnd, offsetBy: 2)...])
    }
    return out
  }

  var isFinished: Bool { finished }
}

/// Pulls `data:` payloads out of an event stream, one per frame.
struct EventStreamParser {
  private var buffer = Data()

  mutating func push(_ data: Data) -> [Data] {
    buffer.append(data)
    var frames: [Data] = []
    while let split = buffer.range(of: Data("\n\n".utf8)) {
      let frame = buffer[buffer.startIndex..<split.lowerBound]
      buffer = Data(buffer[split.upperBound...])
      guard let text = String(data: frame, encoding: .utf8) else { continue }
      let payload =
        text
        .components(separatedBy: "\n")
        .filter { $0.hasPrefix("data:") }
        .map { $0.dropFirst(5).trimmingCharacters(in: .whitespaces) }
        .joined(separator: "\n")
      guard !payload.isEmpty else { continue }
      frames.append(Data(payload.utf8))
    }
    return frames
  }
}

/// The plumbing under `LinkClient`: writes a request, reads one reply, and knows how to keep a
/// connection open when the reply is a stream rather than an answer.
public struct LinkTransport: Sendable {
  public var endpoint: LinkEndpoint
  public var psk: PreSharedKey
  public var token: String?
  /// How long a connection is given to come up. Short, because several addresses are usually
  /// being tried at once and most of them are somewhere this phone is not.
  public var connectTimeout: Double

  public init(
    endpoint: LinkEndpoint, psk: PreSharedKey, token: String? = nil, connectTimeout: Double = 6
  ) {
    self.endpoint = endpoint
    self.psk = psk
    self.token = token
    self.connectTimeout = connectTimeout
  }

  private func head(method: String, path: String, bodyLength: Int, stream: Bool) -> Data {
    var request = "\(method) \(path) HTTP/1.1\r\n"
    request += "Host: \(endpoint.host)\r\n"
    request += "Connection: close\r\n"
    request += "Accept: \(stream ? "text/event-stream" : "application/json")\r\n"
    if let token { request += "Authorization: Bearer \(token)\r\n" }
    if bodyLength > 0 {
      request += "Content-Type: application/json\r\n"
      request += "Content-Length: \(bodyLength)\r\n"
    }
    request += "\r\n"
    return Data(request.utf8)
  }

  func perform(method: String, path: String, body: Data?, timeout: Double = 30) async throws -> Data
  {
    try await withThrowingTaskGroup(of: Data.self) { group in
      group.addTask { try await response(method: method, path: path, body: body) }
      group.addTask {
        try await Task.sleep(for: .seconds(timeout))
        throw LinkTransportError.timedOut
      }
      defer { group.cancelAll() }
      guard let result = try await group.next() else { throw LinkTransportError.closed }
      return result
    }
  }

  private func response(method: String, path: String, body: Data?) async throws -> Data {
    let connection = LinkConnection(endpoint: endpoint, psk: psk)
    defer { connection.cancel() }
    try await connection.open(timeout: connectTimeout)

    var request = head(method: method, path: path, bodyLength: body?.count ?? 0, stream: false)
    if let body { request.append(body) }
    try await connection.write(request)

    var buffer = Data()
    var head: HTTPResponseHead?
    var chunked = ChunkedDecoder()
    var payload = Data()

    while true {
      if head == nil, let parsed = HTTPResponseHead.parse(buffer) {
        head = parsed
        let rest = Data(buffer.dropFirst(parsed.bodyStart))
        payload = parsed.isChunked ? chunked.push(rest) : rest
      }
      if let head {
        if let length = head.contentLength, payload.count >= length {
          return try unwrap(head, Data(payload.prefix(length)))
        }
        if head.isChunked, chunked.isFinished {
          return try unwrap(head, payload)
        }
      }
      guard let next = try await connection.read() else {
        if let head { return try unwrap(head, payload) }
        throw LinkTransportError.closed
      }
      if head == nil {
        buffer.append(next)
      } else if head?.isChunked == true {
        payload.append(chunked.push(next))
      } else {
        payload.append(next)
      }
    }
  }

  private func unwrap(_ head: HTTPResponseHead, _ body: Data) throws -> Data {
    guard head.status < 400 else {
      let failure = try? JSONDecoder.link.decode(LinkFailure.self, from: body)
      throw LinkTransportError.status(head.status, failure?.message ?? "")
    }
    return body
  }

  public func call<Output: Decodable & Sendable>(
    _ method: String, _ path: String, body: (some Encodable)? = Optional<Never>.none,
    timeout: Double = 30, as output: Output.Type = Output.self
  ) async throws -> Output {
    let encoded = try body.map { try JSONEncoder.link.encode($0) }
    let data = try await perform(method: method, path: path, body: encoded, timeout: timeout)
    return try JSONDecoder.link.decode(Output.self, from: data)
  }

  public func callVoid(
    _ method: String, _ path: String, body: (some Encodable)? = Optional<Never>.none
  ) async throws {
    let encoded = try body.map { try JSONEncoder.link.encode($0) }
    _ = try await perform(method: method, path: path, body: encoded)
  }

  /// A connection held open for as long as the caller reads from it. Cancelling the task the
  /// stream is consumed in closes the socket, which is what stops the Mac sending.
  public func events<Event: Decodable & Sendable>(
    _ path: String, as event: Event.Type = Event.self
  ) -> AsyncThrowingStream<Event, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        let connection = LinkConnection(endpoint: endpoint, psk: psk)
        do {
          try await connection.open(timeout: connectTimeout)
          try await connection.write(
            head(method: "GET", path: path, bodyLength: 0, stream: true))

          var buffer = Data()
          var head: HTTPResponseHead?
          var chunked = ChunkedDecoder()
          var frames = EventStreamParser()

          while !Task.isCancelled {
            guard let next = try await connection.read(timeout: 45) else { break }
            if head == nil {
              buffer.append(next)
              guard let parsed = HTTPResponseHead.parse(buffer) else { continue }
              head = parsed
              guard parsed.status < 400 else {
                throw LinkTransportError.status(parsed.status, "")
              }
              let rest = Data(buffer.dropFirst(parsed.bodyStart))
              let body = parsed.isChunked ? chunked.push(rest) : rest
              for frame in frames.push(body) {
                continuation.yield(try JSONDecoder.link.decode(Event.self, from: frame))
              }
              continue
            }
            let body = head?.isChunked == true ? chunked.push(next) : next
            for frame in frames.push(body) {
              continuation.yield(try JSONDecoder.link.decode(Event.self, from: frame))
            }
            if chunked.isFinished { break }
          }
          connection.cancel()
          continuation.finish()
        } catch {
          connection.cancel()
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in task.cancel() }
    }
  }
}
