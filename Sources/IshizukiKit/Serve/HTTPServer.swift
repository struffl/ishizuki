// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Network

public struct HTTPRequest: Sendable {
  public var method: String
  public var path: String
  public var headers: [String: String]
  public var body: Data

  public func json() -> [String: Any]? {
    (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
  }
}

public final class RequestCancellation: @unchecked Sendable {
  private let lock = NSLock()
  private var cancelled = false

  public init() {}

  public var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }

  public func cancel() {
    lock.lock()
    cancelled = true
    lock.unlock()
  }
}

public final class ResponseWriter: @unchecked Sendable {
  private let connection: NWConnection
  private let cancellation: RequestCancellation
  private var headersSent = false
  private var streaming = false
  private var finished = false

  init(connection: NWConnection, cancellation: RequestCancellation) {
    self.connection = connection
    self.cancellation = cancellation
  }

  public var isCancelled: Bool { cancellation.isCancelled }

  /// True once a status line has gone out, so a later failure cannot send a second one.
  public var hasBegun: Bool { headersSent }

  public func send(status: Int = 200, json: Any) {
    let body =
      (try? JSONSerialization.data(withJSONObject: json, options: [.withoutEscapingSlashes]))
      ?? Data()
    sendHead(
      status: status,
      headers: [
        "Content-Type": "application/json",
        "Content-Length": "\(body.count)",
      ])
    write(body)
    finish()
  }

  public func sendError(status: Int, type: String, message: String) {
    send(status: status, json: ["type": "error", "error": ["type": type, "message": message]])
  }

  public func beginEventStream() {
    streaming = true
    sendHead(
      status: 200,
      headers: [
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
      ])
  }

  public func sendEvent(name: String? = nil, data: Any) {
    guard
      let payload = try? JSONSerialization.data(
        withJSONObject: data, options: [.withoutEscapingSlashes]),
      let text = String(data: payload, encoding: .utf8)
    else { return }
    var frame = ""
    if let name { frame += "event: \(name)\n" }
    frame += "data: \(text)\n\n"
    writeChunk(Data(frame.utf8))
  }

  public func sendRaw(_ text: String) {
    writeChunk(Data(text.utf8))
  }

  public func finish() {
    guard !finished else { return }
    finished = true
    if streaming {
      connection.send(
        content: Data("0\r\n\r\n".utf8), completion: .contentProcessed { _ in })
    }
    connection.send(
      content: nil, contentContext: .finalMessage, isComplete: true,
      completion: .contentProcessed { [connection] _ in connection.cancel() })
  }

  private func sendHead(status: Int, headers: [String: String]) {
    guard !headersSent else { return }
    headersSent = true
    var head = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
    for (key, value) in headers { head += "\(key): \(value)\r\n" }
    head += "Access-Control-Allow-Origin: *\r\n\r\n"
    write(Data(head.utf8))
  }

  private func write(_ data: Data) {
    connection.send(
      content: data,
      completion: .contentProcessed { [cancellation] error in
        if error != nil { cancellation.cancel() }
      })
  }

  private func writeChunk(_ data: Data) {
    var framed = Data(String(format: "%llX\r\n", data.count).utf8)
    framed.append(data)
    framed.append(Data("\r\n".utf8))
    write(framed)
  }

  private static func reason(_ status: Int) -> String {
    switch status {
    case 200: "OK"
    case 400: "Bad Request"
    case 404: "Not Found"
    case 500: "Internal Server Error"
    default: "Status"
    }
  }
}

public final class HTTPServer: @unchecked Sendable {
  public typealias Handler = @Sendable (HTTPRequest, ResponseWriter) -> Void

  private let listener: NWListener
  private let handler: Handler
  private let queue = DispatchQueue(label: "bonsai.http", attributes: .concurrent)

  /// A plain TCP listener by default. A key makes every connection TLS with that pre-shared
  /// key, and a service name has the listener answer Bonjour browses for itself.
  public init(
    port: UInt16, psk: PreSharedKey? = nil, advertise: BonjourService? = nil,
    handler: @escaping Handler
  ) throws {
    let parameters: NWParameters
    if let psk {
      parameters = psk.parameters()
    } else {
      parameters = NWParameters.tcp
      parameters.allowLocalEndpointReuse = true
    }
    guard let nwPort = NWEndpoint.Port(rawValue: port) else {
      throw BonsaiError.unsupportedModel("invalid port \(port)")
    }
    self.listener = try NWListener(using: parameters, on: nwPort)
    if let advertise { listener.service = advertise.service }
    self.handler = handler
  }

  public func start() {
    listener.newConnectionHandler = { [weak self] connection in
      guard let self else { return }
      let cancellation = RequestCancellation()
      connection.stateUpdateHandler = { state in
        switch state {
        case .failed, .cancelled: cancellation.cancel()
        default: break
        }
      }
      connection.start(queue: self.queue)
      self.receive(on: connection, buffer: Data(), cancellation: cancellation)
    }
    listener.start(queue: queue)
  }

  public func stop() {
    listener.cancel()
  }

  private func receive(
    on connection: NWConnection, buffer: Data, cancellation: RequestCancellation
  ) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) {
      [weak self] data, _, isComplete, error in
      guard let self else { return }
      var buffer = buffer
      if let data { buffer.append(data) }

      if let request = Self.parse(buffer) {
        self.handler(
          request, ResponseWriter(connection: connection, cancellation: cancellation))
        self.watch(connection, cancellation)
        return
      }
      if error != nil || isComplete {
        cancellation.cancel()
        connection.cancel()
        return
      }
      self.receive(on: connection, buffer: buffer, cancellation: cancellation)
    }
  }

  private func watch(_ connection: NWConnection, _ cancellation: RequestCancellation) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) {
      [weak self] _, _, isComplete, error in
      if isComplete || error != nil {
        cancellation.cancel()
        return
      }
      self?.watch(connection, cancellation)
    }
  }

  private static func parse(_ buffer: Data) -> HTTPRequest? {
    let separator = Data("\r\n\r\n".utf8)
    guard let headerEnd = buffer.range(of: separator),
      let headerText = String(data: buffer[..<headerEnd.lowerBound], encoding: .utf8)
    else { return nil }

    var lines = headerText.components(separatedBy: "\r\n")
    guard !lines.isEmpty else { return nil }
    let requestLine = lines.removeFirst().components(separatedBy: " ")
    guard requestLine.count >= 2 else { return nil }

    var headers: [String: String] = [:]
    for line in lines {
      guard let colon = line.firstIndex(of: ":") else { continue }
      let key = String(line[line.startIndex..<colon]).lowercased()
      let value = String(line[line.index(after: colon)...])
        .trimmingCharacters(in: .whitespaces)
      headers[key] = value
    }

    let expected = Int(headers["content-length"] ?? "0") ?? 0
    let body = buffer[headerEnd.upperBound...]
    guard body.count >= expected else { return nil }

    return HTTPRequest(
      method: requestLine[0], path: requestLine[1], headers: headers,
      body: Data(body.prefix(expected)))
  }
}
