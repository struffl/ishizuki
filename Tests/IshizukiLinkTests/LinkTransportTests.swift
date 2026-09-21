// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import IshizukiKit
import Network
import Testing

@testable import IshizukiLink

@Suite("Link transport", .serialized)
struct LinkTransportTests {
  private struct Greeting: Codable, Equatable {
    var hello: String
    var count: Int
  }

  /// A listener with the same key on both ends, answering one JSON route and one event stream.
  private func listener(port: UInt16, psk: PreSharedKey) throws -> HTTPServer {
    let server = try HTTPServer(port: port, psk: psk) { request, writer in
      switch request.path {
      case "/greeting":
        writer.send(json: ["hello": "phone", "count": 2])
      case "/echo":
        let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]
        writer.send(json: ["hello": body?["hello"] as? String ?? "", "count": 1])
      case "/events":
        writer.beginEventStream()
        for index in 1...3 {
          writer.sendEvent(data: ["hello": "frame", "count": index])
        }
        writer.finish()
      default:
        writer.sendError(status: 404, type: "not_found", message: "no route")
      }
    }
    server.start()
    return server
  }

  private func port() -> UInt16 { UInt16.random(in: 24000...24999) }

  @Test("a key both ends hold carries a request and its reply")
  func roundTrip() async throws {
    let key = PreSharedKey.generate()
    let port = port()
    let server = try listener(port: port, psk: key)
    defer { server.stop() }

    let transport = LinkTransport(endpoint: LinkEndpoint(host: "127.0.0.1", port: port), psk: key)
    let greeting: Greeting = try await transport.call("GET", "/greeting")
    #expect(greeting == Greeting(hello: "phone", count: 2))
  }

  @Test("a body goes out and comes back")
  func body() async throws {
    let key = PreSharedKey.generate()
    let port = port()
    let server = try listener(port: port, psk: key)
    defer { server.stop() }

    let transport = LinkTransport(endpoint: LinkEndpoint(host: "127.0.0.1", port: port), psk: key)
    let echoed: Greeting = try await transport.call(
      "POST", "/echo", body: Greeting(hello: "mac", count: 0))
    #expect(echoed.hello == "mac")
  }

  @Test("an event stream arrives frame by frame")
  func events() async throws {
    let key = PreSharedKey.generate()
    let port = port()
    let server = try listener(port: port, psk: key)
    defer { server.stop() }

    let transport = LinkTransport(endpoint: LinkEndpoint(host: "127.0.0.1", port: port), psk: key)
    var counts: [Int] = []
    for try await frame in transport.events("/events", as: Greeting.self) {
      counts.append(frame.count)
    }
    #expect(counts == [1, 2, 3])
  }

  @Test("a 404 comes back as a failure rather than as an answer")
  func missingRoute() async throws {
    let key = PreSharedKey.generate()
    let port = port()
    let server = try listener(port: port, psk: key)
    defer { server.stop() }

    let transport = LinkTransport(endpoint: LinkEndpoint(host: "127.0.0.1", port: port), psk: key)
    await #expect(throws: LinkTransportError.self) {
      let _: Greeting = try await transport.call("GET", "/nothing")
    }
  }

  @Test("the wrong key gets nowhere")
  func wrongKey() async throws {
    let key = PreSharedKey.generate()
    let port = port()
    let server = try listener(port: port, psk: key)
    defer { server.stop() }

    let transport = LinkTransport(
      endpoint: LinkEndpoint(host: "127.0.0.1", port: port), psk: PreSharedKey.generate())
    await #expect(throws: (any Error).self) {
      let _: Greeting = try await transport.call("GET", "/greeting")
    }
  }

  @Test("a stalled TLS handshake times out instead of trapping the endpoint race")
  func handshakeTimeout() async throws {
    let listener = try NWListener(using: .tcp, on: .any)
    listener.newConnectionHandler = { connection in
      connection.start(queue: .global())
      DispatchQueue.global().asyncAfter(deadline: .now() + 3) { connection.cancel() }
    }
    listener.start(queue: .global())
    defer { listener.cancel() }
    for _ in 0..<100 where listener.port == nil {
      try await Task.sleep(for: .milliseconds(10))
    }
    let port = try #require(listener.port?.rawValue)
    let connection = LinkConnection(
      endpoint: LinkEndpoint(host: "127.0.0.1", port: port), psk: .generate())
    defer { connection.cancel() }
    let started = Date()
    await #expect(throws: (any Error).self) { try await connection.open(timeout: 0.1) }
    #expect(Date().timeIntervalSince(started) < 2)
  }

  @Test("cancellation before opening does not lose the continuation")
  func cancelledOpen() async {
    let connection = LinkConnection(
      endpoint: LinkEndpoint(host: "127.0.0.1", port: 1), psk: .generate())
    connection.cancel()
    await #expect(throws: (any Error).self) { try await connection.open(timeout: 0.1) }
  }

  @Test("a connected Mac that never replies reaches the request deadline")
  func responseTimeout() async throws {
    let key = PreSharedKey.generate()
    let port = port()
    let server = try HTTPServer(port: port, psk: key) { _, writer in
      DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
        writer.send(json: ["late": true])
      }
    }
    server.start()
    defer { server.stop() }
    let transport = LinkTransport(endpoint: LinkEndpoint(host: "127.0.0.1", port: port), psk: key)
    let started = Date()
    await #expect(throws: LinkTransportError.self) {
      _ = try await transport.perform(method: "GET", path: "/stall", body: nil, timeout: 0.2)
    }
    #expect(Date().timeIntervalSince(started) < 1)
  }

  @Test("a ticket survives being a square on a screen")
  func ticket() throws {
    let key = PreSharedKey.generate()
    let ticket = LinkTicket(
      serverName: "Studio", hosts: ["100.98.1.2", "192.168.1.40"], port: 8129, psk: key.base64,
      service: "Studio")
    let url = try #require(ticket.url)
    let back = try #require(LinkTicket(string: url.absoluteString))
    #expect(back == ticket)
    #expect(back.key?.base64 == key.base64)
  }
}
