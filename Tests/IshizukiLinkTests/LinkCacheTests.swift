import Foundation
import Testing

@testable import IshizukiLink

@Suite("Companion cache")
struct LinkCacheTests {
  @Test func persistsIsolatesAndClears() async throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let first = LinkCache(serverID: "first", root: root)
    await first.save(["saved draft"], key: "../chat")
    let reopened = LinkCache(serverID: "first", root: root)
    #expect(await reopened.load([String].self, key: "../chat") == ["saved draft"])
    let other = LinkCache(serverID: "second", root: root)
    #expect(await other.load([String].self, key: "../chat") == nil)
    await first.clear()
    await first.save(["late response"], key: "../chat")
    #expect(await reopened.load([String].self, key: "../chat") == nil)
  }

  @Test func boundedStorage() async throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let cache = LinkCache(serverID: "mac", root: root, byteLimit: 100)
    await cache.save(String(repeating: "a", count: 60), key: "one")
    await cache.save(String(repeating: "b", count: 60), key: "two")
    #expect(await cache.load(String.self, key: "one") == nil)
    #expect(await cache.load(String.self, key: "two") != nil)
    await cache.save(String(repeating: "c", count: 200), key: "huge")
    #expect(await cache.load(String.self, key: "huge") == nil)
  }
}
