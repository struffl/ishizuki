import CryptoKit
import Foundation

/// Disposable, device-local snapshots. Each paired Mac has a separate bounded cache.
public actor LinkCache {
  private let directory: URL
  private let byteLimit: Int
  private var cleared = false

  public init(serverID: String, root: URL? = nil, byteLimit: Int = 32 * 1024 * 1024) {
    let base =
      root
      ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
      .appending(path: "IshizukiLink")
    directory = base.appending(path: Self.key(serverID), directoryHint: .isDirectory)
    self.byteLimit = byteLimit
  }

  public func load<Value: Decodable & Sendable>(_ type: Value.Type, key: String) -> Value? {
    guard !cleared else { return nil }
    let url = file(key)
    guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
      size <= byteLimit, let data = try? Data(contentsOf: url)
    else { return nil }
    return try? JSONDecoder.link.decode(type, from: data)
  }

  public func save<Value: Encodable & Sendable>(_ value: Value, key: String) {
    guard !cleared, let data = try? JSONEncoder.link.encode(value), data.count <= byteLimit else {
      return
    }
    do {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      #if os(iOS)
        try data.write(
          to: file(key), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
      #else
        try data.write(to: file(key), options: .atomic)
      #endif
      let files = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])
      let entries = files.compactMap { url -> (URL, Int, Date)? in
        guard
          let values = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        else { return nil }
        return (url, values.fileSize ?? 0, values.contentModificationDate ?? .distantPast)
      }.sorted { $0.2 < $1.2 }
      var total = entries.reduce(0) { $0 + $1.1 }
      for entry in entries where total > byteLimit {
        try FileManager.default.removeItem(at: entry.0)
        total -= entry.1
      }
    } catch { /* A cache failure must not prevent a network request. */  }
  }

  public func remove(_ key: String) { try? FileManager.default.removeItem(at: file(key)) }
  public func clear() {
    cleared = true
    try? FileManager.default.removeItem(at: directory)
  }
  private func file(_ key: String) -> URL { directory.appending(path: Self.key(key) + ".json") }
  private static func key(_ text: String) -> String {
    SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
  }
}
