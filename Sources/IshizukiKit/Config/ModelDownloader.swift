// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// Self-healing model fetch from HuggingFace: downloads/repairs a pack against the repo's authoritative file tree.

import CryptoKit
import Foundation

public enum ModelDownloader {
  struct Entry {
    let size: Int
    let sha256: String
  }

  static let essentials = [
    "config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja",
  ]

  /// Fetches a repo into `directory`, or repairs what is already there.
  ///
  /// `only` names the files to take instead of the whole tree, which is what a GGUF repo needs:
  /// those hold every quantization of the same model side by side, and taking the tree would
  /// fetch a dozen copies to use one. A name matches on its full path in the repo or on its
  /// last component, so `--file Qwen3.8-27B-IQ3_S.gguf` finds it wherever the repo filed it.
  public static func ensure(
    directory: URL,
    repo: String,
    revision: String = "main",
    token explicitToken: String? = nil,
    verify: Bool = false,
    only: [String] = [],
    log: @escaping (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
  ) throws {
    let token = resolveToken(explicitToken)

    var manifest: [String: Entry]
    do {
      manifest = try fetchTree(repo: repo, revision: revision, token: token)
    } catch {
      if essentialsPresent(directory, only: only) {
        log("model: offline, using the existing pack at \(directory.path)")
        return
      }
      throw error
    }

    if !only.isEmpty {
      manifest = manifest.filter { path, _ in
        only.contains(path) || only.contains((path as NSString).lastPathComponent)
      }
      let found = Set(manifest.keys.map { ($0 as NSString).lastPathComponent })
      let unknown = only.filter { !found.contains(($0 as NSString).lastPathComponent) }
      guard unknown.isEmpty else {
        throw BonsaiError.missingComponent(
          "\(repo) has no \(unknown.joined(separator: ", ")) at \(revision)")
      }
    }

    var missing: [(name: String, entry: Entry)] = []
    for (name, entry) in manifest {
      let dst = directory.appending(path: destination(for: name, only: only))
      if intact(dst, size: entry.size, sha256: verify ? entry.sha256 : nil) { continue }
      missing.append((name, entry))
    }
    guard !missing.isEmpty else { return }

    missing.sort { $0.entry.size < $1.entry.size }
    let total = missing.reduce(0) { $0 + $1.entry.size }
    log("model: fetching \(missing.count) file(s), \(humanBytes(total)), from \(repo)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    for (name, entry) in missing {
      try fetch(
        name: name, saveAs: destination(for: name, only: only), entry: entry, into: directory,
        repo: repo, revision: revision, token: token, verify: verify)
    }
    log("model: ready at \(directory.path)")
  }

  private static func resolveToken(_ explicit: String?) -> String? {
    if let explicit, !explicit.isEmpty { return explicit }
    let env = ProcessInfo.processInfo.environment
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"] {
      if let value = env[key], !value.isEmpty { return value }
    }
    return nil
  }

  private static func fileURL(_ repo: String, _ revision: String, _ name: String) -> URL {
    URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(name)")!
  }

  private static func fetchTree(repo: String, revision: String, token: String?) throws -> [String:
    Entry]
  {
    let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/\(revision)?recursive=1")!
    let data = try get(url, token: token)
    guard let array = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
      throw BonsaiError.missingComponent("unexpected tree response for \(repo)")
    }
    var entries: [String: Entry] = [:]
    for item in array {
      guard (item["type"] as? String) == "file", let path = item["path"] as? String else {
        continue
      }
      let lfs = item["lfs"] as? [String: Any]
      let size = (lfs?["size"] as? Int) ?? (item["size"] as? Int) ?? 0
      entries[path] = Entry(size: size, sha256: lfs?["oid"] as? String ?? "")
    }
    guard !entries.isEmpty else {
      throw BonsaiError.missingComponent("\(repo) lists no files")
    }
    return entries
  }

  /// A named file is taken out of whatever subdirectory the repo filed it under, so a pull of
  /// one GGUF lands beside the pack directories rather than nested in a stray folder.
  private static func destination(for path: String, only: [String]) -> String {
    only.isEmpty ? path : (path as NSString).lastPathComponent
  }

  private static func essentialsPresent(_ directory: URL, only: [String] = []) -> Bool {
    let wanted = only.isEmpty ? essentials : only.map { ($0 as NSString).lastPathComponent }
    return wanted.allSatisfy { name in
      let size =
        (try? FileManager.default.attributesOfItem(atPath: directory.appending(path: name).path)[
          .size] as? Int) ?? nil
      return (size ?? 0) > 0
    }
  }

  private static func intact(_ url: URL, size: Int, sha256: String?) -> Bool {
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
      let actual = attrs[.size] as? Int, actual == size
    else { return false }
    guard let sha256, !sha256.isEmpty else { return true }
    return (try? sha256Hex(of: url)) == sha256
  }

  private static func fetch(
    name: String, saveAs: String, entry: Entry, into directory: URL, repo: String,
    revision: String, token: String?, verify: Bool
  ) throws {
    let dst = directory.appending(path: saveAs)
    let part = directory.appending(path: saveAs + ".part")
    try FileManager.default.createDirectory(
      at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)

    var offset =
      (try? FileManager.default.attributesOfItem(atPath: part.path)[.size] as? Int) ?? nil ?? 0
    if offset > entry.size {
      try? FileManager.default.removeItem(at: part)
      offset = 0
    }

    if offset < entry.size {
      let url = fileURL(repo, revision, name)
      var attempt = download(
        url: url, part: part, offset: offset, total: entry.size, token: token, label: name)
      if attempt.status == 416, offset > 0 {
        try? FileManager.default.removeItem(at: part)
        attempt = download(
          url: url, part: part, offset: 0, total: entry.size, token: token, label: name)
      }
      if let error = attempt.error { throw error }
    }

    let written =
      (try? FileManager.default.attributesOfItem(atPath: part.path)[.size] as? Int) ?? nil ?? -1
    guard written == entry.size else {
      throw BonsaiError.missingComponent("\(name): expected \(entry.size) bytes, got \(written)")
    }
    if verify, !entry.sha256.isEmpty, try sha256Hex(of: part) != entry.sha256 {
      try? FileManager.default.removeItem(at: part)
      throw BonsaiError.missingComponent("\(name): sha256 mismatch")
    }
    try? FileManager.default.removeItem(at: dst)
    try FileManager.default.moveItem(at: part, to: dst)
  }

  private static func download(
    url: URL, part: URL, offset: Int, total: Int, token: String?, label: String
  ) -> (error: Error?, status: Int) {
    var request = URLRequest(url: url)
    if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
    if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }

    let stream = FileStream(part: part, offset: offset, total: total, label: label)
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = 120
    let session = URLSession(configuration: config, delegate: stream, delegateQueue: nil)
    session.dataTask(with: request).resume()
    stream.done.wait()
    session.finishTasksAndInvalidate()
    return (stream.error, stream.status)
  }

  private static func get(_ url: URL, token: String?) throws -> Data {
    var request = URLRequest(url: url)
    if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
    let done = DispatchSemaphore(value: 0)
    let box = ResponseBox()
    URLSession.shared.dataTask(with: request) { data, response, error in
      if let error {
        box.result = .failure(error)
      } else if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
        box.result = .failure(
          BonsaiError.missingComponent("HTTP \(http.statusCode) for \(url.lastPathComponent)"))
      } else {
        box.result = .success(data ?? Data())
      }
      done.signal()
    }.resume()
    done.wait()
    return try box.result.get()
  }

  private static func sha256Hex(of url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try handle.read(upToCount: 1 << 22), !chunk.isEmpty {
      hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

private func humanBytes(_ count: Int) -> String {
  let units = ["B", "KB", "MB", "GB", "TB"]
  var value = Double(count)
  var unit = 0
  while value >= 1024, unit < units.count - 1 {
    value /= 1024
    unit += 1
  }
  return unit == 0 ? "\(count) B" : String(format: "%.1f %@", value, units[unit])
}

private final class ResponseBox: @unchecked Sendable {
  var result: Result<Data, Error> = .failure(BonsaiError.missingComponent("no response"))
}

private final class FileStream: NSObject, URLSessionDataDelegate, @unchecked Sendable {
  let done = DispatchSemaphore(value: 0)
  var error: Error?
  var status = 0

  private let part: URL
  private let offset: Int
  private let total: Int
  private let label: String
  private var handle: FileHandle?
  private var written = 0
  private var lastReport = Date.distantPast

  init(part: URL, offset: Int, total: Int, label: String) {
    self.part = part
    self.offset = offset
    self.total = total
    self.label = label
  }

  func urlSession(
    _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    let http = response as? HTTPURLResponse
    status = http?.statusCode ?? 0
    guard let http, (200...299).contains(http.statusCode) else {
      error = BonsaiError.missingComponent("\(label): HTTP \(status)")
      completionHandler(.cancel)
      done.signal()
      return
    }
    do {
      if offset > 0, http.statusCode == 206 {
        let handle = try FileHandle(forWritingTo: part)
        try handle.seekToEnd()
        self.handle = handle
        written = offset
      } else {
        FileManager.default.createFile(atPath: part.path, contents: nil)
        handle = try FileHandle(forWritingTo: part)
        written = 0
      }
    } catch {
      self.error = error
      completionHandler(.cancel)
      done.signal()
      return
    }
    completionHandler(.allow)
  }

  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    do {
      try handle?.write(contentsOf: data)
    } catch {
      self.error = error
      dataTask.cancel()
      return
    }
    written += data.count
    if Date().timeIntervalSince(lastReport) > 0.5 {
      report()
      lastReport = Date()
    }
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    if let error, self.error == nil { self.error = error }
    try? handle?.close()
    if self.error == nil {
      report()
      FileHandle.standardError.write(Data("\n".utf8))
    }
    done.signal()
  }

  private func report() {
    let percent = total > 0 ? Int(Double(written) / Double(total) * 100) : 0
    let line = "\r  \(label)  \(humanBytes(written)) / \(humanBytes(total))  \(percent)%   "
    FileHandle.standardError.write(Data(line.utf8))
  }
}
