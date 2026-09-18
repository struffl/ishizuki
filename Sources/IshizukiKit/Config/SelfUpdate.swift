// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// In-place update from a GitHub release: resolve the tarball, check its digest and Developer ID, swap the files.

import CryptoKit
import Foundation

public enum SelfUpdate {
  public static let defaultRepo = "struffl/ishizuki"

  // Release artifacts are signed with this Developer ID team. An update has to match it, or the
  // team of the binary it replaces when that one carries a signature of its own.
  public static let publisherTeamID = "2WTR6KH74L"

  public static let optOutVariable = "ISHIZUKI_NO_UPDATE_CHECK"

  public struct Version: Comparable, CustomStringConvertible, Sendable {
    public let major: Int, minor: Int, patch: Int

    // Accepts a release tag or a `git describe` stamp: v0.2.0, 0.2.0, v0.2.0-4-gdeadbee-dirty.
    public init?(_ text: String) {
      var trimmed = text.trimmingCharacters(in: .whitespaces)
      if trimmed.hasPrefix("v") { trimmed.removeFirst() }
      let core = trimmed.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)[0]
      let parts = core.split(separator: ".")
      guard let major = parts.first.flatMap({ Int($0) }) else { return nil }
      self.major = major
      self.minor = parts.count > 1 ? Int(parts[1]) ?? 0 : 0
      self.patch = parts.count > 2 ? Int(parts[2]) ?? 0 : 0
    }

    public var description: String { "v\(major).\(minor).\(patch)" }

    public static func < (a: Version, b: Version) -> Bool {
      (a.major, a.minor, a.patch) < (b.major, b.minor, b.patch)
    }
  }

  public struct Release: Sendable {
    public let tag: String
    public let publishedAt: String
    public let assetName: String
    public let assetURL: URL
    public let assetSize: Int
    public let assetSHA256: String?

    public var version: Version? { Version(tag) }
    public var releaseDate: String { String(publishedAt.prefix(10)) }
  }

  // MARK: - Resolving

  public static func release(repo: String = defaultRepo, tag: String? = nil) throws -> Release {
    let path = tag.map { "tags/\($0)" } ?? "latest"
    guard let url = URL(string: "https://api.github.com/repos/\(repo)/releases/\(path)") else {
      throw BonsaiError.missingComponent("bad repository '\(repo)'")
    }
    let data = try get(url, authenticated: true)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let tagName = root["tag_name"] as? String
    else {
      throw BonsaiError.missingComponent("unexpected release response from \(repo)")
    }

    let suffix = "-macos-\(machineArchitecture()).tar.gz"
    let assets = root["assets"] as? [[String: Any]] ?? []
    guard
      let asset = assets.first(where: {
        ($0["name"] as? String)?.hasSuffix(suffix) == true
          && ($0["state"] as? String) != "starter"
      }),
      let name = asset["name"] as? String,
      let href = asset["browser_download_url"] as? String,
      let assetURL = URL(string: href)
    else {
      throw BonsaiError.missingComponent(
        "release \(tagName) has no *\(suffix) asset — install it by hand from "
          + "https://github.com/\(repo)/releases/tag/\(tagName)")
    }

    let digest = (asset["digest"] as? String).flatMap { value -> String? in
      value.hasPrefix("sha256:") ? String(value.dropFirst("sha256:".count)) : nil
    }

    return Release(
      tag: tagName,
      publishedAt: root["published_at"] as? String ?? "",
      assetName: name,
      assetURL: assetURL,
      assetSize: asset["size"] as? Int ?? 0,
      assetSHA256: digest)
  }

  // MARK: - Installing

  // The executable of the running process, symlinks resolved — the ~/.local/bin shim execs the
  // real path, so this is the file to replace whichever way ishizuki was installed.
  public static func runningExecutable() -> URL {
    var capacity = UInt32(4096)
    var buffer = [CChar](repeating: 0, count: Int(capacity))
    if _NSGetExecutablePath(&buffer, &capacity) == 0 {
      let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
      return URL(filePath: String(decoding: bytes, as: UTF8.self)).resolvingSymlinksInPath()
    }
    return URL(filePath: CommandLine.arguments[0]).resolvingSymlinksInPath()
  }

  public static func install(
    _ release: Release,
    replacing executable: URL = runningExecutable(),
    log: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
  ) throws {
    let directory = executable.deletingLastPathComponent()
    guard FileManager.default.isWritableFile(atPath: directory.path) else {
      throw BonsaiError.missingComponent(
        "no write access to \(directory.path) — re-run with sudo, or install the .pkg")
    }

    let staging = try FileManager.default.url(
      for: .itemReplacementDirectory, in: .userDomainMask, appropriateFor: executable, create: true)
    defer { try? FileManager.default.removeItem(at: staging) }

    let archive = staging.appending(path: release.assetName)
    try download(release.assetURL, to: archive, size: release.assetSize, label: release.assetName)

    if let expected = release.assetSHA256 {
      let actual = try sha256Hex(of: archive)
      guard actual == expected else {
        throw BonsaiError.missingComponent(
          "sha256 mismatch for \(release.assetName): expected \(expected), got \(actual)")
      }
      log("sha256 matches the release metadata")
    } else {
      log("release metadata carries no digest — relying on the signature check")
    }

    let unpacked = staging.appending(path: "unpacked")
    try FileManager.default.createDirectory(at: unpacked, withIntermediateDirectories: true)
    let untar = capture("/usr/bin/tar", ["-xzf", archive.path, "-C", unpacked.path])
    guard untar.status == 0 else {
      throw BonsaiError.missingComponent("could not unpack \(release.assetName): \(untar.output)")
    }

    guard let newExecutable = locate("ishizuki", under: unpacked) else {
      throw BonsaiError.missingComponent("\(release.assetName) contains no ishizuki executable")
    }
    guard let newMetallib = locate("mlx.metallib", under: unpacked) else {
      throw BonsaiError.missingComponent("\(release.assetName) contains no mlx.metallib")
    }

    try verify(newExecutable, against: executable, log: log)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o755], ofItemAtPath: newExecutable.path)

    // MLX loads the metallib colocated with the executable before anything in the SwiftPM bundle,
    // so writing it here keeps kernels and binary in step without resealing the signed bundle.
    var applied: [Replacement] = []
    do {
      applied.append(try replace(executable, with: newExecutable))
      applied.append(
        try replace(directory.appending(path: "mlx.metallib"), with: newMetallib))
    } catch {
      for replacement in applied.reversed() { replacement.undo() }
      throw error
    }
    for replacement in applied { replacement.commit() }
    log("replaced ishizuki and mlx.metallib in \(directory.path)")
    record(tag: release.tag)
  }

  private static func verify(_ candidate: URL, against current: URL, log: (String) -> Void) throws {
    let team = teamIdentifier(of: current) ?? publisherTeamID
    let requirement =
      "=anchor apple generic"
      + " and certificate leaf[subject.OU] = \"\(team)\""
      + " and certificate 1[field.1.2.840.113635.100.6.2.6] exists"
      + " and certificate leaf[field.1.2.840.113635.100.6.1.13] exists"
    let result = capture(
      "/usr/bin/codesign", ["--verify", "--strict", "-R", requirement, candidate.path])
    guard result.status == 0 else {
      throw BonsaiError.missingComponent(
        "the downloaded executable is not signed by Developer ID team \(team) — refusing to "
          + "install it (\(result.output.trimmingCharacters(in: .whitespacesAndNewlines)))")
    }
    log("signature is Developer ID team \(team), intact")
  }

  private struct Replacement {
    let destination: URL
    let backup: URL

    func undo() {
      guard FileManager.default.fileExists(atPath: backup.path) else { return }
      try? FileManager.default.removeItem(at: destination)
      try? FileManager.default.moveItem(at: backup, to: destination)
    }

    func commit() { try? FileManager.default.removeItem(at: backup) }
  }

  // Rename the old file aside and rename the new one in: the running process keeps the inode it
  // already opened, and a half-finished update can be rolled back file by file.
  private static func replace(_ destination: URL, with source: URL) throws -> Replacement {
    let backup = destination.deletingLastPathComponent()
      .appending(path: "." + destination.lastPathComponent + ".previous")
    try? FileManager.default.removeItem(at: backup)
    let existed = FileManager.default.fileExists(atPath: destination.path)
    if existed { try FileManager.default.moveItem(at: destination, to: backup) }
    do {
      try FileManager.default.moveItem(at: source, to: destination)
    } catch {
      let replacement = Replacement(destination: destination, backup: backup)
      replacement.undo()
      throw error
    }
    return Replacement(destination: destination, backup: backup)
  }

  private static func locate(_ name: String, under directory: URL) -> URL? {
    let manager = FileManager.default
    let candidates =
      (try? manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
    if let direct = candidates.first(where: { $0.lastPathComponent == name }) { return direct }
    for candidate in candidates {
      let nested = candidate.appending(path: name)
      if manager.fileExists(atPath: nested.path) { return nested }
    }
    return nil
  }

  private static func teamIdentifier(of executable: URL) -> String? {
    let result = capture("/usr/bin/codesign", ["--display", "--verbose=4", executable.path])
    guard result.status == 0 else { return nil }
    for line in result.output.split(separator: "\n") where line.hasPrefix("TeamIdentifier=") {
      let value = String(line.dropFirst("TeamIdentifier=".count))
      return value == "not set" ? nil : value
    }
    return nil
  }

  // MARK: - Update notice

  // A line for the serve/launch header, from the last cached check. Refreshing happens in the
  // background so no command ever waits on GitHub to start.
  public static func notice(repo: String = defaultRepo, current: String = BuildInfo.version)
    -> String?
  {
    guard ProcessInfo.processInfo.environment[optOutVariable] == nil else { return nil }
    refreshInBackground(repo: repo)
    guard let running = Version(current), let cached = cachedTag(), let latest = Version(cached),
      latest > running
    else { return nil }
    return "\(cached) is available — run `ishizuki update`"
  }

  private static let checkInterval: TimeInterval = 86_400

  private static var cacheURL: URL {
    let base =
      FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
      ?? URL(filePath: NSHomeDirectory()).appending(path: "Library/Application Support")
    return base.appending(path: "Ishizuki/update-check.json")
  }

  private static func cachedTag() -> String? {
    guard let data = FileManager.default.contents(atPath: cacheURL.path),
      let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    return root["tag"] as? String
  }

  private static func record(tag: String) {
    let payload: [String: Any] = ["tag": tag, "checkedAt": Date().timeIntervalSince1970]
    guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return }
    try? FileManager.default.createDirectory(
      at: cacheURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try? data.write(to: cacheURL, options: .atomic)
  }

  private static func refreshInBackground(repo: String) {
    let checkedAt =
      (try? FileManager.default.attributesOfItem(atPath: cacheURL.path)[.modificationDate]
        as? Date) ?? nil
    if let checkedAt, Date().timeIntervalSince(checkedAt) < checkInterval { return }
    DispatchQueue.global(qos: .utility).async {
      guard let latest = try? release(repo: repo) else { return }
      record(tag: latest.tag)
    }
  }

  // MARK: - Plumbing

  public static func humanBytes(_ count: Int) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"]
    var value = Double(count)
    var unit = 0
    while value >= 1024, unit < units.count - 1 {
      value /= 1024
      unit += 1
    }
    return unit == 0 ? "\(count) B" : String(format: "%.1f %@", value, units[unit])
  }

  private static func machineArchitecture() -> String {
    var info = utsname()
    guard uname(&info) == 0 else { return "arm64" }
    let machine = withUnsafeBytes(of: &info.machine) { raw in
      String(cString: raw.baseAddress!.assumingMemoryBound(to: CChar.self))
    }
    return machine.isEmpty ? "arm64" : machine
  }

  private static func githubToken() -> String? {
    let environment = ProcessInfo.processInfo.environment
    for key in ["GITHUB_TOKEN", "GH_TOKEN"] {
      if let value = environment[key], !value.isEmpty { return value }
    }
    return nil
  }

  private static func get(_ url: URL, authenticated: Bool) throws -> Data {
    var request = URLRequest(url: url)
    request.timeoutInterval = 30
    request.setValue("ishizuki/\(BuildInfo.version)", forHTTPHeaderField: "User-Agent")
    request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
    request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
    // Only the API call carries a token; the asset download redirects to a storage host that
    // rejects a forwarded Authorization header.
    if authenticated, let token = githubToken() {
      request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    let done = DispatchSemaphore(value: 0)
    let box = ResponseBox()
    URLSession.shared.dataTask(with: request) { data, response, error in
      if let error {
        box.result = .failure(error)
      } else if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
        let detail = http.statusCode == 404 ? " (no such repository or release)" : ""
        box.result = .failure(
          BonsaiError.missingComponent("GitHub returned HTTP \(http.statusCode)\(detail)"))
      } else {
        box.result = .success(data ?? Data())
      }
      done.signal()
    }.resume()
    _ = done.wait(timeout: .now() + 60)
    return try box.result.get()
  }

  private static func download(_ url: URL, to destination: URL, size: Int, label: String) throws {
    var request = URLRequest(url: url)
    request.setValue("ishizuki/\(BuildInfo.version)", forHTTPHeaderField: "User-Agent")

    let sink = DownloadSink(destination: destination, total: size, label: label)
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = 120
    let session = URLSession(configuration: config, delegate: sink, delegateQueue: nil)
    session.downloadTask(with: request).resume()
    sink.done.wait()
    session.finishTasksAndInvalidate()
    if let error = sink.error { throw error }
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

  private static func capture(_ tool: String, _ arguments: [String]) -> (
    status: Int32, output: String
  ) {
    let process = Process()
    process.executableURL = URL(filePath: tool)
    process.arguments = arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    do {
      try process.run()
    } catch {
      return (-1, "\(tool): \(error)")
    }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
  }
}

private final class ResponseBox: @unchecked Sendable {
  var result: Result<Data, Error> = .failure(BonsaiError.missingComponent("no response"))
}

private final class DownloadSink: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
  let done = DispatchSemaphore(value: 0)
  var error: Error?

  private let destination: URL
  private let total: Int
  private let label: String
  private var lastReport = Date.distantPast

  init(destination: URL, total: Int, label: String) {
    self.destination = destination
    self.total = total
    self.label = label
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64,
    totalBytesWritten: Int64, totalBytesExpectedToWrite _: Int64
  ) {
    guard Date().timeIntervalSince(lastReport) > 0.25 else { return }
    lastReport = Date()
    report(Int(totalBytesWritten))
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
    guard (200...299).contains(status) else {
      error = BonsaiError.missingComponent("\(label): HTTP \(status)")
      return
    }
    do {
      try? FileManager.default.removeItem(at: destination)
      try FileManager.default.moveItem(at: location, to: destination)
      report(total)
      FileHandle.standardError.write(Data("\n".utf8))
    } catch {
      self.error = error
    }
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    if let error, self.error == nil { self.error = error }
    done.signal()
  }

  private func report(_ written: Int) {
    let percent = total > 0 ? Int(Double(written) / Double(total) * 100) : 0
    let line =
      "\r  \(label)  \(SelfUpdate.humanBytes(written)) / \(SelfUpdate.humanBytes(total))  \(percent)%   "
    FileHandle.standardError.write(Data(line.utf8))
  }
}
