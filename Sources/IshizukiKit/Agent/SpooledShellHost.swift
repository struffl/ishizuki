// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A workspace that lives somewhere else — a VM, a pod, a machine over ssh — presented as the
// shell host the agent tools already expect. A job's whole state is files in the sandbox, so a
// build survives the app being closed and is still there to read when it opens again.

import Foundation

public final class SpooledShellHost: ShellHost {
  public let workspace: URL
  public let transport: any ExecTransport
  /// Where a job's script, output and cursors live. Inside the sandbox, not on this Mac.
  public let spool: String

  public init(
    workspace: URL, transport: any ExecTransport, spool: String = "/tmp/.ishizuki-jobs"
  ) {
    self.workspace = workspace
    self.transport = transport
    self.spool = spool
  }

  public var isAvailable: Bool { transport.isAvailable }

  public var describes: String { transport.describes }

  public func run(
    _ command: String, cwd: URL?, timeout: Double, byteLimit: Int
  ) async throws -> ShellResult {
    try await transport.prepare()
    let directory = try (cwd.map { try resolve($0.path) } ?? workspace).path
    let script = "cd \(quoted(directory)) || exit 127\n" + command
    return try await transport.exec(
      script, stdin: nil, timeout: timeout, byteLimit: byteLimit)
  }

  public func start(_ command: String, cwd: URL?) async throws -> ShellJob {
    try await transport.prepare()
    let directory = try (cwd.map { try resolve($0.path) } ?? workspace).path
    let result = try await transport.exec(
      Scripts.start(spool: spool, cwd: directory, command: command), stdin: nil, timeout: 60,
      byteLimit: 4096)
    guard result.succeeded,
      let id = result.stdout.split(separator: "\n").last.map(String.init), !id.isEmpty
    else {
      throw SandboxError.failed("starting a command", result.stderr)
    }
    let listed = try await jobs()
    return listed.first { $0.id == id }
      ?? ShellJob(
        id: id, command: command, pid: 0, started: Date(), state: .running, seconds: 0,
        pending: 0)
  }

  public func jobs() async throws -> [ShellJob] {
    try await transport.prepare()
    let result = try await transport.exec(
      Scripts.list(spool: spool), stdin: nil, timeout: 30, byteLimit: 256 * 1024)
    guard result.succeeded else { return [] }
    return result.stdout.split(separator: "\n").compactMap { Self.job(from: String($0)) }
  }

  public func read(job id: String, wait: Double, byteLimit: Int) async throws -> ShellJobOutput {
    try await transport.prepare()
    try check(id)
    let result = try await transport.exec(
      Scripts.read(spool: spool, id: id, wait: wait, limit: byteLimit), stdin: nil,
      timeout: wait + 60, byteLimit: 4 * byteLimit + 8192)
    guard !result.stdout.contains("ISHIZUKI-NOJOB") else { throw ShellError.noSuchJob(id) }
    guard result.succeeded else {
      throw SandboxError.failed("reading \(id)", result.stderr)
    }
    return try Self.output(from: result.stdout, id: id)
  }

  public func stop(job id: String, force: Bool) async throws -> ShellJob {
    try await transport.prepare()
    try check(id)
    let result = try await transport.exec(
      Scripts.stop(spool: spool, id: id, force: force), stdin: nil, timeout: 30,
      byteLimit: 4096)
    guard !result.stdout.contains("ISHIZUKI-NOJOB") else { throw ShellError.noSuchJob(id) }
    let listed = try await jobs()
    guard let job = listed.first(where: { $0.id == id }) else { throw ShellError.noSuchJob(id) }
    return job
  }

  public func contents(at url: URL) async throws -> Data? {
    try await transport.prepare()
    let result = try await transport.exec(
      Scripts.readFile(path: url.path), stdin: nil, timeout: 60, byteLimit: 64 * 1024 * 1024)
    guard result.exitCode != 44 else { return nil }
    guard result.succeeded else {
      throw SandboxError.failed("reading \(url.path)", result.stderr)
    }
    return Data(base64Encoded: result.stdout.filter { !$0.isWhitespace })
  }

  public func write(_ data: Data, to url: URL) async throws {
    try await transport.prepare()
    let encoded = Data(data.base64EncodedString().utf8)
    let result = try await transport.exec(
      Scripts.writeFile(path: url.path), stdin: encoded, timeout: 120, byteLimit: 4096)
    guard result.succeeded else {
      throw SandboxError.failed("writing \(url.path)", result.stderr)
    }
  }

  public func locate(_ program: String) async -> String? {
    let found = try? await transport.exec(
      "command -v \(quoted(program)) 2>/dev/null", stdin: nil, timeout: 30, byteLimit: 4096)
    guard let found, found.succeeded else { return nil }
    let path = found.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    return path.isEmpty ? nil : path
  }

  /// Paths are checked as text here: there is no local filesystem to follow a link through,
  /// and a sandbox's workspace is a path in the sandbox.
  public func resolve(_ path: String) throws -> URL {
    let base = workspace.path
    guard !path.hasPrefix("~") else { throw ShellError.outsideWorkspace(path) }
    let joined = path.hasPrefix("/") ? path : base + "/" + path

    var parts: [String] = []
    for piece in joined.split(separator: "/") {
      switch piece {
      case ".": continue
      case "..":
        guard !parts.isEmpty else { throw ShellError.outsideWorkspace(path) }
        parts.removeLast()
      default: parts.append(String(piece))
      }
    }
    let normalized = "/" + parts.joined(separator: "/")
    guard normalized == base || normalized.hasPrefix(base + "/") else {
      throw ShellError.outsideWorkspace(path)
    }
    return URL(filePath: normalized)
  }

  private func check(_ id: String) throws {
    guard id.allSatisfy({ $0.isLetter || $0.isNumber }) else {
      throw ShellError.noSuchJob(id)
    }
  }

  private func quoted(_ value: String) -> String { shellQuoted(value) }

  // MARK: - Reading what the sandbox said

  private static func job(from line: String) -> ShellJob? {
    let fields = line.split(separator: " ", omittingEmptySubsequences: false).map(String.init)
    guard fields.count >= 11, fields[0].hasPrefix("job") else { return nil }
    let pid = Int32(fields[1]) ?? 0
    let started = Double(fields[2]) ?? 0
    let alive = fields[3] == "1"
    let status = Int32(fields[4])
    let outSize = Int(fields[5]) ?? 0
    let errSize = Int(fields[6]) ?? 0
    let outCursor = Int(fields[7]) ?? 0
    let errCursor = Int(fields[8]) ?? 0
    let killed = fields[9] == "1"
    let now = Double(fields[10]) ?? Date().timeIntervalSince1970
    let command = fields.count > 11 ? decode(fields[11]) : ""

    return ShellJob(
      id: fields[0],
      command: command,
      pid: pid,
      started: Date(timeIntervalSince1970: started),
      state: alive ? .running : (killed ? .killed : .exited),
      exitCode: alive ? nil : (status ?? (killed ? -1 : 0)),
      seconds: max(0, now - started),
      pending: max(0, outSize - outCursor) + max(0, errSize - errCursor))
  }

  private static func output(from payload: String, id: String) throws -> ShellJobOutput {
    var state = "done"
    var code: Int32?
    var killed = false
    var started = Date().timeIntervalSince1970
    var now = started
    var pid: Int32 = 0
    var stdout = ""
    var stderr = ""
    var outRemaining = 0
    var errRemaining = 0
    var command = ""

    for line in payload.split(separator: "\n", omittingEmptySubsequences: false) {
      let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
      guard let key = parts.first else { continue }
      let value = parts.count > 1 ? parts[1] : ""
      switch key {
      case "state":
        let fields = value.split(separator: " ").map(String.init)
        state = fields.first ?? "done"
        code = fields.count > 1 ? Int32(fields[1]) : nil
        killed = fields.count > 2 && fields[2] == "1"
        started = fields.count > 3 ? Double(fields[3]) ?? started : started
        now = fields.count > 4 ? Double(fields[4]) ?? now : now
        pid = fields.count > 5 ? Int32(fields[5]) ?? 0 : 0
      case "command": command = decode(value)
      case "out": stdout = decode(value).map { String(decoding: $0, as: UTF8.self) } ?? ""
      case "err": stderr = decode(value).map { String(decoding: $0, as: UTF8.self) } ?? ""
      case "outremain": outRemaining = Int(value) ?? 0
      case "errremain": errRemaining = Int(value) ?? 0
      default: continue
      }
    }

    let running = state == "alive"
    let job = ShellJob(
      id: id,
      command: command,
      pid: pid,
      started: Date(timeIntervalSince1970: started),
      state: running ? .running : (killed ? .killed : .exited),
      exitCode: running ? nil : (code ?? (killed ? -1 : 0)),
      seconds: max(0, now - started),
      pending: outRemaining + errRemaining)
    return ShellJobOutput(
      job: job, stdout: stdout, stderr: stderr, skipped: 0,
      remaining: outRemaining + errRemaining)
  }

  private static func decode(_ base64: String) -> Data? {
    Data(base64Encoded: base64.filter { !$0.isWhitespace })
  }

  private static func decode(_ base64: String) -> String {
    let data: Data? = decode(base64)
    return data.map { String(decoding: $0, as: UTF8.self) } ?? ""
  }
}

/// The shell the sandbox actually runs. Written for the plainest sh there is, because the far
/// side may be a distroless image with busybox and nothing else.
enum Scripts {
  static let delimiter = "ISHIZUKI_SCRIPT_EOF"

  static func start(spool: String, cwd: String, command: String) -> String {
    """
    set -e
    spool=\(shellQuoted(spool))
    mkdir -p "$spool"
    n=1
    while [ -e "$spool/job$n" ]; do n=$((n+1)); done
    dir="$spool/job$n"
    mkdir -p "$dir"
    cat > "$dir/script" <<'\(delimiter)'
    \(command)
    \(delimiter)
    printf '%s' \(shellQuoted(cwd)) > "$dir/cwd"
    date +%s > "$dir/started"
    cat > "$dir/run" <<'ISHIZUKI_RUN_EOF'
    cd "$(cat "$1/cwd")" || exit 127
    sh "$1/script"
    echo $? > "$1/status"
    ISHIZUKI_RUN_EOF
    if command -v setsid >/dev/null 2>&1; then
      setsid sh "$dir/run" "$dir" > "$dir/out" 2> "$dir/err" < /dev/null &
      echo 1 > "$dir/group"
    else
      sh "$dir/run" "$dir" > "$dir/out" 2> "$dir/err" < /dev/null &
      echo 0 > "$dir/group"
    fi
    echo $! > "$dir/pid"
    echo "job$n"
    """
  }

  static func list(spool: String) -> String {
    """
    spool=\(shellQuoted(spool))
    [ -d "$spool" ] || exit 0
    now=$(date +%s)
    for dir in "$spool"/job*; do
      [ -d "$dir" ] || continue
      id=$(basename "$dir")
      pid=$(cat "$dir/pid" 2>/dev/null || echo 0)
      started=$(cat "$dir/started" 2>/dev/null || echo "$now")
      status=$(cat "$dir/status" 2>/dev/null || echo "")
      alive=0
      if [ -z "$status" ] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
      outsz=$(wc -c < "$dir/out" 2>/dev/null | tr -d ' ' || echo 0)
      errsz=$(wc -c < "$dir/err" 2>/dev/null | tr -d ' ' || echo 0)
      outcur=$(cat "$dir/out.cursor" 2>/dev/null || echo 0)
      errcur=$(cat "$dir/err.cursor" 2>/dev/null || echo 0)
      killed=0
      [ -f "$dir/killed" ] && killed=1
      cmd=$(base64 < "$dir/script" 2>/dev/null | tr -d '\\n')
      echo "$id $pid $started $alive ${status:-0} $outsz $errsz $outcur $errcur $killed $now $cmd"
    done
    """
  }

  static func read(spool: String, id: String, wait: Double, limit: Int) -> String {
    """
    dir=\(shellQuoted(spool))/\(shellQuoted(id))
    [ -d "$dir" ] || { echo ISHIZUKI-NOJOB; exit 0; }
    if sleep 0.1 2>/dev/null; then nap="sleep 0.1"; step=1; else nap="sleep 1"; step=10; fi
    waited=0
    want=\(Int((max(0, wait) * 10).rounded()))
    pid=$(cat "$dir/pid" 2>/dev/null || echo 0)
    while [ "$waited" -lt "$want" ]; do
      [ -f "$dir/status" ] && break
      kill -0 "$pid" 2>/dev/null || break
      $nap
      waited=$((waited+step))
    done

    now=$(date +%s)
    started=$(cat "$dir/started" 2>/dev/null || echo "$now")
    status=$(cat "$dir/status" 2>/dev/null || echo "")
    alive=0
    if [ -z "$status" ] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
    killed=0
    [ -f "$dir/killed" ] && killed=1
    if [ "$alive" = "1" ]; then state=alive; else state=done; fi
    echo "state $state ${status:-0} $killed $started $now $pid"
    echo "command $(base64 < "$dir/script" 2>/dev/null | tr -d '\\n')"

    slice() {
      size=$(wc -c < "$1" 2>/dev/null | tr -d ' ')
      [ -z "$size" ] && size=0
      cur=$(cat "$2" 2>/dev/null || echo 0)
      [ "$cur" -gt "$size" ] && cur=0
      avail=$((size-cur))
      take=$avail
      [ "$take" -gt \(limit) ] && take=\(limit)
      if [ "$take" -gt 0 ]; then
        echo "$3 $(tail -c +$((cur+1)) "$1" | head -c "$take" | base64 | tr -d '\\n')"
      else
        echo "$3 "
      fi
      echo "$cur $take" > "$2.next"
      echo $((cur+take)) > "$2"
      echo "$4 $((avail-take))"
    }
    slice "$dir/out" "$dir/out.cursor" out outremain
    slice "$dir/err" "$dir/err.cursor" err errremain

    if [ "$alive" = "0" ]; then
      outleft=$(( $(wc -c < "$dir/out" 2>/dev/null | tr -d ' ') - $(cat "$dir/out.cursor") ))
      errleft=$(( $(wc -c < "$dir/err" 2>/dev/null | tr -d ' ') - $(cat "$dir/err.cursor") ))
      if [ "$outleft" -le 0 ] && [ "$errleft" -le 0 ]; then rm -rf "$dir"; fi
    fi
    """
  }

  static func stop(spool: String, id: String, force: Bool) -> String {
    """
    dir=\(shellQuoted(spool))/\(shellQuoted(id))
    [ -d "$dir" ] || { echo ISHIZUKI-NOJOB; exit 0; }
    touch "$dir/killed"
    pid=$(cat "$dir/pid" 2>/dev/null || echo 0)
    group=$(cat "$dir/group" 2>/dev/null || echo 0)
    sig=\(force ? "KILL" : "TERM")
    if [ "$group" = "1" ]; then
      kill -$sig -"$pid" 2>/dev/null || kill -$sig "$pid" 2>/dev/null || true
    else
      kill -$sig "$pid" 2>/dev/null || true
    fi
    if [ "$sig" = "TERM" ]; then
      (
        sleep 3
        if kill -0 "$pid" 2>/dev/null; then
          if [ "$group" = "1" ]; then kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
          else kill -KILL "$pid" 2>/dev/null; fi
        fi
      ) >/dev/null 2>&1 &
    fi
    echo stopped
    """
  }

  static func readFile(path: String) -> String {
    """
    p=\(shellQuoted(path))
    [ -f "$p" ] || exit 44
    base64 < "$p"
    """
  }

  static func writeFile(path: String) -> String {
    """
    p=\(shellQuoted(path))
    mkdir -p "$(dirname "$p")"
    if base64 -d </dev/null >/dev/null 2>&1; then dec="base64 -d"; else dec="base64 -D"; fi
    $dec > "$p.ishizuki-part"
    mv "$p.ishizuki-part" "$p"
    """
  }
}
