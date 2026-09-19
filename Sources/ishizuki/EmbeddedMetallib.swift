// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MachO

// MLX resolves its kernels by looking for mlx.metallib beside the executable and nowhere else:
// the lookup is a memoised dladdr on one of its own symbols, with no environment override and no
// way to hand it bytes. Carrying the library in a Mach-O section and laying it down on first run
// is what lets the release ship as a single file.
enum EmbeddedMetallib {
  static func materialize() {
    guard let payload = section(), let directory = binaryDirectory() else { return }
    let target = directory.appending(path: "mlx.metallib")

    if let existing = try? Data(contentsOf: target, options: .mappedIfSafe),
      existing.count == payload.count, existing == payload
    {
      return
    }
    try? payload.write(to: target, options: .atomic)
  }

  private static func section() -> Data? {
    guard let header = _dyld_get_image_header(0) else { return nil }
    var size: UInt = 0
    let bytes = header.withMemoryRebound(to: mach_header_64.self, capacity: 1) {
      getsectiondata($0, "__MLX", "__metallib", &size)
    }
    guard let bytes, size > 0 else { return nil }
    return Data(bytes: bytes, count: Int(size))
  }

  private static func binaryDirectory() -> URL? {
    guard let name = _dyld_get_image_name(0) else { return nil }
    return URL(filePath: String(cString: name))
      .resolvingSymlinksInPath()
      .deletingLastPathComponent()
  }
}
