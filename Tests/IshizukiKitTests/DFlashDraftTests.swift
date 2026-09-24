// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Finding a DFlash drafter for a backbone.

import Foundation
import Testing

@testable import IshizukiKit

@Suite("DFlash draft")
struct DFlashDraftTests {
  private func drafter(
    _ directory: URL, layers: Int = 64, hidden: Int = 5120, second: Bool = true
  ) throws {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let config: [String: Any] = [
      "architectures": [second ? "DFlash2DraftModel" : "DFlashDraftModel"],
      "num_target_layers": layers, "hidden_size": hidden, "vocab_size": 248320,
    ]
    try JSONSerialization.data(withJSONObject: config)
      .write(to: directory.appending(path: "config.json"))
  }

  @Test("finds a drafter beside the pack by the backbone's shape, DFlash 2 first")
  func findsByShape() throws {
    let library = URL(filePath: NSTemporaryDirectory()).appending(path: "dflash-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: library) }
    let pack = library.appending(path: "Bonsai")
    try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
    try drafter(library.appending(path: "Other-DFlash2"), layers: 40)
    try drafter(library.appending(path: "Qwen-DFlash"), second: false)
    #expect(
      DFlashDraft.find(layers: 64, hidden: 5120, vocab: 248320, beside: pack, hub: library)?.lastPathComponent
        == "Qwen-DFlash")
    try drafter(library.appending(path: "Qwen-DFlash2"))
    #expect(
      DFlashDraft.find(layers: 64, hidden: 5120, vocab: 248320, beside: pack, hub: library)?.lastPathComponent
        == "Qwen-DFlash2")
    #expect(DFlashDraft.find(layers: 48, hidden: 5120, vocab: 248320, beside: pack, hub: library) == nil)
  }
}
