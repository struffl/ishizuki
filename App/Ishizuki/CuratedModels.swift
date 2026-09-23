// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The packs offered for download, beyond whatever the machine already holds.

import Foundation
import IshizukiKit

struct CuratedModel: Identifiable, Sendable {
  var id: String { repo + (only.first ?? "") }
  let repo: String
  let name: String
  let summary: String
  let bytes: Int
  /// Named files rather than the whole tree, which is what a GGUF repo needs: those carry
  /// every quantization side by side.
  var only: [String] = []

  var directoryName: String { (repo as NSString).lastPathComponent }

  /// The catalog names a pack three ways depending on where it sits, so an offer to download
  /// one already on disk is checked against all of them.
  func isInstalled(in catalog: ModelCatalog) -> Bool {
    var names: Set<String> = [repo, directoryName]
    for file in only where file.hasSuffix(".gguf") {
      names.insert((file as NSString).deletingPathExtension)
    }
    return catalog.entries.contains { names.contains($0.id) }
  }

  static let all: [CuratedModel] = [
    CuratedModel(
      repo: IshizukiPaths.defaultRepo,
      name: "Ternary Bonsai 2 27B",
      summary: "2-bit Hadamard-rotated. Vision, tool calling, the pack ishizuki was built for.",
      bytes: 8_600_000_000),
    CuratedModel(
      repo: "dealignai/Bonsai-2-27B-CRACK-Ternary-JANG",
      name: "Bonsai 2 27B CRACK",
      summary: "The same ternary weights with refusals ablated, repacked as JANG. 6-bit vision.",
      bytes: 8_220_000_000),
    CuratedModel(
      repo: "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF",
      name: "Qwen3.8 27B GSQ-RCO",
      summary: "IQ2_XS with the MTP head, for speculative decoding. BF16 vision projector.",
      bytes: 9_700_000_000,
      only: ["Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf", "mmproj-Qwen3.8-27B-BF16.gguf"]),
  ]
}
