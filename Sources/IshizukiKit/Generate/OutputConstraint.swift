// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation

/// Restricts decoding to a finite set of documents, one token at a time.
///
/// The documents are held as a byte trie, so the decoder's position is a single node index and
/// testing a candidate token is a walk from that node. At each step the allowed tokens are
/// gathered by scanning only the vocabulary buckets whose first byte the trie currently admits.
public final class OutputConstraint: @unchecked Sendable {
  private struct Node {
    var children: [UInt8: Int] = [:]
    var terminal = false
  }

  private var nodes: [Node] = [Node()]
  private var node = 0
  private var emitted: [UInt8] = []

  public let documentCount: Int

  public init(documents: [String]) {
    documentCount = documents.count
    for document in documents {
      var current = 0
      for byte in Array(document.utf8) {
        if let next = nodes[current].children[byte] {
          current = next
        } else {
          nodes.append(Node())
          let next = nodes.count - 1
          nodes[current].children[byte] = next
          current = next
        }
      }
      nodes[current].terminal = true
    }
  }

  /// True when what has been emitted is already a complete document.
  public var isComplete: Bool { nodes[node].terminal }

  /// True when nothing further can be emitted.
  public var isExhausted: Bool { nodes[node].children.isEmpty }

  public var text: String { String(decoding: emitted, as: UTF8.self) }

  /// Token ids that keep the output inside the document set. Empty means only stopping is legal.
  public func allowedTokens(tokenizer: BonsaiTokenizer) -> [Int] {
    var allowed: [Int] = []
    for byte in nodes[node].children.keys {
      for id in tokenizer.tokensByFirstByte[Int(byte)] {
        if walk(tokenizer.tokenBytes(id)) != nil { allowed.append(id) }
      }
    }
    return allowed
  }

  @discardableResult
  public func accept(_ bytes: [UInt8]) -> Bool {
    guard let landed = walk(bytes) else { return false }
    node = landed
    emitted.append(contentsOf: bytes)
    return true
  }

  private func walk(_ bytes: [UInt8]) -> Int? {
    var current = node
    for byte in bytes {
      guard let next = nodes[current].children[byte] else { return nil }
      current = next
    }
    return current
  }
}
