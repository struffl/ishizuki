// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

public final class BonsaiTokenizer: @unchecked Sendable {
  public let vocabulary: [String: Int]
  public let reverseVocabulary: [Int: String]
  private let mergeRanks: [String: Int]
  private let splitPattern: NSRegularExpression
  private let addedTokenPattern: NSRegularExpression?
  private let addedTokenIds: [String: Int]

  private let byteToUnicode: [UInt8: Character]
  private let unicodeToByte: [Character: UInt8]

  public let eosTokenIds: Set<Int>
  public let imageTokenId: Int?
  public let visionStartTokenId: Int?
  public let visionEndTokenId: Int?

  private var cache: [String: [Int]] = [:]
  private let cacheLock = NSLock()

  public init(directory: URL, config: BonsaiConfig? = nil) throws {
    let data = try Data(contentsOf: directory.appending(path: "tokenizer.json"))
    guard
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let model = root["model"] as? [String: Any],
      let vocab = model["vocab"] as? [String: Int]
    else {
      throw BonsaiError.unsupportedModel("tokenizer.json is missing model.vocab")
    }

    var vocabulary = vocab
    var addedIds: [String: Int] = [:]
    if let added = root["added_tokens"] as? [[String: Any]] {
      for entry in added {
        guard let content = entry["content"] as? String, let id = entry["id"] as? Int
        else { continue }
        addedIds[content] = id
        vocabulary[content] = id
      }
    }
    self.vocabulary = vocabulary
    self.addedTokenIds = addedIds
    self.reverseVocabulary = Dictionary(
      vocabulary.map { ($0.value, $0.key) }, uniquingKeysWith: { a, _ in a })

    var ranks: [String: Int] = [:]
    if let merges = model["merges"] as? [String] {
      ranks.reserveCapacity(merges.count)
      for (rank, merge) in merges.enumerated() {
        guard let space = merge.firstIndex(of: " ") else { continue }
        let left = String(merge[merge.startIndex..<space])
        let right = String(merge[merge.index(after: space)...])
        ranks[left + "\u{0}" + right] = rank
      }
    } else if let merges = model["merges"] as? [[String]] {
      for (rank, pair) in merges.enumerated() where pair.count == 2 {
        ranks[pair[0] + "\u{0}" + pair[1]] = rank
      }
    }
    self.mergeRanks = ranks

    var pattern =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
      + "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    if let pre = root["pre_tokenizer"] as? [String: Any] {
      let candidates: [[String: Any]]
      if let sequence = pre["pretokenizers"] as? [[String: Any]] {
        candidates = sequence
      } else {
        candidates = [pre]
      }
      for entry in candidates where entry["type"] as? String == "Split" {
        if let p = entry["pattern"] as? [String: Any],
          let regex = p["Regex"] as? String
        {
          pattern = regex
        }
      }
    }
    self.splitPattern = try NSRegularExpression(pattern: pattern)

    if addedIds.isEmpty {
      self.addedTokenPattern = nil
    } else {
      let alternatives =
        addedIds.keys
        .sorted { $0.count > $1.count }
        .map { NSRegularExpression.escapedPattern(for: $0) }
        .joined(separator: "|")
      self.addedTokenPattern = try NSRegularExpression(pattern: alternatives)
    }

    let (toUnicode, toByte) = Self.byteLevelMaps()
    self.byteToUnicode = toUnicode
    self.unicodeToByte = toByte

    var eos = Set<Int>()
    if let id = addedIds["<|im_end|>"] { eos.insert(id) }
    if let id = addedIds["<|endoftext|>"] { eos.insert(id) }
    if let id = config?.textConfig.eosTokenId { eos.insert(id) }
    self.eosTokenIds = eos

    self.imageTokenId = config?.imageTokenId ?? addedIds["<|image_pad|>"]
    self.visionStartTokenId = config?.visionStartTokenId ?? addedIds["<|vision_start|>"]
    self.visionEndTokenId = config?.visionEndTokenId ?? addedIds["<|vision_end|>"]
  }

  private static func byteLevelMaps() -> ([UInt8: Character], [Character: UInt8]) {
    var bytes: [Int] = []
    bytes.append(contentsOf: Int(UInt8(ascii: "!"))...Int(UInt8(ascii: "~")))
    bytes.append(contentsOf: 0xA1...0xAC)
    bytes.append(contentsOf: 0xAE...0xFF)

    var codes = bytes
    var next = 0
    for b in 0...255 where !bytes.contains(b) {
      bytes.append(b)
      codes.append(256 + next)
      next += 1
    }

    var toUnicode: [UInt8: Character] = [:]
    var toByte: [Character: UInt8] = [:]
    for (b, c) in zip(bytes, codes) {
      let character = Character(UnicodeScalar(c)!)
      toUnicode[UInt8(b)] = character
      toByte[character] = UInt8(b)
    }
    return (toUnicode, toByte)
  }

  public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
    guard addSpecialTokens, let addedTokenPattern else {
      return encodeOrdinary(text)
    }

    var ids: [Int] = []
    let ns = text as NSString
    var cursor = 0
    addedTokenPattern.enumerateMatches(
      in: text, range: NSRange(location: 0, length: ns.length)
    ) { match, _, _ in
      guard let match else { return }
      if match.range.location > cursor {
        let chunk = ns.substring(
          with: NSRange(location: cursor, length: match.range.location - cursor))
        ids.append(contentsOf: encodeOrdinary(chunk))
      }
      let literal = ns.substring(with: match.range)
      if let id = addedTokenIds[literal] { ids.append(id) }
      cursor = match.range.location + match.range.length
    }
    if cursor < ns.length {
      let chunk = ns.substring(
        with: NSRange(location: cursor, length: ns.length - cursor))
      ids.append(contentsOf: encodeOrdinary(chunk))
    }
    return ids
  }

  private func encodeOrdinary(_ text: String) -> [Int] {
    guard !text.isEmpty else { return [] }
    let normalized = text.precomposedStringWithCanonicalMapping
    let ns = normalized as NSString

    var ids: [Int] = []
    splitPattern.enumerateMatches(
      in: normalized, range: NSRange(location: 0, length: ns.length)
    ) { match, _, _ in
      guard let match, match.range.length > 0 else { return }
      let piece = ns.substring(with: match.range)
      ids.append(contentsOf: encodePiece(piece))
    }
    return ids
  }

  private func encodePiece(_ piece: String) -> [Int] {
    cacheLock.lock()
    if let cached = cache[piece] {
      cacheLock.unlock()
      return cached
    }
    cacheLock.unlock()

    var symbols: [String] = []
    symbols.reserveCapacity(piece.utf8.count)
    for byte in Array(piece.utf8) {
      if let character = byteToUnicode[byte] {
        symbols.append(String(character))
      }
    }
    guard !symbols.isEmpty else { return [] }

    symbols = applyBPE(symbols)

    var ids: [Int] = []
    ids.reserveCapacity(symbols.count)
    for symbol in symbols {
      if let id = vocabulary[symbol] {
        ids.append(id)
      }
    }

    cacheLock.lock()
    cache[piece] = ids
    cacheLock.unlock()
    return ids
  }

  private func applyBPE(_ initial: [String]) -> [String] {
    var symbols = initial
    guard symbols.count > 1 else { return symbols }

    while true {
      var bestRank = Int.max
      var bestIndex = -1
      for i in 0..<(symbols.count - 1) {
        if let rank = mergeRanks[symbols[i] + "\u{0}" + symbols[i + 1]], rank < bestRank {
          bestRank = rank
          bestIndex = i
        }
      }
      guard bestIndex >= 0 else { break }
      symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1]
      symbols.remove(at: bestIndex + 1)
      if symbols.count == 1 { break }
    }
    return symbols
  }

  public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
    var text = ""
    for id in ids {
      guard let token = reverseVocabulary[id] else { continue }
      if skipSpecialTokens && addedTokenIds[token] != nil { continue }
      text += token
    }
    return bytesFromByteLevel(text)
  }

  public func tokenString(_ id: Int) -> String? { reverseVocabulary[id] }

  public func isAddedToken(_ id: Int) -> Bool {
    guard let token = reverseVocabulary[id] else { return false }
    return addedTokenIds[token] != nil
  }

  private func bytesFromByteLevel(_ text: String) -> String {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(text.count)
    for character in text {
      if let byte = unicodeToByte[character] {
        bytes.append(byte)
      } else {
        bytes.append(contentsOf: Array(String(character).utf8))
      }
    }
    return String(decoding: bytes, as: UTF8.self)
  }
}

public struct StreamingDetokenizer {
  private let tokenizer: BonsaiTokenizer
  private var ids: [Int] = []
  private var emittedCharacters = 0

  public init(tokenizer: BonsaiTokenizer) {
    self.tokenizer = tokenizer
  }

  public mutating func append(_ id: Int) -> String {
    ids.append(id)
    let full = tokenizer.decode(ids)
    guard !full.contains("\u{FFFD}") else { return "" }
    let characters = Array(full)
    guard characters.count > emittedCharacters else { return "" }
    let new = String(characters[emittedCharacters...])
    emittedCharacters = characters.count
    return new
  }
}
