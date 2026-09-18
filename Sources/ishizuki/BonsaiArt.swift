// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Darwin
import Foundation
import IshizukiKit

enum BonsaiArt {
  static let maxWidth = 160
  static let canopyHeight = 11
  static let stoneHeight = 4
  static let blossomRange = 2...4
  static let fullCanopy = 60
  static let attempts = 24

  static let foliage: Set<Character> = ["▒", "♣", "*", "·"]

  static func lines(seed: UInt64? = nil) -> [String] {
    let columns = terminalColumns()
    guard columns >= 26 else { return [] }

    let width = min(maxWidth, columns - 2)
    let height = canopyHeight + stoneHeight
    let base = width / 2
    let spread = max(0, (width - 52) / 10)
    let wanted = Int(Double(width) * 0.62)
    var rng = Rng(seed: seed ?? UInt64(DispatchTime.now().uptimeNanoseconds))
    var best = Canvas(width: width, height: height)

    for _ in 0..<attempts {
      var canvas = Canvas(width: width, height: height)
      let half = stone(on: &canvas, rng: &rng, base: base)
      grow(
        on: &canvas, rng: &rng, x: base, y: height - stoneHeight - 1,
        kind: .trunk, life: canopyHeight + 4, depth: 0, spread: spread)
      roots(on: &canvas, rng: &rng, base: base, half: half)
      if canvas.score > best.score { best = canvas }
      if best.leafCount >= fullCanopy, best.leafSpread >= wanted, best.leafTop <= 2 { break }
    }

    best.scatterBlossoms(count: rng.int(blossomRange), rng: &rng)
    hang(on: &best, rng: &rng)

    return best.render()
  }

  private static let shading: [Character] = ["░", "▒", "▓", "█"]

  private static func stone(on canvas: inout Canvas, rng: inout Rng, base: Int) -> Int {
    let half = min(22, max(7, canvas.width / 5))
    let top = canvas.height - stoneHeight
    let profile = [0.5, 0.82, 1.0, 0.9]

    for row in 0..<stoneHeight {
      let vertical = Double(row) / Double(stoneHeight - 1)
      let span = Int((Double(half) * profile[min(row, profile.count - 1)]).rounded())
      guard span > 0 else { continue }

      for offset in -span...span {
        let horizontal = Double(offset + span) / Double(2 * span)
        let lit = horizontal * 0.7 + vertical * 1.25
        let ceiling = row < 2 ? 1 : shading.count - 1
        let index = min(ceiling, max(0, Int(lit.rounded())))
        let colour: Style.Colour = index < 2 ? .stone : .stoneDeep
        canvas.put(shading[index], x: base + offset, y: top + row, colour)
      }

      if row == 0 {
        for offset in -span...span where rng.chance(0.22) {
          canvas.put("·", x: base + offset, y: top, .leafDeep)
        }
      }
    }
    return half
  }

  private static func roots(on canvas: inout Canvas, rng: inout Rng, base: Int, half: Int) {
    let top = canvas.height - stoneHeight

    for direction in [-1, 1] {
      for strand in 0..<rng.int(2...3) {
        var x = base + direction * rng.int(0...1)
        for row in 0..<stoneHeight {
          let limit = half - (row == stoneHeight - 1 ? 1 : 0)
          x += direction * (strand == 0 ? 1 : rng.int(1...2))
          x = direction < 0 ? max(x, base - limit) : min(x, base + limit)
          let glyph: Character = row == 0 && strand == 0 ? "┃" : (direction < 0 ? "╱" : "╲")
          canvas.put(glyph, x: x, y: top + row, row < 2 ? .bark : .barkLight)
        }
      }
    }

    canvas.put("┃", x: base, y: top, .barkLight)
    canvas.put("┃", x: base, y: top + 1, .barkLight)
    canvas.put(rng.chance(0.5) ? "╲" : "╱", x: base, y: top + 2, .bark)
  }

  private static func hang(on canvas: inout Canvas, rng: inout Rng) {
    let centre = canvas.width / 2
    for drop in [3, 2] {
      var spots: [(x: Int, y: Int)] = []
      for y in 0..<max(0, canvas.height - stoneHeight - drop) {
        for x in 1..<(canvas.width - 1) where canvas.isBranch(x, y) {
          guard (1...drop).allSatisfy({ !canvas.occupied(x, y + $0) }) else { continue }
          spots.append((x, y))
        }
      }

      let outer = spots.filter { abs($0.x - centre) >= 4 }
      let choices = outer.isEmpty ? spots : outer
      guard !choices.isEmpty else { continue }

      let spot = choices[rng.int(0...(choices.count - 1))]
      canvas.put("╽", x: spot.x, y: spot.y + 1, .faint)
      canvas.put("Ω", x: spot.x, y: spot.y + 2, .warn)
      if drop == 3 { canvas.put("╹", x: spot.x, y: spot.y + 3, .blossomDeep) }
      return
    }
  }

  private enum Kind { case trunk, left, right }

  private static func grow(
    on canvas: inout Canvas, rng: inout Rng, x: Int, y: Int, kind: Kind, life: Int, depth: Int,
    spread: Int
  ) {
    guard depth < 5, life > 0 else { return }
    let span = life
    var x = x
    var y = y
    var life = life
    var age = 0
    var idle = 0
    var rise = 0
    var side = rng.chance(0.5)

    while life > 0 {
      life -= 1
      age += 1

      var dx = 0
      var dy = 0
      switch kind {
      case .trunk:
        dy = age < 2 ? 0 : (rng.chance(0.85) ? -1 : 0)
        dx = age < 2 ? rng.int(-1...1) : (rng.chance(0.4) ? rng.int(-1...1) : 0)
        idle += 1
        if life > 2, depth < 4, idle >= 1, rng.chance(0.75) || idle >= 2 {
          idle = 0
          side = !side
          grow(
            on: &canvas, rng: &rng, x: x, y: y,
            kind: side ? .left : .right, life: life + rng.int(2...5) + spread,
            depth: depth + 1, spread: spread)
        }
      case .left, .right:
        let outward = kind == .left ? -1 : 1
        dx = rng.chance(0.86) ? outward : 0
        dy = rise >= 2 ? -1 : (rng.chance(0.28) ? -1 : 0)
        idle += 1
        if life > 3, depth < 3, idle >= 4, rng.chance(0.35) {
          idle = 0
          grow(
            on: &canvas, rng: &rng, x: x, y: y,
            kind: rng.chance(0.3) ? (kind == .left ? .right : .left) : kind,
            life: life / 2 + 2 + spread / 2, depth: depth + 1, spread: spread)
        }
      }

      rise = dy == 0 ? rise + 1 : 0
      x += dx
      y += dy
      guard canvas.holds(x, y), y < canvas.height - stoneHeight else { break }
      canvas.put(glyph(dx: dx, dy: dy, depth: depth), x: x, y: y, bark(depth: depth, y: y))
      if depth == 0, dx == 0, canvas.holds(x + 1, y), !canvas.occupied(x + 1, y) {
        canvas.put("▓", x: x + 1, y: y, .barkDeep)
      }
      if depth > 0, life * 5 < span * 3, rng.chance(0.85) {
        canopy(on: &canvas, rng: &rng, x: x, y: y, reach: rng.int(3...5))
      }
    }

    canopy(on: &canvas, rng: &rng, x: x, y: y, reach: rng.int(3...5))
  }

  private static func canopy(
    on canvas: inout Canvas, rng: inout Rng, x: Int, y: Int, reach: Int
  ) {
    for offsetY in -1...1 {
      for offsetX in -reach...reach {
        let distance = max(abs(offsetY), (abs(offsetX) + 1) / 2)
        let density = [0.94, 0.78, 0.5, 0.26][min(3, distance)]
        guard rng.chance(density) else { continue }

        let lx = x + offsetX
        let ly = y + offsetY
        guard canvas.holds(lx, ly), ly < canvas.height - stoneHeight else { continue }
        guard !canvas.isBranch(lx, ly) else { continue }

        let glyph: Character
        let colour: Style.Colour
        switch distance {
        case 0:
          glyph = "▒"
          colour = .leafDeep
        case 1:
          glyph = "♣"
          colour = offsetY < 0 ? .leaf : .leafDeep
        case 2:
          glyph = "*"
          colour = offsetY < 0 ? .leafBright : .leaf
        default:
          glyph = "·"
          colour = .leafBright
        }
        canvas.put(glyph, x: lx, y: ly, colour)
      }
    }
  }

  private static func glyph(dx: Int, dy: Int, depth: Int) -> Character {
    let heavy = depth == 0
    if dy == 0 && dx != 0 { return heavy ? "━" : "─" }
    if dx < 0 { return "╲" }
    if dx > 0 { return "╱" }
    return heavy ? "┃" : "│"
  }

  private static func bark(depth: Int, y: Int) -> Style.Colour {
    if depth == 0 { return .bark }
    return depth == 1 ? .barkLight : .bark
  }
}

private struct Canvas {
  struct Cell {
    var glyph: Character
    var colour: Style.Colour
  }

  let width: Int
  let height: Int
  private var cells: [Cell?]

  init(width: Int, height: Int) {
    self.width = width
    self.height = height
    self.cells = Array(repeating: nil, count: width * height)
  }

  func holds(_ x: Int, _ y: Int) -> Bool {
    x >= 0 && x < width && y >= 0 && y < height
  }

  func occupied(_ x: Int, _ y: Int) -> Bool {
    holds(x, y) && cells[y * width + x] != nil
  }

  func isBranch(_ x: Int, _ y: Int) -> Bool {
    guard holds(x, y), let glyph = cells[y * width + x]?.glyph else { return false }
    return "┃│╱╲━─▓".contains(glyph)
  }

  var leafCount: Int { leaves.count }

  var leafSpread: Int { Set(leaves.map { $0 % width }).count }

  var leafTop: Int { leaves.map { $0 / width }.min() ?? height }

  var score: Int { leafCount + 3 * leafSpread - 2 * leafTop }

  private var leaves: [Int] {
    cells.indices.filter { index in
      guard let glyph = cells[index]?.glyph else { return false }
      return BonsaiArt.foliage.contains(glyph)
    }
  }

  mutating func put(_ glyph: Character, x: Int, y: Int, _ colour: Style.Colour) {
    guard holds(x, y) else { return }
    cells[y * width + x] = Cell(glyph: glyph, colour: colour)
  }

  mutating func scatterBlossoms(count: Int, rng: inout Rng) {
    let petals: [Style.Colour] = [.blossom, .blossomDeep, .blossomPlum]
    let buds = cells.indices.filter { cells[$0]?.glyph == "♣" || cells[$0]?.glyph == "*" }
    guard !buds.isEmpty else { return }

    var placed = 0
    var tries = 0
    while placed < count, tries < 60 {
      tries += 1
      let index = buds[rng.int(0...(buds.count - 1))]
      guard cells[index]?.glyph != "❀" else { continue }
      cells[index] = Cell(glyph: "❀", colour: petals[rng.int(0...(petals.count - 1))])
      placed += 1
    }
  }

  func render() -> [String] {
    var lines: [String] = []
    for y in 0..<height {
      var line = ""
      var painted = false
      for x in 0..<width {
        guard let cell = cells[y * width + x] else {
          line += " "
          continue
        }
        line += Style.paint(String(cell.glyph), cell.colour)
        painted = true
      }
      if !painted && lines.isEmpty { continue }
      lines.append(line)
    }
    return lines
  }
}

private struct Rng {
  var seed: UInt64

  mutating func next() -> UInt64 {
    seed &+= 0x9E37_79B9_7F4A_7C15
    var z = seed
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }

  mutating func int(_ range: ClosedRange<Int>) -> Int {
    let span = UInt64(range.upperBound - range.lowerBound + 1)
    return range.lowerBound + Int(next() % span)
  }

  mutating func chance(_ probability: Double) -> Bool {
    Double(next() % 1000) / 1000 < probability
  }
}
