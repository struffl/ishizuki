// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// A conversion of a 360 GB source that fails an hour in should not start again from nothing,
/// and a streamed pack whose layers got different widths has to load at each layer's own.
@Suite("Conversion resume")
struct ConversionResumeTests {
  private var fixture: URL {
    URL(filePath: #filePath).deletingLastPathComponent()
      .appending(path: "Fixtures/qwen4-exp")
  }

  private func scratch() throws -> URL {
    let url = URL(filePath: NSTemporaryDirectory()).appending(path: "resume-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  @Test("a finished table is reused, and a damaged one is written again")
  func reusesTheTable() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }
    let source = try SourceCheckpoint(directory: fixture)

    var notes: [String] = []
    let first = try #require(try EngramRepack.run(source: source, destination: scratch))
    let second = try #require(
      try EngramRepack.run(source: source, destination: scratch) { notes.append($0) })
    #expect(second.layout == first.layout)
    #expect(second.byteCount == first.byteCount)
    #expect(notes.contains { $0.contains("reusing") })

    let part = scratch.appending(path: EngramLayout.fileName(part: 0))
    try Data().write(to: part)
    notes = []
    _ = try EngramRepack.run(source: source, destination: scratch) { notes.append($0) }
    #expect(!notes.contains { $0.contains("reusing") })
    #expect(try Data(contentsOf: part).count > 0)
  }

  @Test("a survey is kept for the same settings and ignored for others")
  func keepsTheSurvey() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }
    let url = scratch.appending(path: SurveyRecord.file)
    let key = SurveyRecord.Key(groupSize: 64, widths: [4, 5, 6], calibrated: false)
    let measured = [
      ModuleMeasurement(path: "a.weight", elements: 10, errorAt: [4: 0.1, 5: 0.05, 6: 0.02]),
      ModuleMeasurement(path: "b.weight", elements: 20, errorAt: [4: 0.2, 5: 0.1, 6: 0.05]),
    ]
    try SurveyRecord.save(measured, key: key, to: url)

    #expect(SurveyRecord.load(url, key: key, covering: ["a.weight", "b.weight", "c"]) == measured)
    var other = key
    other.groupSize = 32
    #expect(SurveyRecord.load(url, key: other, covering: ["a.weight", "b.weight"]) == nil)
    #expect(SurveyRecord.load(url, key: key, covering: ["a.weight"]) == nil)
  }

  @Test("a streamed layer with a layout of its own is read at its own widths")
  func readsEachLayersLayout() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }

    let profile = QuantProfile(
      name: "test", baseBits: 8, boostBits: [], targetBpw: 8, groupSize: 32, summary: "")
    _ = try Quantizer(
      source: try SourceCheckpoint(directory: fixture), profile: profile, destination: scratch,
      streamExperts: true
    ).run()
    #expect(!FileManager.default.fileExists(atPath: scratch.appending(path: SurveyRecord.file).path))

    let shared = try JSONDecoder().decode(
      ExpertLayout.self,
      from: try Data(contentsOf: scratch.appending(path: ExpertRepack.layoutFile)))
    var own = try JSONDecoder().decode(
      ExpertLayout.self,
      from: try Data(contentsOf: scratch.appending(path: ExpertRepack.layerLayoutFile(0))))
    #expect(own == shared)

    own.expertCount += 1
    try ExpertRepack.writeLayout(own, layer: 0, to: scratch)
    let store = try WeightStore(directory: scratch).openingExperts(at: scratch, slots: 4)
    #expect(store.experts(layer: 0)?.layout == own)
    #expect(store.experts(layer: 1)?.layout != own)
  }

  @Test("a streamed pack's slots are sized once, and counted as held")
  func plansTheSlots() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }
    let profile = QuantProfile(
      name: "test", baseBits: 8, boostBits: [], targetBpw: 8, groupSize: 32, summary: "")
    _ = try Quantizer(
      source: try SourceCheckpoint(directory: fixture), profile: profile, destination: scratch,
      streamExperts: true
    ).run()

    let layers = try #require(StreamedPlan.layers(in: scratch))
    #expect(layers.count == 4)
    #expect(layers.expertCount == 4)
    let topK = try BonsaiConfig.load(directory: scratch).textConfig.numExpertsPerTok ?? 0

    #expect(StreamedPlan.slots(for: scratch, requested: 3) == 3)
    #expect(StreamedPlan.slots(for: scratch, requested: 1) == topK)
    #expect(StreamedPlan.slots(for: scratch, requested: 0, ceiling: 0) == topK)
    let roomy = try #require(StreamedPlan.slots(for: scratch, requested: 0, ceiling: 1 << 40))
    #expect(roomy == max(layers.expertCount / StreamedPlan.bankShare / 8 * 8, topK))

    let weights = try #require(MemoryBudget.weightBytes(in: scratch))
    #expect(try #require(StreamedPlan.residentBytes(in: scratch)) > weights)

    let whole = scratch.appending(path: "whole")
    _ = try Quantizer(
      source: try SourceCheckpoint(directory: fixture), profile: profile, destination: whole
    ).run()
    #expect(StreamedPlan.layers(in: whole) == nil)
    #expect(StreamedPlan.residentBytes(in: whole) == MemoryBudget.weightBytes(in: whole))
  }
}
