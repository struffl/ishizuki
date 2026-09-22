// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What a quantize run will do, worked out before it starts so it can be shown and confirmed.

import Foundation

public struct QuantizePlan: Sendable {
  public let source: FullPrecisionScan.Candidate
  public let profile: QuantProfile
  public let destination: URL
  public let sourceBytes: Int
  public let estimateBytes: Int
  public let layerCount: Int
  /// What the n-gram table will weigh in the pack, for a model that ships one. It is carried
  /// across at full width rather than quantized, so it is the one part of a checkpoint that
  /// does not shrink — and on the models that have one it is most of the download.
  public let engramBytes: Int

  public enum Problem: Error, CustomStringConvertible {
    case destinationExists(URL)

    public var description: String {
      switch self {
      case .destinationExists(let url):
        "\(url.path) already exists"
      }
    }
  }

  /// A checkout on disk is a revision hash; the repo name without its owner is what to call the
  /// pack, so it sits beside the pulled ones under the same kind of name.
  public static func destinationName(source: FullPrecisionScan.Candidate, profile: QuantProfile)
    -> String
  {
    "\((source.name as NSString).lastPathComponent)-ishizuki-\(profile.name)"
  }

  public init(
    source: FullPrecisionScan.Candidate,
    profile requested: QuantProfile,
    groupSize: Int? = nil,
    destination: URL? = nil,
    within models: URL
  ) throws {
    let profile =
      groupSize.map { size in
        size == requested.groupSize
          ? requested
          : QuantProfile(
            name: requested.name, baseBits: requested.baseBits, boostBits: requested.boostBits,
            targetBpw: requested.targetBpw, groupSize: size, summary: requested.summary)
      } ?? requested

    self.source = source
    self.profile = profile
    self.destination =
      destination
      ?? models.appending(path: Self.destinationName(source: source, profile: profile))
    self.sourceBytes = MemoryBudget.weightBytes(in: source.directory) ?? 0

    let checkpoint = try SourceCheckpoint(directory: source.directory)
    self.layerCount = checkpoint.layerCount

    // An n-gram table is rows a token looks up, not a projection anything multiplies by, so it
    // is written across at sixteen bits and its bytes do not move. Folding it into the bpw
    // estimate understates a pack of one of these by most of its size.
    var table = 0
    for name in EngramRepack.shards(in: checkpoint) {
      table += ((try? checkpoint.tensor(name).size) ?? 0) * 2
    }
    self.engramBytes = table
    // Scales and biases already sit inside the target width, so this is fairer than the
    // nominal one.
    self.estimateBytes =
      Int(Double(max(0, sourceBytes - table)) * profile.targetBpw / 16.0) + table
  }

  public var destinationExists: Bool {
    FileManager.default.fileExists(atPath: destination.path)
  }

  public func run(
    shardBytes: Int = 4 * 1_073_741_824,
    calibrate: Bool = false,
    replacing: Bool = false,
    progress: @escaping @Sendable (Quantizer.Progress) -> Void
  ) throws -> Quantizer.Outcome {
    let fm = FileManager.default
    if destinationExists {
      guard replacing else { throw Problem.destinationExists(destination) }
      try fm.removeItem(at: destination)
    }

    let checkpoint = try SourceCheckpoint(directory: source.directory)
    let quantizer = Quantizer(
      source: checkpoint, profile: profile, destination: destination,
      shardLimit: shardBytes, calibrate: calibrate, onProgress: progress)
    return try quantizer.run()
  }
}
