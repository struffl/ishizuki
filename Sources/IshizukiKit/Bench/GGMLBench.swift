// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom

/// What a decode step actually costs, one projection at a time.
///
/// Decode is memory bound: a token reads every weight once. The useful number is therefore not
/// milliseconds but how much of the machine's bandwidth a kernel manages to use, which is what
/// makes an affine pack and a GGUF comparable at all despite holding different bits.
///
/// Several launches go into each `eval`. One launch per `eval` measures the host's dispatch cost
/// as much as the kernel's: it puts a floor of a few hundred microseconds under every projection,
/// which is longer than the small ones take and would hide any change to them.
public struct GGMLBench: Sendable {
  public struct Options: Sendable {
    public var gguf: URL
    public var iterations: Int = 50
    public var batch: Int = 1
    public var queue: Int = 16

    public init(gguf: URL) { self.gguf = gguf }
  }

  /// The mean seconds one launch took, with `queue` of them in flight per `eval`.
  private static func time(_ options: Options, _ launch: (Int) -> MLXArray) -> Double {
    for i in 0..<options.queue { eval(launch(i)) }
    var best = Double.infinity
    for _ in 0..<3 {
      let start = Date()
      for _ in 0..<options.iterations { eval((0..<options.queue).map(launch)) }
      best = min(
        best, -start.timeIntervalSinceNow / Double(options.iterations * options.queue))
    }
    return best
  }

  public static func run(
    _ options: Options,
    log: @escaping @Sendable (String) -> Void
  ) throws {
    let file = try GGUFFile(url: options.gguf)
    let store = try GGUFWeights.load(file: file)

    // One tensor per block type the file actually uses, so the cost of each decode is visible
    // rather than averaged. A type with no lookup table is the control.
    var seen: Set<GGMLType> = []
    var wanted: [String] = []
    for tensor in file.tensors {
      guard let name = GGUFTensorNaming.canonical(tensor.name, textLayers: 64),
        let blocks = store.ggml(name), !seen.contains(blocks.type),
        blocks.inputDim >= 4096
      else { continue }
      seen.insert(blocks.type)
      wanted.append(String(name.dropFirst(GGUFTensorNaming.prefix.count).dropLast(".weight".count)))
    }
    log("ggml matvec, batch \(options.batch)")
    log("")

    for path in wanted {
      let name = GGUFTensorNaming.prefix + path + ".weight"
      guard let blocks = store.ggml(name) else { continue }
      let k = blocks.inputDim
      let n = blocks.outputDim
      let bytes = blocks.bytes.size

      let xs = (0..<options.queue).map { _ in
        MLXRandom.normal([options.batch, k]).asType(.bfloat16)
      }
      eval(xs)

      guard
        GGMLKernels.matvec(xs[0], blocks: blocks.bytes, type: blocks.type, outputDim: n) != nil
      else {
        log("  \(path): no kernel for \(blocks.type.name)")
        continue
      }

      let seconds = time(options) { i in
        GGMLKernels.matvec(xs[i], blocks: blocks.bytes, type: blocks.type, outputDim: n)!
      }
      let gbs = Double(bytes) / seconds / 1_073_741_824

      log(
        String(
          format: "  %-34s %5s  %5d x %-5d  %7.3f ms  %6.1f GB/s",
          (path as NSString).utf8String!, (blocks.type.name as NSString).utf8String!,
          n, k, seconds * 1000, gbs))
    }

    log("")
    log("  for scale, the same shapes as MLX affine 4-bit:")
    for (n, k) in [(17408, 5120), (5120, 17408)] {
      let w = MLXRandom.normal([n, k]).asType(.float32)
      let (q, scales, biases) = quantized(w, groupSize: 64, bits: 4)
      let xs = (0..<options.queue).map { _ in
        MLXRandom.normal([options.batch, k]).asType(.bfloat16)
      }
      eval(q, scales, biases!, xs)
      // `size` counts elements; only the GGML blocks are bytes already.
      let bytes = q.size * 4 + scales.size * 2 + biases!.size * 2

      let seconds = time(options) { i in
        quantizedMatmul(
          xs[i], q, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
      }
      log(
        String(
          format: "  mlx quantizedMatmul 4-bit           %5d x %-5d  %7.3f ms  %6.1f GB/s",
          n, k, seconds * 1000, Double(bytes) / seconds / 1_073_741_824))
    }
  }
}
