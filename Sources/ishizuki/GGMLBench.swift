// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit
import MLX
import MLXRandom

/// What a decode step actually costs, one projection at a time.
///
/// Decode is memory bound: a token reads every weight once. The useful number is therefore not
/// milliseconds but how much of the machine's bandwidth a kernel manages to use, which is what
/// makes an affine pack and a GGUF comparable at all despite holding different bits.
struct GGMLBench: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "ggml-bench",
    abstract: "Time the GGML matvec against MLX's own quantized matmul.")

  @Option(name: .long, help: "A .gguf to take real weights from.") var gguf: String
  @Option(name: .long) var iterations: Int = 50
  @Option(name: .long, help: "Rows of x per launch.") var batch: Int = 1

  func run() throws {
    let file = try GGUFFile(url: URL(filePath: gguf))
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
    print(Style.banner("ggml matvec, batch \(batch)"))
    print("")

    for path in wanted {
      let name = GGUFTensorNaming.prefix + path + ".weight"
      guard let blocks = store.ggml(name) else { continue }
      let k = blocks.inputDim
      let n = blocks.outputDim
      let bytes = blocks.bytes.size

      let x = MLXRandom.normal([batch, k]).asType(.bfloat16)
      eval(x)

      guard let warm = GGMLKernels.matvec(x, blocks: blocks.bytes, type: blocks.type, outputDim: n)
      else {
        print("  \(path): no kernel for \(blocks.type.name)")
        continue
      }
      eval(warm)

      let start = Date()
      for _ in 0..<iterations {
        let y = GGMLKernels.matvec(
          x, blocks: blocks.bytes, type: blocks.type, outputDim: n)
        eval(y!)
      }
      let seconds = -start.timeIntervalSinceNow / Double(iterations)
      let gbs = Double(bytes) / seconds / 1_073_741_824

      print(
        String(
          format: "  %-34s %5s  %5d x %-5d  %7.3f ms  %6.1f GB/s",
          (path as NSString).utf8String!, (blocks.type.name as NSString).utf8String!,
          n, k, seconds * 1000, gbs))
    }

    print("")
    print("  for scale, the same shapes as MLX affine 4-bit:")
    for (n, k) in [(17408, 5120), (5120, 17408)] {
      let w = MLXRandom.normal([n, k]).asType(.float32)
      let (q, scales, biases) = quantized(w, groupSize: 64, bits: 4)
      let x = MLXRandom.normal([batch, k]).asType(.bfloat16)
      eval(q, scales, biases!, x)
      // `size` counts elements; only the GGML blocks are bytes already.
      let bytes = q.size * 4 + scales.size * 2 + biases!.size * 2

      let warm = quantizedMatmul(
        x, q, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
      eval(warm)
      let start = Date()
      for _ in 0..<iterations {
        let y = quantizedMatmul(
          x, q, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
        eval(y)
      }
      let seconds = -start.timeIntervalSinceNow / Double(iterations)
      print(
        String(
          format: "  mlx quantizedMatmul 4-bit           %5d x %-5d  %7.3f ms  %6.1f GB/s",
          n, k, seconds * 1000, Double(bytes) / seconds / 1_073_741_824))
    }
  }
}
