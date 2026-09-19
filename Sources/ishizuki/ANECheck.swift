// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import CoreML
import Foundation
import IshizukiKit
import MLX
import MLXRandom

struct ANECheck: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "ane-check",
    abstract: "Measure a projection split across the Neural Engine and Metal.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long, help: "Compiled .mlmodelc or .mlpackage holding the INT8 channel slice.")
  var slice: String
  @Option(name: .long, help: "Projection the slice was cut from.")
  var projection: String = "model.layers.0.mlp.gate_proj"
  @Option(name: .long) var rows: Int = 2048
  @Option(name: .long) var iterations: Int = 10

  func run() throws {
    let packURL = URL(filePath: model)
    let config = try BonsaiConfig.load(directory: packURL)
    try config.validate()
    let store = try WeightStore(directory: packURL)
    let prefix = store.has("language_model.model.norm.weight") ? "language_model." : ""
    let factory = PackedModuleFactory(store: store, config: config, tensorPrefix: prefix)
    let full = try factory.linear(projection)

    let mlConfig = MLModelConfiguration()
    mlConfig.computeUnits = .cpuAndNeuralEngine
    var aneURL = URL(filePath: slice)
    if aneURL.pathExtension == "mlpackage" {
      aneURL = try MLModel.compileModel(at: aneURL)
      print(Style.faint("compiled to \(aneURL.path)"))
    }
    let aneModel = try MLModel(contentsOf: aneURL, configuration: mlConfig)

    guard
      let inDesc = aneModel.modelDescription.inputDescriptionsByName.first?.value,
      let inShape = inDesc.multiArrayConstraint?.shape as? [Int],
      let outDesc = aneModel.modelDescription.outputDescriptionsByName.first?.value,
      let outShape = outDesc.multiArrayConstraint?.shape as? [Int]
    else { throw ValidationError("the CoreML model has no multi-array input/output") }

    let split = outShape[1]
    let inputName = inDesc.name
    guard inShape[0] == rows, inShape[1] == full.inputDim else {
      throw ValidationError(
        "slice expects \(inShape), bench is \(rows)x\(full.inputDim)")
    }
    guard split < full.outputDim else {
      throw ValidationError("slice covers the whole projection; nothing left for Metal")
    }

    let rest = try PackedLinear(
      weight: full.weight[split...], scales: full.scales[split...],
      biases: full.biases[split...], signs: full.signs, block: full.block,
      groupSize: full.groupSize, bits: full.bits)

    MLXRandom.seed(5)
    let x = MLXRandom.normal([rows, full.inputDim]).asType(.float16) * 0.05
    let rotated: MLXArray
    if full.block > 0, let signs = full.signs {
      rotated = hadamardRotate(x, block: full.block, signs: signs, inverse: false)
    } else {
      rotated = x
    }
    eval(rotated)

    let ioType = inDesc.multiArrayConstraint?.dataType ?? .float32
    guard ioType == outDesc.multiArrayConstraint?.dataType,
      ioType == .float16 || ioType == .float32
    else {
      throw ValidationError("the CoreML interface must be float16 or float32 on both sides")
    }

    let buffer = try MLMultiArray(
      shape: [rows, full.inputDim] as [NSNumber], dataType: ioType)
    buffer.withUnsafeMutableBytes { raw, _ in
      if ioType == .float16 {
        let host = rotated.asType(.float16).asArray(Float16.self)
        let dst = raw.bindMemory(to: Float16.self)
        host.withUnsafeBufferPointer {
          dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count)
        }
      } else {
        let host = rotated.asType(.float32).asArray(Float.self)
        let dst = raw.bindMemory(to: Float.self)
        host.withUnsafeBufferPointer {
          dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count)
        }
      }
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: [
      inputName: MLFeatureValue(multiArray: buffer)
    ])

    func metalOnly() {
      let y = quantizedMM(
        rotated, rest.weight, scales: rest.scales, biases: rest.biases,
        transpose: true, groupSize: rest.groupSize, bits: rest.bits, mode: .affine)
      eval(y)
    }
    func metalFull() {
      let y = quantizedMM(
        rotated, full.weight, scales: full.scales, biases: full.biases,
        transpose: true, groupSize: full.groupSize, bits: full.bits, mode: .affine)
      eval(y)
    }
    func aneOnly() { _ = try? aneModel.prediction(from: provider) }

    let head = try PackedLinear(
      weight: full.weight[..<split], scales: full.scales[..<split],
      biases: full.biases[..<split], signs: full.signs, block: full.block,
      groupSize: full.groupSize, bits: full.bits)
    let reference = quantizedMM(
      rotated, head.weight, scales: head.scales, biases: head.biases,
      transpose: true, groupSize: head.groupSize, bits: head.bits, mode: .affine)
    eval(reference)

    let predicted = try aneModel.prediction(from: provider)
    guard let got = predicted.featureValue(for: outDesc.name)?.multiArrayValue else {
      throw ValidationError("the CoreML model returned no array for \(outDesc.name)")
    }
    let count = rows * split
    var aneValues = [Float](repeating: 0, count: count)
    got.withUnsafeBytes { raw in
      if ioType == .float16 {
        let src = raw.bindMemory(to: Float16.self)
        for i in 0..<count { aneValues[i] = Float(src[i]) }
      } else {
        let src = raw.bindMemory(to: Float.self)
        aneValues.withUnsafeMutableBufferPointer { dst in
          dst.baseAddress!.update(from: src.baseAddress!, count: count)
        }
      }
    }
    let aneArray = MLXArray(aneValues).reshaped([rows, split]).asType(.float32)
    let ref32 = reference.asType(.float32)
    let cosine =
      (aneArray * ref32).sum().item(Float.self)
      / (sqrt((aneArray * aneArray).sum().item(Float.self))
        * sqrt((ref32 * ref32).sum().item(Float.self)))
    let maxRel =
      abs(aneArray - ref32).max().item(Float.self)
      / max(abs(ref32).max().item(Float.self), 1e-6)

    let baseline = time(metalFull)
    let metal = time(metalOnly)
    let neural = time(aneOnly)
    let hybrid = time {
      let group = DispatchGroup()
      group.enter()
      DispatchQueue.global(qos: .userInitiated).async {
        aneOnly()
        group.leave()
      }
      metalOnly()
      group.wait()
    }

    let fraction = Double(split) / Double(full.outputDim)
    let serial = metal + neural
    let overlap = (serial - hybrid) / min(metal, neural)

    print(Style.banner("neural engine split"))
    print("")
    print(Style.field("projection", "\(projection)  \(full.inputDim)→\(full.outputDim)"))
    print(Style.field("rows", "\(rows)"))
    print(Style.field("io", ioType == .float16 ? "float16" : "float32"))
    print(
      Style.field(
        "split",
        String(
          format: "%d on ANE (%.1f%%), %d on Metal", split, fraction * 100,
          full.outputDim - split)))
    print("")
    let agree = String(
      format: "  ANE vs Metal on the same rows   cosine %.6f   max rel %.4f", cosine, maxRel)
    print(cosine > 0.999 ? Style.good(agree) : Style.bad(agree))
    print("")
    print(
      String(format: "  Metal, whole projection       %7.2f ms", baseline * 1000))
    print(String(format: "  Metal, its share alone        %7.2f ms", metal * 1000))
    print(String(format: "  ANE, its share alone          %7.2f ms", neural * 1000))
    print(String(format: "  both, dispatched together     %7.2f ms", hybrid * 1000))
    print("")
    print(
      String(
        format: "  serial sum                    %7.2f ms", serial * 1000))
    print(
      String(
        format: "  overlap recovered             %6.0f%% of the shorter leg", overlap * 100))
    let speedup = baseline / hybrid
    let line = String(
      format: "  hybrid vs Metal alone         %6.2fx", speedup)
    print(speedup > 1 ? Style.good(line) : Style.bad(line))

    // Both legs run concurrently, so the split is best where they finish together. Extrapolating
    // each leg to the whole projection gives that point directly.
    let aneWhole = neural / fraction
    let optimal = baseline / (baseline + aneWhole)
    print("")
    print(
      Style.bright(
        String(
          format: "  optimal split here            %5.1f%% on ANE, predicted %.2fx",
          optimal * 100, 1 / (1 - optimal))))
    print(
      Style.faint(
        String(
          format: "  whole projection: Metal %.1f ms, ANE %.1f ms",
          baseline * 1000, aneWhole * 1000)))
  }

  private func time(_ body: () -> Void) -> Double {
    body()
    let start = Date()
    for _ in 0..<iterations { body() }
    return -start.timeIntervalSinceNow / Double(iterations)
  }
}
