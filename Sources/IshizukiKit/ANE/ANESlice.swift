// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

@preconcurrency import CoreML
import Foundation
import MLX

// One projection's leading output channels, held as INT8 for the Neural Engine. The Metal side
// keeps the remaining channels at 2-bit and the two halves run at the same time, so the work is
// handed over as early as possible and collected as late as possible.
public final class ANESlice: @unchecked Sendable {
  public let rows: Int
  public let inputDim: Int
  public let outputDim: Int

  private let model: MLModel
  private let inputName: String
  private let outputName: String
  private let queue: DispatchQueue
  private let buffer: MLMultiArray
  private let provider: MLDictionaryFeatureProvider

  public final class Pending {
    fileprivate let done = DispatchSemaphore(value: 0)
    fileprivate var output: MLMultiArray?
    fileprivate var failure: Error?
    fileprivate let slice: ANESlice

    fileprivate init(slice: ANESlice) { self.slice = slice }

    public func wait() throws -> MLXArray {
      done.wait()
      if let failure { throw failure }
      guard let output else {
        throw BonsaiError.missingComponent("the Neural Engine returned no output")
      }
      return slice.read(output)
    }
  }

  public init(url: URL, queue: DispatchQueue) throws {
    var compiled = url
    if compiled.pathExtension == "mlpackage" {
      compiled = try MLModel.compileModel(at: compiled)
    }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    self.model = try MLModel(contentsOf: compiled, configuration: configuration)
    self.queue = queue

    let description = model.modelDescription
    guard let input = description.inputDescriptionsByName.values.first,
      let output = description.outputDescriptionsByName.values.first,
      let inShape = input.multiArrayConstraint?.shape as? [Int],
      let outShape = output.multiArrayConstraint?.shape as? [Int],
      input.multiArrayConstraint?.dataType == .float16,
      output.multiArrayConstraint?.dataType == .float16
    else {
      throw BonsaiError.unsupportedModel(
        "\(url.lastPathComponent) is not a float16 [rows, dim] Neural Engine slice")
    }
    self.inputName = input.name
    self.outputName = output.name
    self.rows = inShape[0]
    self.inputDim = inShape[1]
    self.outputDim = outShape[1]

    // One input buffer for the life of the slice: every prefill chunk is the same shape, and at
    // this size the allocation costs more than the copy into it.
    self.buffer = try MLMultiArray(
      shape: [inShape[0], inShape[1]] as [NSNumber], dataType: .float16)
    self.provider = try MLDictionaryFeatureProvider(dictionary: [
      input.name: MLFeatureValue(multiArray: buffer)
    ])
  }

  public func dispatch(_ rotated: MLXArray) -> Pending {
    let pending = Pending(slice: self)
    do {
      let host = rotated.asType(.float16).asArray(Float16.self)
      buffer.withUnsafeMutableBytes { raw, _ in
        let destination = raw.bindMemory(to: Float16.self)
        host.withUnsafeBufferPointer {
          destination.baseAddress!.update(from: $0.baseAddress!, count: $0.count)
        }
      }
      queue.async { [self] in
        do {
          pending.output = try model.prediction(from: provider)
            .featureValue(for: outputName)?.multiArrayValue
        } catch {
          pending.failure = error
        }
        pending.done.signal()
      }
    } catch {
      pending.failure = error
      pending.done.signal()
    }
    return pending
  }

  // The result is handed to MLX where it lies: the finalizer holds the CoreML array alive until
  // MLX is done with it, so the output crosses no buffer on the way back.
  private func read(_ array: MLMultiArray) -> MLXArray {
    let held = array
    return MLXArray(
      rawPointer: held.dataPointer, [rows, outputDim], dtype: .float16,
      finalizer: { _ = held })
  }
}
