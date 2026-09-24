// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXFast
import MLXNN

public final class GatedDeltaNet: @unchecked Sendable {
  private let inProjQKV: PackedLinear
  private let inProjZ: PackedLinear
  private let inProjA: any Projection
  private let inProjB: any Projection
  private let conv1dWeight: MLXArray
  private let aLog: MLXArray
  private let dtBias: MLXArray
  private let normWeight: MLXArray
  private let outProj: PackedLinear

  private let numValueHeads: Int
  private let numKeyHeads: Int
  private let keyHeadDim: Int
  private let valueHeadDim: Int
  private let keyDim: Int
  private let valueDim: Int
  private let convDim: Int
  private let kernelSize: Int
  private let normEps: Float
  private let sigmoidOutputGate: Bool
  private let headRepeat: Int
  private let valueHeadLayout: ValueHeadLayout

  private let unitKeyNorm: MLXArray
  private let zSplit: (slice: ANESlice, tail: PackedLinear)?

  public init(
    config: BonsaiConfig.TextConfig, layer: Int,
    factory: PackedModuleFactory, store: WeightStore
  ) throws {
    let prefix = "model.layers.\(layer).linear_attn"
    let tensorPrefix = factory.tensorPrefix + prefix

    self.numValueHeads = config.linearNumValueHeads
    self.numKeyHeads = config.linearNumKeyHeads
    self.keyHeadDim = config.linearKeyHeadDim
    self.valueHeadDim = config.linearValueHeadDim
    self.keyDim = numKeyHeads * keyHeadDim
    self.valueDim = numValueHeads * valueHeadDim
    self.convDim = keyDim * 2 + valueDim
    self.kernelSize = config.linearConvKernelDim
    self.normEps = config.rmsNormEps
    // Qwen3-Next gates the delta-net's output with silu; the hyper-connected models name the
    // activation in the config and ask for a sigmoid. Reading it as silu is a quiet wrong
    // answer, not a failure — the shapes are the same either way.
    self.sigmoidOutputGate = config.outputGateType == "sigmoid"

    guard numValueHeads % numKeyHeads == 0 else {
      throw BonsaiError.unsupportedModel(
        "linear_num_value_heads (\(numValueHeads)) is not a multiple of "
          + "linear_num_key_heads (\(numKeyHeads))")
    }
    self.headRepeat = numValueHeads / numKeyHeads
    self.valueHeadLayout = store.valueHeadLayout

    self.inProjQKV = try factory.linear(prefix + ".in_proj_qkv")
    self.inProjZ = try factory.linear(prefix + ".in_proj_z")
    self.outProj = try factory.linear(prefix + ".out_proj")

    self.inProjA = try factory.projection(prefix + ".in_proj_a")
    self.inProjB = try factory.projection(prefix + ".in_proj_b")
    self.conv1dWeight = try store(tensorPrefix + ".conv1d.weight")
    self.aLog = try store(tensorPrefix + ".A_log")
    self.dtBias = try store(tensorPrefix + ".dt_bias")
    self.normWeight = try store(tensorPrefix + ".norm.weight")

    self.unitKeyNorm = MLXArray.ones([keyHeadDim], dtype: .float32)

    // Only the token-local z is offloaded. qkv feeds the delta rule's state, where an approximate
    // value would not stay local but compound along the prompt, so it keeps the 2-bit path.
    if let bank = BonsaiRuntime.aneBank,
      let slice = bank.slice("\(layer).linear_attn.in_proj_z"),
      slice.inputDim == inProjZ.inputDim, slice.outputDim < inProjZ.outputDim,
      let tail = try? inProjZ.channels(from: slice.outputDim)
    {
      self.zSplit = (slice, tail)
    } else {
      self.zSplit = nil
    }
  }

  public func callAsFunction(_ x: MLXArray, cache: GatedDeltaNetCache?) -> MLXArray {
    let b = x.dim(0)
    let s = x.dim(1)

    // Handed over before the recurrent work starts, so the Neural Engine runs underneath the
    // convolution and the delta rule rather than after them.
    var zPending: ANESlice.Pending?
    var zTail: MLXArray?
    if let zSplit, b * s == zSplit.slice.rows {
      let rotated = inProjZ.rotate(x.reshaped([b * s, x.dim(2)]))
      zPending = zSplit.slice.dispatch(rotated)
      zTail = zSplit.tail.applyRotated(rotated)
    }

    let qkv = inProjQKV(x)
    let aRaw = inProjA(x)
    let bRaw = inProjB(x)

    let keep = kernelSize - 1
    let convState =
      cache?.convState ?? MLXArray.zeros([b, keep, convDim], dtype: qkv.dtype)
    let convInput = concatenated([convState, qkv], axis: 1)
    cache?.convState = convInput[0..., (convInput.dim(1) - keep)..., 0...]

    let convOut = silu(conv1d(convInput, conv1dWeight, groups: convDim))

    var q = convOut[0..., 0..., ..<keyDim].reshaped([b, s, numKeyHeads, keyHeadDim])
    var k = convOut[0..., 0..., keyDim..<(2 * keyDim)]
      .reshaped([b, s, numKeyHeads, keyHeadDim])
    let v = convOut[0..., 0..., (2 * keyDim)...]
      .reshaped([b, s, numValueHeads, valueHeadDim])

    let invScale = Float(keyHeadDim).squareRoot()
    q =
      (1.0 / (invScale * invScale))
      * MLXFast.rmsNorm(q, weight: unitKeyNorm.asType(q.dtype), eps: 1e-6)
    k =
      (1.0 / invScale)
      * MLXFast.rmsNorm(k, weight: unitKeyNorm.asType(k.dtype), eps: 1e-6)

    let beta = sigmoid(bRaw)
    let g = exp(-exp(aLog.asType(.float32)) * softplus((aRaw + dtBias).asType(.float32)))

    let state =
      cache?.recurrentState
      ?? MLXArray.zeros(
        [b, numValueHeads, valueHeadDim, keyHeadDim], dtype: .float32)

    if let cache, cache.recordsSteps {
      cache.steps = GatedDeltaNetCache.DeltaSteps(
        q: q, k: k, v: v, g: g, beta: beta, state: state, convInput: convInput, convKeep: keep,
        headRepeat: headRepeat, layout: valueHeadLayout, start: cache.offset)
    }
    let (y, newState) = GatedDeltaNet.deltaRule(
      q: q, k: k, v: v, g: g, beta: beta, state: state, headRepeat: headRepeat,
      layout: valueHeadLayout)

    cache?.recurrentState = newState
    cache?.advance(s)

    var z: MLXArray?
    if let zPending, let zTail, let head = try? zPending.wait() {
      z = concatenated([head, zTail], axis: -1)
        .reshaped([b, s, numValueHeads, valueHeadDim])
    }
    let zValue = z ?? inProjZ(x).reshaped([b, s, numValueHeads, valueHeadDim])

    let normalized = MLXFast.rmsNorm(y, weight: normWeight.asType(y.dtype), eps: normEps)
    let z32 = zValue.asType(.float32)
    let activated = sigmoidOutputGate ? sigmoid(z32) : silu(z32)
    let gated = (activated * normalized.asType(.float32)).asType(x.dtype)

    return outProj(gated.reshaped([b, s, valueDim]))
  }

  public static func deltaRule(
    q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
    state: MLXArray, headRepeat: Int, layout: ValueHeadLayout = .grouped
  ) -> (MLXArray, MLXArray) {
    #if !targetEnvironment(simulator)
      if Device.defaultDevice().deviceType == .gpu, let kernel = metalKernel {
        let outputs = kernel(
          [q, k, v, g, beta, state, q.dim(1)],
          template: [
            ("InT", q.dtype), ("StT", state.dtype),
            ("Dk", k.dim(3)), ("Dv", v.dim(3)),
            ("Hk", k.dim(2)), ("Hv", v.dim(2)),
            ("Tiled", layout == .tiled),
          ],
          grid: (32, v.dim(3), q.dim(0) * v.dim(2)),
          threadGroup: (32, 4, 1),
          outputShapes: [[q.dim(0), q.dim(1), v.dim(2), v.dim(3)], state.shape],
          outputDTypes: [q.dtype, state.dtype])
        return (outputs[0], outputs[1])
      }
    #endif
    return opsDeltaRule(
      q: q, k: k, v: v, g: g, beta: beta, state: state, headRepeat: headRepeat,
      layout: layout)
  }

  static func opsDeltaRule(
    q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
    state initialState: MLXArray, headRepeat: Int, layout: ValueHeadLayout = .grouped
  ) -> (MLXArray, MLXArray) {
    let steps = q.dim(1)
    let qr = Self.broadcastHeads(q, repeat: headRepeat, layout: layout)
    let kr = Self.broadcastHeads(k, repeat: headRepeat, layout: layout)

    var state = initialState
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(steps)

    for t in 0..<steps {
      let qt = qr[0..., t].asType(.float32).expandedDimensions(axis: 2)
      let kt = kr[0..., t].asType(.float32).expandedDimensions(axis: 2)
      let vt = v[0..., t].asType(.float32)
      let gt = g[0..., t].expandedDimensions(axes: [-1, -2])
      let betaT = beta[0..., t].asType(.float32).expandedDimensions(axis: -1)

      state = state * gt
      let kvMemory = (state * kt).sum(axis: -1)
      let delta = (vt - kvMemory) * betaT
      state = state + kt * delta.expandedDimensions(axis: -1)
      outputs.append((state * qt).sum(axis: -1).asType(q.dtype))
    }

    return (stacked(outputs, axis: 1), state)
  }

  /// Spreads one key head over the value heads it serves: each key head repeated in place for
  /// the grouped order, the whole row of them repeated for the tiled one.
  static func broadcastHeads(
    _ x: MLXArray, repeat count: Int, layout: ValueHeadLayout
  ) -> MLXArray {
    guard count > 1 else { return x }
    switch layout {
    case .grouped:
      return repeated(x, count: count, axis: 2)
    case .tiled:
      return concatenated(Array(repeating: x, count: count), axis: 2)
    }
  }

  private static let metalKernel: MLXFast.MLXFastKernel? = {
    #if canImport(Metal)
      let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = Tiled ? (hv_idx % Hk) : (hv_idx / (Hv / Hk));
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
              }
              kv_mem = simd_sum(kv_mem);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """
      return MLXFast.metalKernel(
        name: "bonsai_gated_delta_step",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: source)
    #else
      return nil
    #endif
  }()
}
