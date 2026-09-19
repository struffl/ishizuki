// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct Generate: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "generate",
    abstract: "Generate text with Bonsai 2.")

  @Option(name: .long, help: "Path to the MLX pack directory.")
  var model: String = defaultModelPath

  @Option(name: .long, help: "HuggingFace repo to fetch the model from if missing.")
  var repo: String = defaultRepo
  @Flag(name: .long, help: "Skip the model download/repair check.") var offline = false

  @Option(name: .shortAndLong, help: "Prompt text.")
  var prompt: String = "Explain what makes ternary weight quantization work, in three sentences."

  @Option(name: .long, help: "System prompt.")
  var system: String?

  @Option(name: .shortAndLong, help: "Maximum tokens to generate.")
  var maxTokens: Int = 256

  @Option(name: .long) var temperature: Float = 0.7
  @Option(name: .long) var topP: Float = 1.0
  @Option(name: .long) var topK: Int = 0

  @Option(
    name: .long,
    help: "Min-p: keep tokens at least this fraction as likely as the top token. Off by default.")
  var minP: Float = 0.0
  @Option(name: .long) var repetitionPenalty: Float = 1.0
  @Option(name: .long) var seed: UInt64?

  @Option(name: .long, help: "Reasoning effort: none, low, medium, high, xhigh.")
  var reasoning: String?

  @Flag(name: .long, help: "Disable the model's thinking block.")
  var noThinking = false

  @Flag(name: .long, help: "Feed the prompt verbatim, skipping the chat template.")
  var raw = false

  @Option(name: .long, help: "Image file to attach. Repeat for several.")
  var image: [String] = []

  @Option(name: .long, help: "Cap on vision tokens per image; lower is faster to prefill.")
  var maxImageTokens: Int = 1024

  @Flag(name: .long, help: "Use the fused Hadamard kernel instead of MLX's ops (measured slower).")
  var fusedHadamard = false

  @Flag(name: .long, help: "Route batch-2..5 projections through the qmv_wide kernel.")
  var qmvWide = false

  @OptionGroup var neural: ANEOption

  @Option(
    name: .long,
    help: "Stretch context past the trained 262144 by this factor, e.g. 2 for ~512K.")
  var contextScale: Float = 1

  @Option(name: .long, help: "How to stretch it: yarn, ntk, linear, none.")
  var ropeScaling: String = "yarn"

  @Option(
    name: .long,
    help:
      "Scheduling: adaptive (default), polite, normal, background. Pass normal on a dedicated machine; background is >90x slower and rarely what you want."
  )
  var politeness: String = "adaptive"

  @Option(name: .long, help: "GPU cache limit in MB (0 leaves MLX's default).")
  var cacheLimit: Int = 0

  @Flag(name: .long, help: "Disable coloured output.")
  var noColor = false

  func run() throws {
    if noColor { Style.disable() }

    let level = Politeness.Level(rawValue: politeness) ?? .adaptive
    Politeness.apply(level)

    let scaling =
      contextScale > 1
      ? RopeScaling(
        method: RopeScaling.Method(rawValue: ropeScaling) ?? .yarn,
        factor: contextScale)
      : RopeScaling.none

    BonsaiRuntime.useFusedHadamard = fusedHadamard
    BonsaiRuntime.useQMVWide = qmvWide

    if cacheLimit > 0 {
      Memory.cacheLimit = cacheLimit * 1024 * 1024
    }

    let packURL = URL(filePath: resolvedModelPath(model, repo: repo))
    try neural.apply(pack: packURL)
    if !offline {
      try ModelDownloader.ensure(directory: packURL, repo: repo)
    }
    let loadStart = Date()
    let bonsai = try BonsaiModel(directory: packURL, ropeScaling: scaling)
    if scaling.isActive {
      note(
        Style.field(
          "context",
          Style.accent("\(scaling.method.rawValue) ×\(contextScale)")
            + Style.faint(" → ~\(scaling.effectiveContext) tok")))
    }
    if level != .normal {
      note(Style.field("sched", Style.faint(Politeness.describe(level))))
    }
    note(
      Style.field("model", Style.accent(packURL.lastPathComponent))
        + Style.faint(String(format: "  %.1fs", -loadStart.timeIntervalSinceNow)))

    let text: String
    if raw {
      text = prompt
    } else {
      let template = try ChatTemplate(directory: packURL)
      var messages: [ChatMessage] = []
      if let system { messages.append(.system(system)) }
      messages.append(
        image.isEmpty
          ? .user(prompt) : .user(text: prompt, imageCount: image.count))
      text = try template.render(
        messages: messages,
        addGenerationPrompt: true,
        enableThinking: !noThinking,
        reasoningEffort: reasoning.flatMap(ReasoningEffort.init(rawValue:)))
    }

    var promptTokens = bonsai.tokenizer.encode(text)

    var promptEmbeddings: MLXArray?
    var positions: MLXArray?
    if !image.isEmpty {
      guard let visionConfig = bonsai.config.visionConfig else {
        throw BonsaiError.missingComponent("this pack has no vision configuration")
      }
      let processor = ImageProcessor(
        config: visionConfig, maxPixels: maxImageTokens * 32 * 32)
      let processed = try image.map {
        try processor.process(contentsOf: URL(filePath: $0))
      }
      for (path, item) in zip(image, processed) {
        let name = (path as NSString).lastPathComponent
        note(
          Style.field(
            "image",
            Style.accent(name)
              + Style.faint(
                " \(item.grid.h)×\(item.grid.w) → \(item.tokenCount) tok")))
      }
      let multimodal = try bonsai.prepareMultimodal(
        tokens: promptTokens, images: processed)
      promptTokens = multimodal.tokens
      promptEmbeddings = multimodal.embeddings
      positions = multimodal.positions
    }

    note(Style.field("prompt", Style.accent("\(promptTokens.count)") + Style.faint(" tok")))
    note("")

    let generator = Generator(
      model: bonsai, prefillChunkSize: Politeness.prefillChunk(for: level))
    generator.politeness = level
    let options = SamplingOptions(
      temperature: temperature, topP: topP, topK: topK, minP: minP,
      repetitionPenalty: repetitionPenalty, seed: seed)

    let result = generator.generate(
      promptTokens: promptTokens, options: options, maxTokens: maxTokens,
      promptEmbeddings: promptEmbeddings, positions: positions,
      onToken: { fragment in
        print(fragment, terminator: "")
        fflush(stdout)
        return true
      })

    print("\n")
    let s = result.stats
    note(Style.rule)
    note(
      Style.field(
        "prefill",
        Style.bright(String(format: "%.1f", s.promptTokensPerSecond))
          + Style.muted(" tok/s")
          + Style.faint(String(format: "  %d tok in %.2fs", s.promptTokens, s.promptSeconds))))
    note(
      Style.field(
        "decode",
        Style.bright(String(format: "%.1f", s.generationTokensPerSecond))
          + Style.muted(" tok/s")
          + Style.faint(String(format: "  %d tok in %.2fs", s.generatedTokens, s.generationSeconds))
      ))
  }
}

func note(_ line: String) {
  FileHandle.standardError.write(Data((line + "\n").utf8))
}
