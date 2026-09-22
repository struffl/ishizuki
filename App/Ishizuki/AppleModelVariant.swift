// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Which of Apple's on-device models is actually answering.

import FoundationModels
import IshizukiKit

/// The name the system gives the model it serves this machine — `AFM 3 Core` on most, the far
/// larger `AFM 3 Core Advanced` on one that runs it.
///
/// Nothing chooses between them: `SystemLanguageModel` takes a use case and a set of guardrails
/// and nothing else, and `variant` is read-only. So this reports what was handed over rather
/// than what was asked for, which is the only honest thing a label here can say.
@available(macOS 27.0, *)
enum AppleModelVariant {
  static let name: String = SystemLanguageModel.default.variant.displayName

  /// The subtitle for a row, with the variant folded in where there is one to name. Private
  /// Cloud Compute has no equivalent: its tiers are not separately addressable.
  static func subtitle(for model: AppleFoundationModel) -> String {
    switch model {
    case .onDevice: "\(name) · \(model.subtitle)"
    case .privateCloudCompute: model.subtitle
    }
  }
}
