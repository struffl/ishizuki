// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Where a job's output lands: the terminal's scrollback, kept legible over glass.

import IshizukiKit
import SwiftUI

struct ConsoleView: View {
  @Bindable var runner: JobRunner

  var body: some View {
    if let job = runner.job {
      GlassCard {
        VStack(alignment: .leading, spacing: 9) {
          HStack(spacing: 9) {
            Text(job.name)
              .font(.system(size: 14.5, weight: .semibold))
            if job.isRunning {
              ProgressView().controlSize(.small)
            }
            Spacer()
            Text(ReadoutFormat.duration(job.elapsed))
              .font(.subheadline.monospacedDigit())
              .foregroundStyle(.secondary)
            if job.isRunning {
              Button("Cancel") { runner.cancel() }
                .buttonStyle(.glass)
            } else {
              Button("Clear") { runner.clear() }
                .buttonStyle(.glass)
            }
          }

          if let progress = job.progress {
            Bar(fraction: progress, tint: .accentColor, width: 320)
          }

          if let failure = job.failure {
            Text(failure)
              .font(.system(.callout, design: .monospaced))
              .fontDesign(.monospaced)
              .foregroundStyle(.red)
              .textSelection(.enabled)
          }

          if !job.lines.isEmpty {
            ScrollViewReader { proxy in
              ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                  ForEach(Array(job.lines.enumerated()), id: \.offset) { index, line in
                    Text(line)
                      .font(.system(.subheadline, design: .monospaced))
                      .fontDesign(.monospaced)
                      .textSelection(.enabled)
                      .frame(maxWidth: .infinity, alignment: .leading)
                      .id(index)
                  }
                }
              }
              .frame(height: 242)
              .onChange(of: job.lines.count) { _, count in
                proxy.scrollTo(count - 1, anchor: .bottom)
              }
            }
          }
        }
      }
    }
  }
}
