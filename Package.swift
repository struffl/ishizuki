// swift-tools-version: 6.4
// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import PackageDescription

let package = Package(
  name: "ishizuki",
  platforms: [
    .macOS(.v26),
    .iOS(.v26),
  ],
  products: [
    .library(name: "IshizukiKit", targets: ["IshizukiKit"]),
    .library(name: "IshizukiAgent", targets: ["IshizukiAgent"]),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.6"),
    .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.1"),
    .package(url: "https://github.com/1amageek/SwiftAgent.git", from: "2.0.1"),
  ],
  targets: [
    .target(
      name: "IshizukiKit",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "Jinja", package: "swift-jinja"),
      ]
    ),
    // The agent loop and nothing else, so the engine library stays clear of SwiftAgent's
    // dependency graph and a build that wants only inference can leave it out.
    .target(
      name: "IshizukiAgent",
      dependencies: [
        "IshizukiKit",
        .product(name: "SwiftAgent", package: "SwiftAgent"),
      ]
    ),
    .testTarget(
      name: "IshizukiKitTests",
      dependencies: ["IshizukiKit"],
      resources: [.copy("Fixtures")]
    ),
  ]
)
