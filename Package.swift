// swift-tools-version: 6.0
// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import PackageDescription

let package = Package(
  name: "ishizuki",
  platforms: [
    .macOS(.v15),
    .iOS(.v18),
  ],
  products: [
    .library(name: "IshizukiKit", targets: ["IshizukiKit"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.6"),
    .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.1"),
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
    .testTarget(
      name: "IshizukiKitTests",
      dependencies: ["IshizukiKit"],
      resources: [.copy("Fixtures")]
    ),
  ]
)
