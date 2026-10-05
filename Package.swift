// swift-tools-version: 6.2
import PackageDescription

// mlx-flashvsr-swift — FlashVSR v1.1 (OpenImagingLab, Apache-2.0 code + weights) one-step streaming diffusion video
// super-resolution (×4) on MLX: a Wan2.1-1.3B-shaped DiT with locality-constrained sparse attention, a causal LQ
// projector and a conditioned tiny decoder (TCDecoder). Generative: recommended for LIVE ACTION, not anime/graphics.
// Weights: mlx-community/FlashVSR-v1.1-bf16 (default) and -fp32 (parity).
//   • FlashVSRMLX — engine-agnostic core: the three networks (isomorphic to upstream), the streaming chunk driver
//     (`FlashVSRPipeline` whole-clip, `FlashVSRStream` frame-in/frame-out) and the PIL-bicubic LQ preparation
//   • MLXFlashVSR — the MLXEngine `videoUpscale` ModelPackage (frame-stream-native decode → stream → HEVC encode)
//   • flashvsr-smoke — S0/S1/E2E parity gates against the upstream CPU-fp32 goldens, the clip runner, memory and time
let package = Package(
    name: "mlx-flashvsr-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "FlashVSRMLX", targets: ["FlashVSRMLX"]),
        .library(name: "MLXFlashVSR", targets: ["MLXFlashVSR"]),
        .executable(name: "flashvsr-smoke", targets: ["FlashVSRSmoke"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.63.0"),
        .package(url: "https://github.com/xocialize/frame-stream-native.git", from: "0.4.0"),
    ],
    targets: [
        .target(
            name: "FlashVSRMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXFlashVSR",
            dependencies: [
                "FlashVSRMLX",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "FrameStreamNative", package: "frame-stream-native"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "FlashVSRSmoke",
            dependencies: [
                "FlashVSRMLX",
                "MLXFlashVSR",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "FlashVSRMLXTests",
            dependencies: [
                "FlashVSRMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXFlashVSRTests",
            dependencies: [
                "MLXFlashVSR",
                "FlashVSRMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
