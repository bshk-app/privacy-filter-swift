// swift-tools-version: 6.0
import PackageDescription

// `pf` — Swift CLI redactor (design: docs/plans/2026-06-19-swift-redaction-cli-design.md).
// Milestone 0 (this file): foundation spike — prove mlx-swift builds and loads the
// privacy-filter weights on Metal. Tokenizer + forward + CLI come in later milestones.
//
// NOTE: dependency versions are best-effort; bump on the first `swift build` if SwiftPM
// resolves something newer. mlx-swift product names: MLX, MLXNN, MLXFast, MLXRandom.
let package = Package(
    name: "pf",
    platforms: [.macOS(.v14)],
    dependencies: [
        // ≥ 0.31.4: includes MLX 0.31.2's thread-safety for independent computation (design §3).
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.4"),
        // Tokenizer (M1). Product "Tokenizers" pulls only Hub (no Models/Generation).
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        // CLI flags (C2). AsyncParsableCommand supports the async tokenizer load.
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        // M2 / Phase C (C1): the parity-proven forward, extracted into a reusable
        // library so both `pf-parity` and the future `pf` CLI share one model. Imports
        // MLX → must build via run.sh (xcodebuild), not `swift build`.
        .target(
            name: "PFModel",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/PFModel"
        ),
        .executableTarget(
            name: "pf-parity",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                "PFModel",
            ],
            path: "Sources/pf-parity"
        ),
        // C2/C3: the real `pf` streaming-redactor CLI. Wires the MLX model (PFModel),
        // the swift-transformers tokenizer (Tokenizers), the pure redaction core (PFCore),
        // and swift-argument-parser. Imports MLX (via PFModel) → must build via run.sh
        // (xcodebuild); plain `swift build`/`swift run` cannot compile the Metal lib.
        .executableTarget(
            name: "pf",
            dependencies: [
                "PFModel",
                "PFCore",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/pf"
        ),
        // M3 / Phase B: pure-Swift redaction core (BIOES→spans, stable-token redactor).
        // NO MLX, NO Tokenizers — so `swift test` builds & runs without Metal (fast).
        .target(name: "PFCore", path: "Sources/PFCore"),
        .testTarget(
            name: "PFCoreTests",
            dependencies: ["PFCore"],
            path: "Tests/PFCoreTests"
        )
    ]
)
