# mlx-flashvsr-swift

**FlashVSR v1.1** — one-step streaming diffusion video super-resolution (×4) — on Apple silicon with
[MLX Swift](https://github.com/ml-explore/mlx-swift). A port of
[OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (`cf910c61`, Apache-2.0) and its released v1.1
weights ([JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1), Apache-2.0), with an
[MLXEngine](https://github.com/xocialize/mlx-engine-swift) `videoUpscale` package.

> ### ⚠️ Recommended for live-action footage — not anime, cartoons or graphics
> FlashVSR is **generative**: a Wan2.1-1.3B-shaped DiT, distilled to a single step, invents plausible detail at the
> target resolution. On live action that is the point — resolution-limited faces come back with real photographic
> detail where faithful upscalers give a blur. On **drawn content (anime, cartoons, motion graphics, UI, text
> overlays)** it renders flat colour and clean line art as photographic texture and pushes the drawing toward
> realism — use a faithful upscaler there. It is also aggressive with **defocus**: bokeh and soft backgrounds can
> come back as crisp invented texture.

It streams — a 25-frame start, then 8 frames per chunk — with FlashVSR's locality-constrained block-sparse
self-attention (a Metal kernel here, standing in for upstream's CUDA one), a causal LQ projector and a small
LQ-conditioned decoder. No text encoder and no Wan VAE run at inference.

| Product | What |
|---|---|
| `FlashVSRMLX` | The engine-agnostic core: `FlashVSRDiT`, `LQProjector`, `TCDecoder` (isomorphic to upstream, upstream's checkpoint keys), `FlashVSRPipeline` (whole clip, upstream's call shape), `FlashVSRStream` (frame in / frame out, every frame kept), `FlashVSRPreprocess` (the PIL-bicubic LQ preparation the model was trained on) |
| `MLXFlashVSR` | The MLXEngine `videoUpscale` `ModelPackage`: AVFoundation decode ([frame-stream-native](https://github.com/xocialize/frame-stream-native)) → stream → HEVC encode; ×4 native and ×2 |
| `flashvsr-smoke` | Parity gates (S0 keys, S1 sub-op taps, floor-relative E2E), the frame-directory runner, the real-engine drive, attention probes |

## Weights

Two precision lanes on the Hub, converted by `oracle/convert_weights.py`:

| repo | size | lane |
|---|---|---|
| [`mlx-community/FlashVSR-v1.1-bf16`](https://huggingface.co/mlx-community/FlashVSR-v1.1-bf16) | 3.5 GB | production (default) — bit-identical to casting the fp32 lane at load |
| [`mlx-community/FlashVSR-v1.1-fp32`](https://huggingface.co/mlx-community/FlashVSR-v1.1-fp32) | 7.0 GB | parity / reference |

Through MLXEngine the package downloads its lane into the engine's model store on first use. To work offline, point
`FlashVSRConfiguration.weightsDirectory` (or `FlashVSRPipeline.load(directory:)`) at a downloaded or locally
converted lane directory.

## Install

```swift
.package(url: "https://github.com/xocialize/mlx-flashvsr-swift", from: "0.2.0"),
// products: "FlashVSRMLX" (core) and/or "MLXFlashVSR" (engine package)
```

macOS 26, Apple silicon. Build with `--build-system swiftbuild` (the Metal library MLX needs ships with it). mlx-swift
0.31.2 or newer — verified on 0.31.6 and 0.32.3 (use ≥ 0.1.1: 0.1.0 needed a 0.32-only API despite declaring 0.31).

## Use

Through the engine:

```swift
import MLXServeCore, MLXToolKit, MLXFlashVSR
let engine = MLXServeEngine()
let id = try await engine.register(FlashVSRUpscalePackage.registration, configuration: FlashVSRConfiguration())
let out = try await engine.run(VideoUpscaleRequest(video: video, scale: 4), package: id) as! VideoUpscaleResponse
```

Or the core, frame by frame:

```swift
import FlashVSRMLX
let pipe = try FlashVSRPipeline.load(directory: laneDir, dtypes: .production)   // bf16
let stream = FlashVSRStream(pipeline: pipe, height: 768, width: 1280)            // LQ size, multiples of 128
for frame in lqFrames {               // (H, W, 3) in [−1, 1], already ×4 bicubic (FlashVSRPreprocess.bicubic8)
    for out in try stream.push(frame) { /* (H, W, 3) in [−1, 1] */ }
}
for out in try stream.finish() { /* the tail: one output per pushed frame */ }
```

## Status

Parity, quality, memory and open items are in [`PORTING-SPEC.md`](PORTING-SPEC.md). In short:

- **Parity.** Every component matches upstream at isolated relative error ≤ 1e-5. End to end the port sits at
  upstream's own sensitivity floor: the hard top-k block selection makes a 1e-6 input change move upstream itself
  by 52.9 dB on a sparse golden, and the port lands on exactly that.
- **Quality.** On ×4 full-reference clips (320×192 → 1280×768) the port lands in upstream's band — live action
  SSIMULACRA2 −16.3 vs torch −17.0 — and bf16 is inside the fp32 seed spread.
- **Memory** (through MLXEngine, bf16, process peak): 19.1 GB at 1280×768 output, 33.7 GB at 1920×1152. Streaming
  keeps it independent of clip length; the package declares the measured scaling and refuses outputs above
  1920×1152 (bf16) / 1280×768 (fp32) before loading.
- **Speed** (idle M5 Max, bf16, s per output frame, pipeline only): **0.50 at 1280×768 and 1.33 at 1920×1152** —
  1.6× and 2.3× upstream's PyTorch-MPS run on the same machine (0.80 / 3.09). Through MLXEngine end to end (decode,
  upscale, HEVC encode): 0.72 and ~2.1. At 1920×1152 the Metal block-sparse kernel is 2.2× the dense path; at
  1280×768, where FlashVSR keeps 50–67 % of key blocks, dense is as fast, and the default picks per call.

## Licence

Apache-2.0 (the port derives from upstream's Apache-2.0 code). The weights are Apache-2.0 from their authors; see
[`NOTICE`](NOTICE) for attribution (FlashVSR, DiffSynth-Studio, Wan2.1, TAEHV).

```bibtex
@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}
```
