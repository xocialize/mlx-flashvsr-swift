#!/usr/bin/env python
"""model_card.py — write the Hub model card (README.md) into a converted lane directory.

usage: python model_card.py <laneDir> --lane bf16|fp32
"""
import argparse, os

CARD = """---
library_name: mlx
license: apache-2.0
license_link: https://github.com/OpenImagingLab/FlashVSR/blob/main/LICENSE
pipeline_tag: video-to-video
base_model: JunhaoZhuang/FlashVSR-v1.1
tags:
  - mlx
  - apple-silicon
  - video-super-resolution
  - super-resolution
  - video-to-video
  - flashvsr
---

# mlx-community/FlashVSR-v1.1-{lane}

**FlashVSR v1.1** ([JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1),
[OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR), [arXiv 2510.12747](https://arxiv.org/abs/2510.12747)) —
one-step, streaming, diffusion-based ×4 video super-resolution — converted to MLX ({lane_desc}) for the Swift/MLX port
[`xocialize/mlx-flashvsr-swift`](https://github.com/xocialize/mlx-flashvsr-swift).

> ### ⚠️ Recommended for live-action footage — not anime, cartoons or graphics
> FlashVSR is **generative**: it invents plausible detail at the target resolution rather than reconstructing it.
> On live action that is its strength — resolution-limited faces come back with real photographic detail where
> faithful upscalers give a blur. On **drawn content (anime, cartoons, motion graphics, UI, text overlays)** it renders flat
> colour and clean line art as photographic texture and pushes the drawing toward realism; use a faithful upscaler
> (e.g. Real-ESRGAN anime models) there instead. It is also aggressive with **defocus**: bokeh and soft backgrounds
> can come back as crisp invented texture.

## Files

| file | component | params | notes |
|---|---|---|---|
| `dit_{lane}.safetensors` | Wan2.1-1.3B-shaped DiT, DMD-distilled to one step at t = 1000 | 1,418,996,800 | `patch_embedding` Conv3d (O,D,H,W,I) |
| `lq_proj_{lane}.safetensors` | `Causal_LQ4x_Proj` — the LQ conditioning path | 287,845,888 | Conv3d (O,D,H,W,I); RMS gammas (C) |
| `tcdecoder_{lane}.safetensors` | TCDecoder — the LQ-conditioned tiny decoder (TAEHV-wide) | 45,338,371 | Conv2d (O,H,W,I) |
| `prompt_{lane}.safetensors` | the fixed prompt context (1, 512, 4096) | — | no text encoder at runtime |
| `config.json` | architecture, pipeline defaults, key contract, provenance (source sha256s) | | |

Keys are upstream's verbatim; only conv layouts change (channels-last). Produced by the port's
`oracle/convert_weights.py --lane {lane}` from `JunhaoZhuang/FlashVSR-v1.1 @ 27561b18`. No Wan VAE and no umT5
are needed: the tiny pipeline decodes with the TCDecoder and conditions on the fixed prompt tensor.

**Lanes.** `mlx-community/FlashVSR-v1.1-bf16` is the production lane (3.5 GB): every tensor rounded to bf16 —
bit-identical to casting the fp32 lane to bf16 at load (verified over all 899 parameters).
`mlx-community/FlashVSR-v1.1-fp32` is the parity / reference lane (7.0 GB): the DiT and decoder as released, the
LQ projector and prompt (released in bf16) upcast exactly.

## Parity with upstream

Gated against upstream's own code executed verbatim (CPU, fp32; 512×384 ×33 frames, two goldens — upstream's
default sparsity and a strongly sparse setting with empty query blocks):

- every component at isolated relative error ≤ 1e-5 (DiT blocks fed the golden block inputs, LQ projector, decoder);
- end to end at **upstream's own sensitivity floor**: FlashVSR's locality-constrained sparse attention selects
  128×128 blocks by a hard top-k, so a 1e-6 change to the input flips near-tied blocks — upstream itself moves to
  52.9 dB under such a nudge, and so does this port (identical max-abs error). The port's E2E is 100–113 dB where
  upstream's floor is high.

The block-sparse attention (upstream: mit-han-lab's CUDA `Block-Sparse-Attention`) is implemented as a Metal kernel
that computes only the selected blocks — the LCSA path upstream's card asks third-party ports not to drop.

## Quality

×4 (320×192 → 1280×768), full-reference against the native 1280×768 source, against upstream on PyTorch-MPS:

| clip | upstream (torch) SSIMULACRA2 | this port, bf16 | PSNR torch / port |
|---|---|---|---|
| live action | −17.0 | −16.3 | 27.49 / 27.59 |
| anime pan | −45.7 | −44.1 | 25.40 / 25.48 |

bf16 sits inside the fp32 seed-to-seed spread on every metric. (The anime row is why the recommendation above
exists: the score reflects invented texture, with an image gradient twice the reference's.)

## Memory

Streaming keeps memory independent of clip length; it scales with output pixels per frame. Measured through
MLXEngine on an M5 Max (process peak `phys_footprint`, bf16): 640×384 out 9.2 GB · 1280×768 out 19.1 GB ·
1920×1152 out 33.7 GB. fp32 at 1280×768: 34.4 GB.

## Use with mlx-flashvsr-swift

```swift
import MLXServeCore, MLXToolKit, MLXFlashVSR
let engine = MLXServeEngine()
let id = try await engine.register(FlashVSRUpscalePackage.registration,
                                   configuration: FlashVSRConfiguration(precision: .{precision}))
let out = try await engine.run(VideoUpscaleRequest(video: video, scale: 4), package: id)   // ×4 or ×2
```

The engine downloads this repo into its model store on first use. Or drive the core directly:

```swift
import FlashVSRMLX
let pipe = try FlashVSRPipeline.load(directory: laneDir, dtypes: .{dtypes})
let stream = FlashVSRStream(pipeline: pipe, height: 768, width: 1280)   // ×4 bicubic LQ size, multiples of 128
for frame in lqFrames { for out in try stream.push(frame) { /* … */ } }
for out in try stream.finish() { /* one output per input frame */ }
```

## Licences and provenance

Apache-2.0: FlashVSR code and v1.1 weights (OpenImagingLab / Junhao Zhuang et al.); the DiT is the Wan2.1
architecture (Apache-2.0); the TCDecoder derives from TAEHV (MIT). The training set (VSR-120K) is described by its
authors but not released; this re-host takes the declared weight licence as governing.

```bibtex
@article{{zhuang2025flashvsr,
  title={{FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}},
  author={{Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan}},
  journal={{arXiv preprint arXiv:2510.12747}},
  year={{2025}}
}}
```
"""

LANES = {
    "bf16": dict(lane_desc="bf16, the production lane", precision="bf16", dtypes="production"),
    "fp32": dict(lane_desc="fp32, the parity / reference lane", precision="fp32", dtypes="parity"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--lane", choices=list(LANES), required=True)
    a = ap.parse_args()
    with open(os.path.join(a.dir, "README.md"), "w") as f:
        card = CARD.replace("{{", "{").replace("}}", "}")
        for k, v in dict(lane=a.lane, **LANES[a.lane]).items():   # plain replacement: the card holds code braces
            card = card.replace("{" + k + "}", v)
        f.write(card)
    print("wrote", os.path.join(a.dir, "README.md"))


if __name__ == "__main__":
    main()
