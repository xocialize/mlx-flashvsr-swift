#!/usr/bin/env python
"""convert_weights.py — FlashVSR v1.1 checkpoints → one MLX-layout precision lane (keys unchanged, convs channels-last).

The published lanes (mlx-community's `<UpstreamRepoName>-<precision>` grammar, one repo per lane) are exactly this
script's output:
  mlx-community/FlashVSR-v1.1-fp32  --lane fp32   the parity lane: DiT + TCDecoder as released (fp32); the LQ projector
                                                  and the prompt tensor, released in bf16, upcast to fp32 (exact)
  mlx-community/FlashVSR-v1.1-bf16  --lane bf16   the production lane: every tensor rounded to bf16 (round-to-nearest-
                                                  even, the same rounding MLX applies when the fp32 lane is cast at load
                                                  — the two give bit-identical bf16 parameters)

Files per lane (<lane> = fp32 | bf16):
  dit_<lane>.safetensors        825 tensors, 1,418,996,800 params (Wan2.1-1.3B-shaped DiT; patch_embedding OIDHW → ODHWI)
  lq_proj_<lane>.safetensors    8 tensors, 287,845,888 params (Causal_LQ4x_Proj; Conv3d OIDHW → ODHWI; RMS gammas → [C])
  tcdecoder_<lane>.safetensors  66 tensors, 45,338,371 params (TAEHV-wide; Conv2d OIHW → OHWI)
  prompt_<lane>.safetensors     the fixed prompt context (1, 512, 4096) — no text encoder at runtime
  config.json                   architecture, pipeline defaults, key contract, provenance (source sha256s)

Weights: HF JunhaoZhuang/FlashVSR-v1.1 (Apache-2.0). Every source file's sha256 goes into each output's metadata.
usage: FLASHVSR_MIRROR=<mirror> python convert_weights.py <outDir> --lane fp32|bf16
"""
import argparse, hashlib, json, os

import torch
from safetensors.torch import load_file, save_file

M = os.environ["FLASHVSR_MIRROR"]
W = os.path.join(M, "w")
UPSTREAM = dict(code="OpenImagingLab/FlashVSR@cf910c61a60733e610e9c6e8b607f80c3a6c202b",
                weights="JunhaoZhuang/FlashVSR-v1.1@27561b186ded3402d7c975f4fd722e2885b6135f")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cl(t):   # channels-last for conv weights
    if t.ndim == 5:
        return t.permute(0, 2, 3, 4, 1).contiguous()
    if t.ndim == 4:
        return t.permute(0, 2, 3, 1).contiguous()
    return t.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--lane", choices=["fp32", "bf16"], required=True)
    a = ap.parse_args()
    dt = torch.float32 if a.lane == "fp32" else torch.bfloat16
    os.makedirs(a.out, exist_ok=True)
    sources, report = {}, {}

    def write(name, tensors, src):
        sources[os.path.basename(src)] = sha(src)
        tensors = {k: v.to(dt).contiguous() for k, v in tensors.items()}
        path = os.path.join(a.out, f"{name}_{a.lane}.safetensors")
        save_file(tensors, path, metadata=dict(layout="channels-last convs", lane=a.lane,
                                                source=os.path.basename(src), source_sha256=sources[os.path.basename(src)],
                                                upstream=UPSTREAM["weights"]))
        report[name] = dict(file=os.path.basename(path), tensors=len(tensors),
                            params=sum(v.numel() for v in tensors.values()), dtype=a.lane)
        print(name, report[name])

    src = os.path.join(W, "diffusion_pytorch_model_streaming_dmd.safetensors")
    dit = load_file(src)
    write("dit", {k: (cl(v) if k == "patch_embedding.weight" else v) for k, v in dit.items()}, src)

    src = os.path.join(W, "LQ_proj_in.ckpt")
    lq = torch.load(src, map_location="cpu")
    write("lq_proj", {k: (v.reshape(-1) if "gamma" in k else cl(v)) for k, v in lq.items()}, src)

    src = os.path.join(W, "TCDecoder.ckpt")
    write("tcdecoder", {k: cl(v) for k, v in torch.load(src, map_location="cpu").items()}, src)

    src = os.path.join(W, "posi_prompt.pth")
    write("prompt", {"context": torch.load(src, map_location="cpu")}, src)

    cfg = dict(
        model="FlashVSR v1.1 (tiny pipeline: one-step DiT + LQ projector + TCDecoder)",
        lane=a.lane, upstream=UPSTREAM, converter="mlx-flashvsr-swift oracle/convert_weights.py",
        components=report,
        dit=dict(dim=1536, in_dim=16, out_dim=16, ffn_dim=8960, text_dim=4096, freq_dim=256, num_heads=12,
                 head_dim=128, num_layers=30, eps=1e-6, patch_size=[1, 2, 2], timestep=1000,
                 rope="3-D factorised over the 128-wide head: temporal 44 | height 42 | width 42, theta 10000"),
        lq_proj=dict(pixel_unshuffle=[1, 16, 16], hidden=[2048, 3072], kernel=[4, 3, 3], stride=[2, 1, 1],
                     padding="replicate, temporal front 2 (cached 2 frames)", layer_num=1, out_dim=1536),
        tcdecoder=dict(channels=[512, 256, 128, 128], latent_channels=16, cond_channels=768,
                       cond_pixel_unshuffle=[4, 8, 8], frames_to_trim=3, time_upscale=4, space_upscale=8),
        pipeline=dict(scale=4, window=[2, 8, 8], topk_ratio="2·768·1280 / (H·W) at the output size", kv_ratio=3,
                      local_range=11, first_chunk_lq_frames=25, frames_per_chunk=8, color_fix="adain",
                      lq_preprocess="PIL bicubic ×scale on 8-bit RGB, then [0,255] → [-1,1]; H, W multiples of 128"),
        key_contract="upstream checkpoint keys verbatim; Conv3d (O,D,H,W,I), Conv2d (O,H,W,I); RMS gammas (C)",
        sources_sha256=sources,
    )
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print("config.json written")


if __name__ == "__main__":
    main()
