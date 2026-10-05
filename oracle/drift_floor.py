#!/usr/bin/env python
"""drift_floor.py — upstream FlashVSR's OWN sensitivity: how far two correct runs of the identical code part ways.

The LCSA block selection is a hard top-k over softmaxed block scores, so a perturbation far below any meaningful
precision can flip a near-tie 128×128 block, and a flip moves the output by ~1e-2. An E2E parity bar must therefore
be relative to this floor, not absolute (the AnimeSR-v1 lesson, AB-L-0194).

Runs upstream (CPU, fp32, verbatim, the same block-sparse stand-in) on a golden's inputs, then:
  nudge    the LQ input + 1e-6 · Rademacher noise (≈ the port's LQ-token error)
  threads  torch.set_num_threads(1) (a different summation order, same input)
and reports PSNR (peak 1 on [-1, 1]) and max-abs of each against the golden's out.frames, plus the final latents.

usage: FLASHVSR_MIRROR=<mirror> python drift_floor.py <golden.safetensors> <out.json>
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dump_goldens as dg  # noqa: E402  (mirror path setup + the shadowed block-sparse kernel)
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def build():
    pipe = dg.FlashVSRTinyPipeline(device="cpu", torch_dtype=torch.float32)
    dit = dg.WanModel(**dg.WAN_1_3B)
    dit.load_state_dict(load_file(os.path.join(dg.W, "diffusion_pytorch_model_streaming_dmd.safetensors")))
    pipe.dit = dit.eval()
    pipe.dit.LQ_proj_in = dg.Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    pipe.dit.LQ_proj_in.load_state_dict(torch.load(os.path.join(dg.W, "LQ_proj_in.ckpt"), map_location="cpu"))
    pipe.dit.LQ_proj_in.float()
    pipe.TCDecoder = dg.build_tcdecoder(new_channels=[512, 256, 128, 128], device="cpu", dtype=torch.float32,
                                        new_latent_channels=16 + 768)
    pipe.TCDecoder.load_state_dict(torch.load(os.path.join(dg.W, "TCDecoder.ckpt"), map_location="cpu"), strict=False)
    pipe.init_cross_kv(context_tensor=torch.load(os.path.join(dg.W, "posi_prompt.pth"), map_location="cpu"))
    return pipe


def run(pipe, lq, noise, meta):
    pipe.generate_noise = lambda shape, seed=None, device="cpu", dtype=torch.float32: noise.clone()
    F, (w, h) = lq.shape[2], map(int, meta["size"].split("x"))
    return pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0, LQ_video=lq,
                num_frames=F, height=h, width=w, is_full_block=False, if_buffer=True,
                topk_ratio=float(meta["topk_ratio"]), kv_ratio=3.0, local_range=int(meta["local_range"]),
                color_fix=True)


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def main():
    golden, out = sys.argv[1], sys.argv[2]
    torch.set_grad_enabled(False)
    with safe_open(golden, "pt") as f:
        meta = f.metadata()
        lq, noise, ref = f.get_tensor("input.lq"), f.get_tensor("input.noise"), f.get_tensor("out.frames")
    pipe = build()
    res = {"golden": os.path.basename(golden), "meta": meta}
    base = run(pipe, lq, noise, meta)
    res["rerun"] = dict(psnr=psnr(base, ref), max_abs=(base - ref).abs().max().item())
    g = torch.Generator("cpu").manual_seed(7)
    nudge = (torch.randint(0, 2, lq.shape, generator=g).float() * 2 - 1) * 1e-6
    v = run(pipe, lq + nudge, noise, meta)
    res["nudge_1e-6"] = dict(psnr=psnr(v, ref), max_abs=(v - ref).abs().max().item(),
                             per_frame=[psnr(v[:, i], ref[:, i]) for i in range(ref.shape[1])])
    n0 = torch.get_num_threads()
    torch.set_num_threads(1)
    v = run(pipe, lq, noise, meta)
    torch.set_num_threads(n0)
    res["threads_1"] = dict(psnr=psnr(v, ref), max_abs=(v - ref).abs().max().item(),
                            per_frame=[psnr(v[:, i], ref[:, i]) for i in range(ref.shape[1])])
    fin = lambda v: None if isinstance(v, float) and math.isinf(v) else v   # strict JSON: identical → null
    clean = {k: ({kk: ([fin(x) for x in vv] if isinstance(vv, list) else fin(vv)) for kk, vv in v.items()}
                 if isinstance(v, dict) and k != "meta" else v) for k, v in res.items()}
    json.dump(clean, open(out, "w"), indent=1, allow_nan=False)
    for k in ("rerun", "nudge_1e-6", "threads_1"):
        print(f"{k:12s} psnr {res[k]['psnr']:7.1f} dB  max_abs {res[k]['max_abs']:.3e}")  # inf = bit-identical


if __name__ == "__main__":
    main()
