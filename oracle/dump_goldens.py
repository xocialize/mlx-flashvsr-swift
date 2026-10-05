#!/usr/bin/env python
"""dump_goldens.py — FlashVSR v1.1 tiny, upstream code VERBATIM on the CPU in fp32, every sub-op recorded.

The pipeline is upstream's own (`FlashVSRTinyPipeline.__call__`, `model_fn_wan_video`, `WanModel`/`DiTBlock`,
`Causal_LQ4x_Proj`, TAEHV `decode_video`, the AdaIN colour fix) from the mirror `setup_mirror.sh` builds
(OpenImagingLab/FlashVSR @ cf910c61). The only substitution is the CUDA block-sparse kernel → `block_sparse_attn.py`
(dense SDPA under the identical block mask). On the CPU upstream's float64 RoPE / embedding run as written.
Noise is injected from a seeded CPU generator so the Swift side can replay it.

Records (NHWC / token-major where the Swift port consumes them; everything fp32):
  input.lq            (1, 3, F, H, W) in [-1, 1] — after upstream's bicubic ×4 + 128-multiple crop
  input.noise         (1, 16, (F−1)/4, H/8, W/8)
  c{k}.x_in           the chunk's latents into model_fn; c{k}.lq.{i} LQ_proj_in outputs (one per layer)
  c{k}.b0.* / c{k}.b29.*  block input / self-attn in / q,k,v (window-reordered) / mask / attn / self-attn out /
                      cross-attn out / block out;  c{k}.cache_k0, cache_v0 (block-0 K/V cache after the chunk)
  c{k}.pred           model_fn output (noise prediction), c{k}.latents (x − pred)
  const.t, const.t_mod, const.context, const.cross_k0/v0
  dec.latents, dec.cond, dec.out (TCDecoder, before colour fix), out.frames (final, [-1, 1])

usage: FLASHVSR_MIRROR=<mirror> python dump_goldens.py <lrFramesDir|procedural> <out.safetensors> [--frames 33]
"""
import argparse, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = os.environ["FLASHVSR_MIRROR"]
sys.path.insert(0, HERE)
sys.path.insert(0, MIRROR)

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

import block_sparse_attn  # noqa: F401  (shadows the CUDA package)
import diffsynth.models.wan_video_dit as _dit
import diffsynth.pipelines.flashvsr_tiny as _tiny
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.flashvsr_tiny import FlashVSRTinyPipeline
from fv_tcdecoder import build_tcdecoder
from fv_utils import Causal_LQ4x_Proj

W = os.path.join(MIRROR, "w")
WAN_1_3B = dict(dim=1536, in_dim=16, ffn_dim=8960, out_dim=16, text_dim=4096, freq_dim=256, eps=1e-6,
                patch_size=(1, 2, 2), num_heads=12, num_layers=30)
REC = {}
STATE = {"chunk": -1, "block": None}


def rec(k, t):
    if isinstance(t, torch.Tensor):
        REC[k] = t.detach().float().contiguous().clone()


def procedural(n, w, h, seed=3):
    rng = np.random.default_rng(seed)
    H, W_ = h + 3 * n, w + 3 * n
    yy, xx = np.mgrid[0:H, 0:W_].astype(np.float32)
    img = np.stack([xx / W_, yy / H, 0.5 + 0.5 * np.sin(xx / 4.0) * np.cos(yy / 6.0)], -1)
    img[((xx // 7 + yy // 9) % 2) == 0] *= 0.4
    img[::5] = [0.05, 0.05, 0.1]
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    img = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
    return [Image.fromarray(img[i * 3 // 2: i * 3 // 2 + h, i * 3: i * 3 + w]) for i in range(n)]


def prepare(imgs, scale=4):
    w0, h0 = imgs[0].size
    sW, sH = int(round(w0 * scale)), int(round(h0 * scale))
    tW, tH = (sW // 128) * 128, (sH // 128) * 128
    seq = imgs + [imgs[-1]] * 4
    F = ((len(seq) - 1) // 8) * 8 + 1
    frames = []
    for img in seq[:F]:
        up = img.convert("RGB").resize((sW, sH), Image.BICUBIC)
        l, t = (sW - tW) // 2, (sH - tH) // 2
        up = up.crop((l, t, l + tW, t + tH))
        x = torch.from_numpy(np.asarray(up, np.uint8)).float()
        frames.append(x.permute(2, 0, 1) / 255.0 * 2.0 - 1.0)
    return torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0), tH, tW, F


def install_taps(pipe):
    dit = pipe.dit
    # model_fn (per chunk)
    orig_fn = _tiny.model_fn_wan_video

    def fn(dit_, x, *a, LQ_latents=None, pre_cache_k=None, pre_cache_v=None, **kw):
        STATE["chunk"] += 1
        k = STATE["chunk"]
        rec(f"c{k}.x_in", x)
        for i, l in enumerate(LQ_latents or []):
            rec(f"c{k}.lq.{i}", l)
        out, ck, cv = orig_fn(dit_, x, *a, LQ_latents=LQ_latents, pre_cache_k=pre_cache_k,
                              pre_cache_v=pre_cache_v, **kw)
        rec(f"c{k}.pred", out)
        rec(f"c{k}.cache_k0", ck[0])
        rec(f"c{k}.cache_v0", cv[0])
        return out, ck, cv

    _tiny.model_fn_wan_video = fn
    # attention kernel call (q/k/v window-reordered + block mask) for the tapped blocks
    orig_fa = _dit.flash_attention

    def fa(q, k, v, num_heads, compatibility_mode=False, attention_mask=None, return_KV=False):
        out = orig_fa(q, k, v, num_heads, compatibility_mode, attention_mask, return_KV)
        b = STATE["block"]
        if attention_mask is not None and b is not None:
            p = f"c{STATE['chunk']}.b{b}"
            rec(f"{p}.q", q); rec(f"{p}.k", k); rec(f"{p}.v", v)
            REC[f"{p}.mask"] = attention_mask.detach().to(torch.float32).contiguous().clone()
            rec(f"{p}.attn", out)
        return out

    _dit.flash_attention = fa
    for bi in (0, 29):
        blk = dit.blocks[bi]

        def pre(m, args, bi=bi):
            STATE["block"] = bi
            rec(f"c{STATE['chunk']}.b{bi}.in", args[0])

        def post(m, args, out, bi=bi):
            rec(f"c{STATE['chunk']}.b{bi}.out", out[0] if isinstance(out, tuple) else out)
            STATE["block"] = None

        blk.register_forward_pre_hook(pre)
        blk.register_forward_hook(post)
        def sa_hook(m, a, o, bi=bi):   # a forward hook that RETURNS a value replaces the output — return None
            rec(f"c{STATE['chunk']}.b{bi}.sa_in", a[0])
            rec(f"c{STATE['chunk']}.b{bi}.sa_out", o[0] if isinstance(o, tuple) else o)

        def ca_hook(m, a, o, bi=bi):
            rec(f"c{STATE['chunk']}.b{bi}.ca_in", a[0])
            rec(f"c{STATE['chunk']}.b{bi}.ca_out", o)

        blk.self_attn.register_forward_hook(sa_hook)
        blk.cross_attn.register_forward_hook(ca_hook)
    # decoder
    orig_dec = pipe.TCDecoder.decode_video

    def dec(x, parallel=True, show_progress_bar=False, cond=None):
        rec("dec.latents", x)
        rec("dec.cond", cond)
        out = orig_dec(x, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)
        rec("dec.out", out)
        return out

    pipe.TCDecoder.decode_video = dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--size", default="128x96")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--topk-ratio", type=float, default=None, help="override 2·768·1280/(H·W) — to exercise real top-k")
    ap.add_argument("--local-range", type=int, default=11)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.src == "procedural":
        w, h = map(int, a.size.split("x"))
        imgs = procedural(a.frames, w, h)
    else:
        imgs = [Image.open(p) for p in sorted(glob.glob(os.path.join(a.src, "f*.png")))[: a.frames]]
    dev, dt = "cpu", torch.float32
    pipe = FlashVSRTinyPipeline(device=dev, torch_dtype=dt)
    dit = WanModel(**WAN_1_3B)
    from safetensors.torch import load_file
    m, u = dit.load_state_dict(load_file(os.path.join(W, "diffusion_pytorch_model_streaming_dmd.safetensors")), strict=False)
    assert not m and not u, (m, u)
    pipe.dit = dit.eval()
    pipe.dit.LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    pipe.dit.LQ_proj_in.load_state_dict(torch.load(os.path.join(W, "LQ_proj_in.ckpt"), map_location="cpu"), strict=True)
    pipe.dit.LQ_proj_in.float()
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device=dev, dtype=dt, new_latent_channels=16 + 768)
    tc = pipe.TCDecoder.load_state_dict(torch.load(os.path.join(W, "TCDecoder.ckpt"), map_location="cpu"), strict=False)
    print("TCDecoder load:", tc)
    ctx = torch.load(os.path.join(W, "posi_prompt.pth"), map_location="cpu")
    pipe.init_cross_kv(context_tensor=ctx)
    rec("const.context", ctx)
    rec("const.t", pipe.t)
    rec("const.t_mod", pipe.t_mod)
    rec("const.cross_k0", dit.blocks[0].cross_attn.cache_k)
    rec("const.cross_v0", dit.blocks[0].cross_attn.cache_v)

    LQ, th, tw, F = prepare(imgs)
    rec("input.lq", LQ)
    g = torch.Generator("cpu").manual_seed(a.seed)
    noise = torch.randn((1, 16, (F - 1) // 4, th // 8, tw // 8), generator=g, dtype=torch.float32)
    rec("input.noise", noise)
    pipe.generate_noise = lambda shape, seed=None, device="cpu", dtype=torch.float32: noise.clone()
    install_taps(pipe)
    topk_ratio = a.topk_ratio if a.topk_ratio is not None else 2.0 * 768 * 1280 / (th * tw)
    video = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0, LQ_video=LQ,
                 num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True, topk_ratio=topk_ratio,
                 kv_ratio=3.0, local_range=a.local_range, color_fix=True)
    rec("out.frames", video)
    meta = dict(frames_in=str(len(imgs)), F=str(F), size=f"{tw}x{th}", topk_ratio=f"{topk_ratio:.6f}",
                kv_ratio="3.0", local_range=str(a.local_range), seed=str(a.seed), chunks=str(STATE["chunk"] + 1),
                upstream="OpenImagingLab/FlashVSR@cf910c61 + HF FlashVSR-v1.1", device="cpu fp32")
    save_file(REC, a.out, metadata=meta)
    tot = sum(v.numel() for v in REC.values()) * 4 / 1e6
    print(f"{len(REC)} tensors ({tot:.0f} MB) -> {a.out}; {meta}")
    for k in sorted(REC):
        print(f"  {k:22s} {tuple(REC[k].shape)}")


if __name__ == "__main__":
    main()
