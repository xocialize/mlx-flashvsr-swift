# mlx-flashvsr-swift — porting spec (FlashVSR v1.1 → Swift/MLX)

Reference: **OpenImagingLab/FlashVSR** at **`cf910c61a60733e610e9c6e8b607f80c3a6c202b`** (Apache-2.0) — the v1.1
"tiny" pipeline: `diffsynth/models/wan_video_dit.py` (DiT + LCSA), `diffsynth/pipelines/flashvsr_tiny.py` (streaming
driver, colour fix), `examples/WanVSR/utils/utils.py` (`Causal_LQ4x_Proj`), `examples/WanVSR/utils/TCDecoder.py`
(the conditioned TAEHV decoder), and the fixed prompt tensor `posi_prompt.pth`. Weights: HF
**`JunhaoZhuang/FlashVSR-v1.1`** (Apache-2.0): `diffusion_pytorch_model_streaming_dmd.safetensors`,
`LQ_proj_in.ckpt`, `TCDecoder.ckpt`. Why it was ported and how it compared with SeedVR2:
`mlxengine-forge/Docs/FLASHVSR-SPIKE.md`. Path B, PyTorch → Swift directly with per-sub-op goldens from an oracle
that executes upstream verbatim.

The one substitution the oracle itself makes: mit-han-lab's CUDA `block_sparse_attn_func` → `oracle/block_sparse_attn.py`
(dense SDPA per head under the identical 128×128 block mask; a query block that selects nothing outputs 0). Whether
the CUDA kernel's empty-row convention is 0 is an assumption until a CUDA oracle run confirms it (billable — needs
the operator's go-ahead). The goldens exercise empty rows (13 in the sparse golden's chunk 0).

## What is ported (isomorphic to upstream)

| Swift (`FlashVSRMLX`) | upstream | notes |
|---|---|---|
| `FlashVSRDiT` | `WanModel` (dim 1536, 30 blocks, 12×128 heads, ffn 8960) | keys verbatim (825 tensors); `patch_embedding` Conv3d (1,2,2) channels-last; `text_embedding`/`time_embedding`/`time_projection`/`ffn` as layer arrays so `.0`/`.2` line up |
| `prepare(context:)` | `init_cross_kv` + the t = 1000 constants | the prompt is a fixed tensor (no umT5 at runtime); cross K/V cached once per block |
| `RopeTables` / `ropeApply` | `precompute_freqs_cis_3d` / `rope_apply` | complex128 → (cos, sin) built in Double, applied in fp32; temporal offset 4 + 2k for chunk k |
| `windowPartition` / `windowReverse` | `WindowPartition3D` (2, 8, 8) | 128-token windows ordered (nf, nh, nw), tokens (wf, wh, ww) |
| `localBlockMask` | `build_local_block_mask_shifted_vec_normal_slide` | key block visible iff within [r − R//2, r − R//2 + R − 1] per axis, unclamped |
| `draftBlockMask` | `generate_draft_block_mask` | mean-pooled q/k per window, scores/√d + local −inf, softmax over key blocks, keep STRICTLY above the (k+1)-th largest per (head, query slice) |
| `kernelBlockAttention` (GPU default) / `blockMaskedAttention` (reference, CPU) | `block_sparse_attn_func` | a Metal block-sparse flash kernel over the selected blocks only / dense SDPA per head under the expanded boolean mask; empty query block → 0 in both |
| `FlashVSRDiT.step` | `model_fn_wan_video(is_stream=True)` | LQ tokens added before block 0; window K/V cache trimmed to `kv_ratio` = 3 slices; eval per block |
| `LQProjector` | `Causal_LQ4x_Proj` (layer_num 1) | pixel-unshuffle 16×16, two causal Conv3d (4,3,3)/(2,1,1) with REPLICATE padding and a 2-frame cache, RMS norm (`F.normalize`·√C·γ), SiLU, Linear 3072→1536; first call primes |
| `TCDecoder` | `TAEHV` (widened 512/256/128/128, deepened) | layer indices = upstream's `decoder.N`; MemBlock past = previous INPUT; TGrow fans out depth-first (upstream's work queue); first 3 frames trimmed; 3×3 convs at ≥ 192 in-channels via conv3d kT=1 (AB-L-0163) |
| `FlashVSRPipeline.run` | `FlashVSRTinyPipeline.__call__` (if_buffer, cfg 1) | chunk 0 = 6 latents, then 2 per chunk; decoded per chunk (equal to upstream's decode-at-end: the decoder is sequential with carried state); AdaIN colour fix per frame |

## Phase gates (stamped after the run)

| phase | gate | status |
|---|---|---|
| S0 keys | `flashvsr-smoke s0 oracle/weights` — every component loads `verify: [.all]` | **PASSED** — DiT 825 / 1,418,996,800 · LQ proj 8 / 287,845,888 · TCDecoder 66 / 45,338,371 |
| structure + package | `swift test --build-system swiftbuild` — key contracts, window order, local + draft masks, unshuffle orders, chunk plan, colour fix; the package gates below | **PASSED** 18/18 |
| S1 | `flashvsr-smoke s1 oracle/weights <golden> --e2e --floor <drift.json> [--gpu]` on two goldens (procedural 128×96 → 512×384, 33 frames, 2 chunks): `procedural` (upstream default topk 10, local 11 — dense regime) and `sparse` (topk 0.8, local 3 — real top-k, empty query rows) | see results below |

### S1 results

Isolated = each tapped block fed the golden block input (attention under the golden block mask, so the kernel is
gated apart from the selection rule); chained = the whole step from golden latents + LQ tokens through the port's own
caches.

- **Constants** (t, t_mod, cross K/V): rel ≤ 2.5e-7. **LQ projector** tokens per chunk: rel ≤ 2.5e-6.
- **Isolated blocks 0 and 29, both chunks:** every tap (sa_in, q, k, v, attn, sa_out, ca_in, ca_out, out) rel ≤ 9e-6
  CPU. Block 0 masks identical.
- **Chained:** pred 128.5–130.3 dB, block-0 caches rel ≤ 1.2e-6. **TCDecoder** on golden latents + cond: rel 1.1e-5
  (124–127 dB).
- **E2E** (gated relative to upstream's own floor — see the sensitivity finding):

  | golden | lane | port vs upstream | upstream nudge floor | bar | |
  |---|---|---|---|---|---|
  | sparse (topk 0.8, local 3) | CPU (dense attention) | 100.2 dB | 52.9 dB | 46.9 | PASS |
  | sparse | GPU, gathered attention (TF32 off) | 52.9 dB (max-abs 0.1902) | 52.9 dB (max-abs 0.1902) | 46.9 | PASS |
  | procedural (dense) | CPU (dense attention) | 100.2 dB | 115.7 dB | 90.0 | PASS |
  | procedural | GPU, gathered attention | 111.6 dB | 115.7 dB | 90.0 | PASS |

  (Before the colour-fix reduction fix — finding below — both CPU rows read 92.7 dB. The dense-masked GPU
  attention gave the same E2E numbers as the gathered one.)

- **Query-chunked attention** (`FLASHVSR_ATTN_BUDGET_MB=2` forces one 128-query block per group): GPU E2E unchanged
  at 111.6 dB — the grouping is exact.
- **Gathered block-sparse attention** (the GPU default — only selected key blocks, MLX's fused flash kernel):
  isolated attention rel ≤ 5.6e-6 on both goldens, empty query blocks included; E2E identical to the dense path.
- **Stream vs whole clip** (`FlashVSRStream` pushed the golden's 33 frames, golden noise injected): 33 in → 33 out;
  bit-identical (0.0) to the whole-clip run on the frames both emit, CPU and GPU.

## Quality on real ×4 clips (the spike's cells, the spike's own metric code)

`flashvsr-smoke run` on `X4-*/lr_clean` (49 frames 320×192 → 45 frames 1280×768, upstream's frame bookkeeping),
scored by `bench/x4_quality.py` (SSIMULACRA2 via the same gate tool, PSNR, temporal error, gradient) against the
1280×768 reference. `torch-bf16` = upstream on MPS from the spike. Receipts: `bench/x4_quality.csv`.

| cell | arm | S2 mean (p10) | PSNR | temporal err | grad (ref) |
|---|---|---|---|---|---|
| X4-Wp (anime pan) | torch-bf16 (spike) | −45.66 (−49.10) | 25.40 | 8.79 | 51.1 (26.1) |
| | MLX fp32 seed 0 | −44.28 (−48.08) | 25.49 | 8.71 | 50.6 |
| | MLX fp32 seed 1 | −44.24 (−48.04) | 25.48 | 8.60 | 50.0 |
| | **MLX bf16** | −44.13 (−48.08) | 25.48 | 8.71 | 50.5 |
| | MLX DiT-bf16 | −43.94 (−47.86) | 25.51 | 8.68 | 50.4 |
| X4-Hp (live action) | torch-bf16 (spike) | −17.02 (−22.02) | 27.49 | 6.92 | 28.8 (29.5) |
| | MLX fp32 seed 0 | −16.11 (−22.19) | 27.59 | 6.84 | 28.6 |
| | MLX fp32 seed 1 | −16.88 (−22.92) | 27.54 | 6.89 | 28.9 |
| | **MLX bf16** | −16.36 (−22.57) | 27.59 | 6.84 | 28.5 |
| | MLX DiT-bf16 | −16.06 (−22.51) | 27.61 | 6.83 | 28.4 |

The port lands in upstream's band (slightly better on both cells), and bf16 everywhere sits inside the fp32 seed
spread on every metric → **bf16 is the production default** (`FlashVSRDTypes.production`, the package default);
fp32 is the parity lane. (Large negative SSIMULACRA2 on the anime pan is the generative tier inventing texture — the
spike's finding, not a port artefact; the gradient is twice the reference's.)

## Memory (1280×768 output, smoke runner, GPU)

| config | MLX peak | phys peak (cache uncapped) | notes |
|---|---|---|---|
| fp32, first cut | 44.9 GB | 80.2 GB | decoder built a time step's 4 full-res frames as one lazy graph |
| fp32, eval per decoded frame | 32.9 GB | 66.7 GB | live set: 7 GB weights + 8.5 GB DiT window K/V cache + decoder transients |
| fp32, + `Memory.cacheLimit` 2 GB | 32.9 GB | 33.4 GB | but decode 2.3 → 6.5 s/chunk (fresh allocations every frame); 8 GB cap: 39.5 GB, 4.9 s |
| **bf16** | 17.3 GB | 37.7 GB | |
| DiT-only bf16 | 24.3 GB | 53.4–54.7 GB | |

Per stage (fp32, chunk 0 / chunk 1): LQ projector 9.8 / 19.1 GB, DiT 19.8 / 21.8 GB, decode 29.9 / 32.3 GB (MLX
peak since stage start, so each includes the resident set). Timing receipts in this table's runs are VOID — another
session's LTX-2 job held the GPU at 93–98 %; the clean re-measure is pending. The first clean fp32 run (before the
memory fixes, GPU idle) was **34.7 s for 45 frames, 0.77 s/frame** vs upstream torch-MPS bf16 0.865 s/frame.

## Attention implementations (`FLASHVSR_ATTN=dense|gathered|kernel`)

Upstream calls mit-han-lab's CUDA `block_sparse_attn_func`; the oracle stands in dense SDPA under the block mask.
Three MLX implementations, all gated (isolated attention under the golden mask rel ≤ 6.7e-6 on both goldens, empty
query blocks included; E2E at or above the floor-relative bar):

| path | how | status |
|---|---|---|
| `dense` | per head, dense SDPA under the expanded boolean mask, queries grouped to a 512 MB score budget | the reference and the CPU path |
| `gathered` | per query block, gather the selected K/V blocks (padded) → MLX's fused flash kernel | exact but **2–4× slower** than dense at 1280×768 (interleaved A/B): the gather replicates K/V per query block |
| **`kernel`** (GPU default) | `SparseAttention.swift`: a Metal block-sparse flash kernel — a threadgroup per 32 query rows of a block, walks only the selected key blocks by index (on-GPU argsort, no copies), 16-key tiles through threadgroup memory, 8×8 simdgroup MMAs, fp32 online softmax | exact (fp32 rel 2.4–4.7e-7 vs dense on random probes); chunk-1-shaped probe (12 heads, 7680 × 30720) **3.3× faster than dense at density 0.5, 4.4× at 0.22**; bf16 ×4 quality identical to dense (X4-Hp −16.34 vs −16.36, X4-Wp −44.10 vs −44.13) |

Full-run A/B at 1280×768 bf16 (contended GPU, interleaved, 2 reps): chunk-1 DiT 19.2/22.8 s kernel vs 28.3/33.4 s
dense (1.47×); chunk 0 equal (48/64 s both) — at this size chunk 0 keeps ~67 % of blocks; whole run ~10 % faster.
At 1920×1152 bf16 (same contention, back to back): **chunk-0 DiT 113.7 s kernel vs 244.1 s dense (2.15×), chunk-1
40.4 vs 148.2 s (3.67×), whole run 205.9 vs 451.1 s (2.19×)**, same memory (MLX peak 33.1 GB — the decoder),
no NaN. The topk ratio there is 0.89 → ~22 % of key blocks selected, which is where skipping pays. All of these
wall-times are contended (GPU 94–99 % busy before each run); the ratios are the finding, the absolutes are not.

## The engine package (`MLXFlashVSR`)

`FlashVSRUpscalePackage` — `videoUpscale`, ×4 native and ×2 (the same model on a ×2 bicubic input). C7/C8:
Apache-2.0 weights / Apache-2.0 port. **Recommended for live action only — not anime, cartoons or graphics** (it
renders drawn content as photographic texture; the manifest summary says so). Materialization: one Hub repo per
precision lane, `mlx-community/FlashVSR-v1.1-bf16` (default, 3.5 GB) and `-fp32` (parity, 7.0 GB) — this repo's
`oracle/convert_weights.py --lane` output from `JunhaoZhuang/FlashVSR-v1.1 @ 27561b18` (pinned in each lane's
`config.json`). The bf16 lane is bit-identical to the fp32 lane cast at load (`flashvsr-smoke check-lanes`: 899
parameters, 0 differ); S0 and S1 re-pass on the fp32 lane files exactly as published. The driver PIL-bicubic-upscales each decoded frame, edge-pads to multiples of 128, streams through
`FlashVSRStream` (25-frame start, then 8 per chunk), crops back, and pairs every output with its own source PTS.

**Engine drive** (`flashvsr-smoke engine`, the real `MLXServeEngine`, one clip per process; every frame kept —
49 in → 49 out, 48 → 48; the ±1-frame alignment check passes; ×2 output visibly sharper than bicubic):

| input → output | lane | phys peak | MLX peak |
|---|---|---|---|
| 320×192 → 640×384 (×2) | bf16 | 9.18 GB | 6.86 GB |
| 320×192 → 1280×768 (×4) | bf16 | 19.11 GB | 15.95 GB |
| 480×288 → 1920×1152 (×4) | bf16 | 33.68 GB | 31.37 GB |
| 320×192 → 1280×768 (×4) | fp32 | 34.38 GB | 31.55 GB |

Declared (contract 1.41 `ActivationScaling`, axis `outputPixels` — streaming makes memory independent of clip
length): bf16 resident 3.6 GB, representative activation 19.0 GB (the 1280×768 case), line 4.73 GB + 14.23 KB/px
(the measured 7.45 GB + 11.86 KB/px of phys less the weights, ×1.2), ceiling 1920×1152; fp32 resident 7.1 GB, flat
33.1 GB, ceiling 1280×768 (the parity lane's one measured point). `WorkloadDeclaring` reads the source geometry from
the container's video `tkhd` box (no AVFoundation, synchronous), so the engine refuses an out-of-envelope run before
loading weights. Wall times in these runs are contended (GPU 94–99 % busy with another session's LTX-2 jobs).

`swift test --build-system swiftbuild`: 13 package tests (MAT per precision, local source, partial directory =
missing, licence on both layers, a `.blocking` permissive-only host registers it, footprints + ceilings, the FIT gate,
workload = output pixels from a real AVFoundation-written .mov, CAN-1..3, load refusal, the driver emitting 21 frames
at the 25th push and 27 for 27 in PTS order on a miniature random-weight pipeline, 8-bit pixel-buffer round trip)
+ 7 structure tests.

## Open

- **Clean timing.** Every wall time above was taken while another session held the GPU at 93–99 %; ratios are
  trustworthy (interleaved), absolutes are not. Re-measure s/frame at 1280×768 and 1920×1152 on an idle GPU
  (`run.json` records `gpu_util_before_pct` and thermal state).
- **Kernel tuning.** The block-sparse kernel stages K/V as fp32 in 16-key tiles; bf16 staging (32-key tiles) and
  bf16 simdgroup MMAs (M3+) are the obvious next steps. Chunk 0 at 1280×768 (67 % density) does not yet beat dense.
- **Decoder memory** is now the peak stage (~12 GB above the DiT's resident set at 1920×1152): spatial tiling is
  approximate because MemBlock state carries spatial context across frames.
- **CUDA oracle** for the empty-query-block convention (billable — needs the operator's go-ahead).
- **ForgeCore registration** (a forge-area decision: route to live action only, never anime/graphics — spike §0).

## Findings

- **The LCSA selection is a hard top-k, and the model is as sensitive as that implies.** With the sparse golden on the
  GPU, a 1.3e-6 difference in the LQ tokens (GPU vs CPU arithmetic) grows to 2.4e-5 by block 16, where two key blocks
  tie with their neighbours to four digits (draft probability 4.046e-2 on both runs) and flip; the flips cascade to
  rel 1e-2 by block 29 and 52.9 dB on the final frames. Upstream does EXACTLY the same: `oracle/drift_floor.py` runs
  upstream (CPU fp32) with the LQ input nudged by 1e-6 and lands on 52.9 dB / max-abs 0.1902 — the identical numbers.
  So an absolute E2E bar is meaningless here; the gate is `min(90 dB, upstream nudge floor − 6 dB)` (the AnimeSR-v1
  lesson, AB-L-0194). The dense regime (procedural golden) has a 115.7 dB floor.
- **fp32 softmax underflow decides ties in the dense regime — a mask-level effect only.** When `topk ≥` the row
  length (upstream's default topk_ratio at small sizes), the threshold is the row minimum and the selection drops every
  entry tied at it. Torch's CPU softmax keeps denormal probabilities (a block at p ≈ 1e-40 survives); MLX flushes them
  to 0 (the block ties the minimum and is dropped): 26–29 blocks flip at block 29, all with port-side p = 0. Their E2E
  cost is small — the GPU runs carry the same flips and score 111.6 dB against a 115.7 dB nudge floor. (A first
  reading blamed them for the CPU's 92.7 dB; that was the next finding — AB-L-0199 is amended.)
- **MLX's CPU reduction over the non-innermost axes of channels-last data loses ~1e-3** (AB-L-0200). The AdaIN colour
  fix reduces each (frame, channel) over H·W; on (B, T, H, W, C) the CPU reduction strides past C and moved every
  output pixel by up to 1.2e-3 (65.6 dB vs upstream) — hidden on the gates because the golden's LQ is a transposed
  NCDHW view whose H·W happens to be contiguous (92.7 dB), and exposed when the stream fed contiguous frames. Fixed by
  reducing over a transposed (B, T, C, H·W) contiguous last axis: CPU E2E 92.7 → 100.2 dB, all paths bit-equal.
- **MLX boolean SDPA masks fill holes with `finfo.min`, not −inf** (AB-L-0198): an all-masked query row returns the
  average of every value instead of NaN. Switching the block mask to boolean (4× smaller) took isolated block-29
  attention from rel 5.6e-6 to 0.26 until empty query blocks were zeroed explicitly from the block mask.
- **Debug builds cannot run the CPU gates** — MLX's CPU elementwise kernels (the tanh-GELU, the mask `where`) are
  single-threaded and unoptimised in a debug build: > 10 minutes for chunk 0. Release: 2.5 minutes for S1 end to end.

## Tooling

`flashvsr-smoke` (`swift build -c release --build-system swiftbuild --scratch-path .build-release`, binary at
`.build-release/out/Products/Release/flashvsr-smoke`):

```
flashvsr-smoke s0 <laneDir>                                     # e.g. oracle/weights/fp32
flashvsr-smoke check-lanes <fp32LaneDir> <bf16LaneDir>          # bf16 lane == fp32 lane cast at load, bit for bit
flashvsr-smoke s1 <weightsDir> <golden> [--rel 1e-4] [--chain-rel 1e-3] [--e2e [--floor drift.json]] [--gpu [--tf32]]
flashvsr-smoke diag-mask <weightsDir> <golden> [--gpu]          # where two runs part ways, block by block
flashvsr-smoke diag-stream <weightsDir> <golden> [--gpu]        # run vs run vs stream, colour-fix layouts
flashvsr-smoke probe-attn                                       # kernel vs dense: accuracy (fp32, bf16) + timing
flashvsr-smoke engine <in.mov|mp4> <out.mp4> <weightsDir> [--scale 2|4] [--fp32] [--seed S] [--warmup clip]
flashvsr-smoke run <lrFramesDir> <outDir> <weightsDir> [--dtype|--lq-dtype|--dec-dtype fp32|bf16|fp16] [--limit N]
                   [--seed S] [--topk R] [--local N]
```
Oracle (`oracle/`): `setup_mirror.sh <dir>` (pins + weights) → `convert_weights.py <laneDir> --lane fp32|bf16` →
`model_card.py <laneDir> --lane …` (the Hub card) →
`dump_goldens.py procedural <out> [--topk-ratio 0.8 --local-range 3]` → `drift_floor.py <golden> <out.json>`.
Gate modes set `MLX_ENABLE_TF32=0` before the first MLX op unless `--tf32`. Env: `FLASHVSR_ATTN=dense|gathered|kernel`,
`FLASHVSR_ATTN_BUDGET_MB` (dense grouping), `FLASHVSR_PROFILE=1` (per-stage time + MLX peak), `FLASHVSR_DEC_TRACE=1`.
Quality: `bench/x4_quality.py` scores a run against the spike's ×4 cells with the spike's own metric code.
