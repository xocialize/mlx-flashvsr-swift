"""block_sparse_attn.py — an exact, slow stand-in for mit-han-lab Block-Sparse-Attention's `block_sparse_attn_func`
(CUDA-only), so FlashVSR's own `wan_video_dit.py` runs unmodified on MPS/CPU.

FlashVSR calls it with 128-token blocks on both axes and `head_mask_type = 1` (block-sparse) for every head:
`base_blockmask[0, h, i, j]` says whether query block i attends key block j. Block-sparse attention computes exactly
softmax-attention restricted to the selected blocks; the same thing is dense SDPA with that block mask expanded to
tokens (masked logits → −inf). This file does that, one head at a time to bound memory. Slower than the sparse
kernel (it still forms every q·k product), identical in what it computes.

Rows whose query block selects NO key block would be NaN under SDPA; FlashAttention-family kernels write zeros for an
empty row, so we do the same and count them (`EMPTY_ROWS`) — the receipt reports the count.
"""
import torch
import torch.nn.functional as F

BLOCK = 128
EMPTY_ROWS = {"blocks": 0, "calls": 0}


def block_sparse_attn_func(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask,
                           max_seqlen_q, max_seqlen_k, p_dropout, deterministic=False, softmax_scale=None,
                           is_causal=False, exact_streaming=False, return_attn_probs=False):
    assert not is_causal and p_dropout == 0.0 and streaming_info is None
    assert base_blockmask.shape[0] == 1, "batch 1 only (FlashVSR asserts the same)"
    lq, nh, d = q.shape
    lk = k.shape[0]
    scale = softmax_scale if softmax_scale is not None else d ** -0.5
    out = torch.empty_like(q)
    EMPTY_ROWS["calls"] += 1
    for h in range(nh):
        assert int(head_mask_type[h]) == 1, "only block-sparse heads are used by FlashVSR"
        bm = base_blockmask[0, h].to(torch.bool)                                   # (nqb, nkb)
        EMPTY_ROWS["blocks"] += int((~bm.any(dim=1)).sum())
        tok = bm.repeat_interleave(BLOCK, 0)[:lq].repeat_interleave(BLOCK, 1)[:, :lk]   # (lq, lk) True = attend
        qh, kh, vh = q[:, h][None, None], k[:, h][None, None], v[:, h][None, None]     # (1,1,L,d)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=tok[None, None], scale=scale)[0, 0]
        out[:, h] = torch.nan_to_num(o, nan=0.0)
    return out
