// FlashVSR's DiT — MLX-Swift port of OpenImagingLab/FlashVSR @ cf910c61 `diffsynth/models/wan_video_dit.py`
// (Apache-2.0): a Wan2.1-T2V-1.3B-shaped transformer, DMD-distilled to ONE step, run as a STREAMING model with
// locality-constrained sparse self-attention (LCSA) and a per-chunk K/V cache.
//
// Isomorphic to upstream: same module names and parameter paths (`blocks.N.self_attn.{q,k,v,o,norm_q,norm_k}`,
// `blocks.N.cross_attn.*`, `blocks.N.{norm3,ffn.0,ffn.2,modulation}`, `patch_embedding`, `text_embedding.{0,2}`,
// `time_embedding.{0,2}`, `time_projection.1`, `head.{head,modulation}`) — the checkpoint keys ARE upstream's.
// Substitutions only:
//   • NCDHW → NDHWC; tokens are (f, h, w)-major exactly as upstream's `rearrange('b c f h w -> b (f h w) c')`.
//   • RoPE: upstream multiplies in complex128 (CUDA/CPU); here (cos, sin) tables built in Double, applied in fp32.
//   • Block-sparse attention (mit-han-lab Block-Sparse-Attention, CUDA): computed as dense SDPA under the identical
//     128×128 block mask, one head at a time; a query block that selects no key block outputs zeros (the
//     FlashAttention-family convention the CUDA kernel follows — an assumption the CUDA oracle run must confirm).
// Dimensions (the released checkpoint): dim 1536, 30 blocks, 12 heads × 128, ffn 8960, patch (1, 2, 2), in/out 16.

import Foundation
import MLX
import MLXNN

public struct FlashVSRDiTConfig: Sendable {
    public var dim = 1536, inDim = 16, ffnDim = 8960, outDim = 16, textDim = 4096, freqDim = 256
    public var numHeads = 12, numLayers = 30
    public var eps: Float = 1e-6
    public var headDim: Int { dim / numHeads }
    public init() {}
    /// A shape-compatible miniature for plumbing tests (head dim stays 128 — the RoPE split needs it).
    public static func tiny(layers: Int = 1) -> FlashVSRDiTConfig {
        var c = FlashVSRDiTConfig()
        c.dim = 128; c.numHeads = 1; c.numLayers = layers; c.ffnDim = 256; c.textDim = 64; c.freqDim = 32
        return c
    }
}

/// Wan's RMSNorm: normalise in fp32, cast back, then scale (upstream `RMSNorm.forward`).
final class WanRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) {
        weight = MLXArray.ones([dim])
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let n = xf * rsqrt((xf * xf).mean(axis: -1, keepDims: true) + eps)
        return n.asType(x.dtype) * weight
    }
}

/// `nn.Sequential(Linear, GELU(tanh), Linear)` etc. are kept as layer arrays so `ffn.0` / `ffn.2` line up.
@inline(__always) func seq(_ layers: [UnaryLayer], _ x: MLXArray) -> MLXArray { layers.reduce(x) { $1($0) } }

// MARK: - RoPE (3-D factorised: temporal 44 | height 42 | width 42 of the 128-wide head)

/// Upstream `precompute_freqs_cis_3d(head_dim)`: three tables (end = 1024) over sub-dims 44, 42, 42, each normalised by
/// its OWN sub-dimension. Returned as (cos, sin) in fp32, built from Double angles.
struct RopeTables {
    let cosT: [Float], sinT: [Float], cosH: [Float], sinH: [Float], cosW: [Float], sinW: [Float]
    let nT: Int, nH: Int, nW: Int   // complex pairs per axis: 22, 21, 21
    static let end = 1024

    init(headDim d: Int, theta: Double = 10000) {
        let dT = d - 2 * (d / 3), dHW = d / 3
        func table(_ dim: Int) -> ([Float], [Float], Int) {
            let n = dim / 2
            var c = [Float](repeating: 0, count: Self.end * n), s = c
            for p in 0 ..< Self.end {
                for j in 0 ..< n {
                    let freq = 1.0 / pow(theta, Double(2 * j) / Double(dim))
                    let a = Double(p) * freq
                    c[p * n + j] = Float(cos(a)); s[p * n + j] = Float(sin(a))
                }
            }
            return (c, s, n)
        }
        (cosT, sinT, nT) = table(dT)
        (cosH, sinH, nH) = table(dHW)
        (cosW, sinW, nW) = table(dHW)
    }

    /// (cos, sin) for every token of an (f, h, w) grid whose temporal positions start at `t0`: shape [L, 1, 64].
    func grid(f: Int, h: Int, w: Int, t0: Int) -> (MLXArray, MLXArray) {
        let n = nT + nH + nW
        var c = [Float](repeating: 0, count: f * h * w * n), s = c
        var i = 0
        for t in 0 ..< f {
            for y in 0 ..< h {
                for x in 0 ..< w {
                    let o = i * n
                    for j in 0 ..< nT { c[o + j] = cosT[(t0 + t) * nT + j]; s[o + j] = sinT[(t0 + t) * nT + j] }
                    for j in 0 ..< nH { c[o + nT + j] = cosH[y * nH + j]; s[o + nT + j] = sinH[y * nH + j] }
                    for j in 0 ..< nW { c[o + nT + nH + j] = cosW[x * nW + j]; s[o + nT + nH + j] = sinW[x * nW + j] }
                    i += 1
                }
            }
        }
        return (MLXArray(c, [f * h * w, 1, n]), MLXArray(s, [f * h * w, 1, n]))
    }
}

/// Upstream `rope_apply`: view (…, 64, 2) pairs as complex (re = even, im = odd) and multiply by e^{iθ}.
func ropeApply(_ x: MLXArray, heads: Int, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2))
    let xf = x.asType(.float32).reshaped([b, l, heads, d / heads / 2, 2])
    let re = xf[.ellipsis, 0], im = xf[.ellipsis, 1]
    let outRe = re * cos - im * sin, outIm = re * sin + im * cos
    return stacked([outRe, outIm], axis: -1).reshaped([b, l, d]).asType(x.dtype)
}

// MARK: - LCSA

/// Upstream `WindowPartition3D.partition` for win (2, 8, 8): (B, F, H, W, C) → (B·nf·nh·nw, 128, C), windows ordered
/// (nf, nh, nw), tokens inside a window ordered (wf, wh, ww).
func windowPartition(_ x: MLXArray, f: Int, h: Int, w: Int) -> MLXArray {
    let (b, c) = (x.dim(0), x.dim(-1))
    return x.reshaped([b, f / 2, 2, h / 8, 8, w / 8, 8, c])
        .transposed(0, 1, 3, 5, 2, 4, 6, 7)
        .reshaped([b * (f / 2) * (h / 8) * (w / 8), 128, c])
}

/// Inverse of `windowPartition` back to (B, F·H·W, C) token order.
func windowReverse(_ x: MLXArray, b: Int, f: Int, h: Int, w: Int) -> MLXArray {
    let c = x.dim(-1)
    return x.reshaped([b, f / 2, h / 8, w / 8, 2, 8, 8, c])
        .transposed(0, 1, 4, 2, 5, 3, 6, 7)
        .reshaped([b, f * h * w, c])
}

/// Upstream `build_local_block_mask_shifted_vec_normal_slide(bh, bw, r, r, include_self=True)`: key block (r', c')
/// is visible to query block (r, c) iff r' ∈ [r − r//2, r − r//2 + r − 1] and likewise for columns (no clamping).
func localBlockMask(bh: Int, bw: Int, range r: Int) -> [Bool] {
    let half = r / 2
    var m = [Bool](repeating: false, count: bh * bw * bh * bw)
    for qr in 0 ..< bh {
        for qc in 0 ..< bw {
            for kr in 0 ..< bh {
                for kc in 0 ..< bw {
                    let inRow = kr >= qr - half && kr <= qr - half + r - 1
                    let inCol = kc >= qc - half && kc <= qc - half + r - 1
                    m[(qr * bw + qc) * (bh * bw) + kr * bw + kc] = inRow && inCol
                }
            }
        }
    }
    return m
}

/// Upstream `generate_draft_block_mask`: mean-pooled q/k per 128-token window → per-head block scores / √d, + the
/// local mask (−inf outside), softmax over key blocks, then per (head, query temporal slice) keep the block pairs
/// STRICTLY above the (topk + 1)-th largest probability. Returns (heads, Nq, Nk) bool as fp32 0/1.
func draftBlockMask(qW: MLXArray, kW: MLXArray, heads: Int, qSlices: Int, spatialBlocks s: Int,
                    local: [Bool], topk: Int) -> MLXArray {
    draftBlockMaskAndProbs(qW: qW, kW: kW, heads: heads, qSlices: qSlices, spatialBlocks: s, local: local,
                           topk: topk).mask
}

/// `draftBlockMask` plus the softmaxed block probabilities it thresholds (heads, Nq, Nk) — the gates use them to tell
/// a flipped near-tie from a wrong mask.
func draftBlockMaskAndProbs(qW: MLXArray, kW: MLXArray, heads: Int, qSlices: Int, spatialBlocks s: Int,
                            local: [Bool], topk: Int) -> (mask: MLXArray, probs: MLXArray) {
    let nq = qW.dim(0), nk = kW.dim(0), d = qW.dim(-1) / heads
    let pq = qW.asType(.float32).mean(axis: 1).reshaped([nq, heads, d]).transposed(1, 0, 2)   // (H, Nq, d)
    let pk = kW.asType(.float32).mean(axis: 1).reshaped([nk, heads, d]).transposed(1, 0, 2)   // (H, Nk, d)
    var scores = matmul(pq, pk.transposed(0, 2, 1)) / Float(d).squareRoot()                     // (H, Nq, Nk)
    // local mask tiled over (query slice, key slice) pairs
    let lm = MLXArray(local.map { $0 ? Float(0) : -Float.infinity }, [s, s])
    let tiled = broadcast(lm.reshaped([1, s, 1, s]), to: [nq / s, s, nk / s, s]).reshaped([nq, nk])
    scores = scores + tiled
    let attn = softmax(scores, axis: -1)                                                         // (H, Nq, Nk)
    let flat = attn.reshaped([heads * qSlices, (nq / qSlices) * nk])                             // (H·it, s1·s2)
    let n = flat.dim(1)
    let k = min(n - 1, topk)
    let threshold = sorted(flat, axis: 1)[0..., (n - 1 - k) ..< (n - k)]                         // (k+1)-th largest
    return ((flat .> threshold).asType(.float32).reshaped([heads, nq, nk]), attn)
}

/// Attention restricted to the selected 128×128 blocks — dense SDPA under the expanded BOOLEAN block mask, one head at
/// a time, and the queries of a head in groups of whole 128-blocks sized so one group's fp32 score matrix stays near
/// `scoreBudgetBytes` (MLX takes the unfused path at head-dim 128, so scores are materialised: one head at
/// 1920×1152 would be ~10 GB in one shot). Per-row arithmetic is unchanged by the grouping. On large grids each head
/// is evaluated before the next is built. Empty query blocks → 0.
/// Which attention implementation `blockForward` uses (FLASHVSR_ATTN=dense|gathered|kernel):
///   • `.kernel` — the block-sparse Metal kernel (`SparseAttention.swift`): the GPU default. Exact (fp32 rel ≤ 5e-7
///     vs dense), 2.2× faster end to end at 1920×1152 (DiT 2.2–3.7×), ~10 % at 1280×768, never slower.
///   • `.dense` — dense SDPA under the expanded block mask: the reference, and the CPU path.
///   • `.gathered` — per query block, the selected K/V blocks gathered and run through MLX's fused flash kernel.
///     Exact, but the gather replicates K/V per query block: 2–4× SLOWER than dense at 1280×768. Kept for study.
public enum FlashVSRAttention: String, Sendable {
    case dense, gathered, kernel
    public static var current: FlashVSRAttention {
        let gpu = Device.defaultDevice().deviceType == .gpu
        if let e = ProcessInfo.processInfo.environment["FLASHVSR_ATTN"], let v = FlashVSRAttention(rawValue: e),
           v == .dense || gpu {
            return v
        }
        return gpu ? .kernel : .dense
    }
}

/// Block-sparse attention that does only the selected work — the role of upstream's CUDA `block_sparse_attn_func`.
/// All 128 queries of a query block share one key selection, so per head the query blocks become a BATCH: each
/// gathers its selected key/value blocks (padded to the group's largest selection), and MLX's fused flash kernel runs
/// over them with a per-key padding mask that broadcasts over the queries. No score matrix is materialised and the
/// work scales with the selection density (~22 % at 1920×1152, where dense attention is ~84 % of the DiT's FLOPs).
/// Query blocks are grouped so the gathered K+V stay near `gatherBudgetBytes`. Empty query blocks → 0. GPU only (the
/// fused kernel is Metal).
func gatheredBlockAttention(q: MLXArray, k: MLXArray, v: MLXArray, blockMask: MLXArray, heads: Int,
                            gatherBudgetBytes: Int = 1 << 30) -> MLXArray {
    let (b, lq, d) = (q.dim(0), q.dim(1), q.dim(2))
    precondition(b == 1, "gathered attention: batch 1 (upstream asserts the same)")
    let lk = k.dim(1), hd = d / heads
    let nq = blockMask.dim(1), nk = blockMask.dim(2)
    let scale = 1 / Float(hd).squareRoot()
    let sel = blockMask .> 0.5                                                              // (H, Nq, Nk)
    let counts = sel.asType(.int32).sum(axis: -1)                                           // (H, Nq)
    // selected key-block indices first (order within a row is irrelevant to the softmax)
    let order = argSort((.!sel).asType(.int32), axis: -1)                                   // (H, Nq, Nk)
    let countsHost = counts.asArray(Int32.self)
    var outs: [MLXArray] = []
    for h in 0 ..< heads {
        let qh = q[0, 0..., (h * hd) ..< ((h + 1) * hd)].reshaped([nq, 1, 128, hd])
        let kb = k[0, 0..., (h * hd) ..< ((h + 1) * hd)].reshaped([nk, 128, hd])
        let vb = v[0, 0..., (h * hd) ..< ((h + 1) * hd)].reshaped([nk, 128, hd])
        let ch = Array(countsHost[(h * nq) ..< ((h + 1) * nq)])
        var parts: [MLXArray] = []
        var g0 = 0
        while g0 < nq {
            // grow the group while the padded gather fits the budget
            var g1 = g0, maxC = 0
            while g1 < nq {
                let m = max(maxC, Int(ch[g1]))
                if g1 > g0, (g1 + 1 - g0) * max(m, 1) * 128 * hd * k.itemSize * 2 > gatherBudgetBytes { break }
                maxC = m; g1 += 1
            }
            let n = g1 - g0
            if maxC == 0 {
                parts.append(MLXArray.zeros([n, 1, 128, hd], dtype: q.dtype))
                g0 = g1; continue
            }
            let idx = order[h, g0 ..< g1, 0 ..< maxC]                                       // (n, maxC)
            let kg = kb[idx.flattened()].reshaped([n, 1, maxC * 128, hd])
            let vg = vb[idx.flattened()].reshaped([n, 1, maxC * 128, hd])
            let cnt = counts[h, g0 ..< g1].reshaped([n, 1])
            let valid = MLXArray(0 ..< Int32(maxC)).reshaped([1, maxC]) .< cnt              // (n, maxC)
            let keyMask = broadcast(valid.reshaped([n, maxC, 1]), to: [n, maxC, 128])
                .reshaped([n, 1, 1, maxC * 128])
            var o = MLXFast.scaledDotProductAttention(queries: qh[g0 ..< g1], keys: kg, values: vg, scale: scale,
                                                      mask: .array(keyMask), forceFused: true)
            o = MLX.where(cnt.reshaped([n, 1, 1, 1]) .> 0, o, MLXArray(Float(0)).asType(o.dtype))
            parts.append(o)
            g0 = g1
        }
        var o = parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)                 // (Nq, 1, 128, hd)
        o = o.reshaped([1, lq, hd])
        if lq * lk >= 1 << 24 { eval(o) }
        outs.append(o)
    }
    return concatenated(outs, axis: -1)
}

/// FLASHVSR_ATTN_BUDGET_MB overrides the per-group score budget (the gates set it small to exercise the grouping).
let attentionScoreBudget: Int = ProcessInfo.processInfo.environment["FLASHVSR_ATTN_BUDGET_MB"]
    .flatMap(Int.init).map { $0 << 20 } ?? 512 << 20

func blockMaskedAttention(q: MLXArray, k: MLXArray, v: MLXArray, blockMask: MLXArray, heads: Int,
                          scoreBudgetBytes: Int = attentionScoreBudget) -> MLXArray {
    let (b, lq, d) = (q.dim(0), q.dim(1), q.dim(2))
    let lk = k.dim(1), hd = d / heads
    let nq = blockMask.dim(1), nk = blockMask.dim(2)
    let scale = 1 / Float(hd).squareRoot()
    let evalEachHead = lq * lk >= 1 << 24
    let group = max(1, min(nq, scoreBudgetBytes / (128 * lk * 4)))
    var outs: [MLXArray] = []
    for h in 0 ..< heads {
        let qh = q[0..., 0..., (h * hd) ..< ((h + 1) * hd)].reshaped([b, lq, 1, hd]).transposed(0, 2, 1, 3)
        let kh = k[0..., 0..., (h * hd) ..< ((h + 1) * hd)].reshaped([b, lk, 1, hd]).transposed(0, 2, 1, 3)
        let vh = v[0..., 0..., (h * hd) ..< ((h + 1) * hd)].reshaped([b, lk, 1, hd]).transposed(0, 2, 1, 3)
        let bm = blockMask[h] .> 0.5                                                        // (Nq, Nk) bool
        var parts: [MLXArray] = []
        for g0 in stride(from: 0, to: nq, by: group) {
            let g1 = min(nq, g0 + group), n = g1 - g0
            let bmg = bm[g0 ..< g1]
            let tok = broadcast(bmg.reshaped([n, 1, nk, 1]), to: [n, 128, nk, 128]).reshaped([1, 1, n * 128, lk])
            var o = MLXFast.scaledDotProductAttention(queries: qh[0..., 0..., (g0 * 128) ..< (g1 * 128)],
                                                      keys: kh, values: vh, scale: scale, mask: tok)
            // MLX fills a boolean mask's holes with finfo.min (finite), so a query block that selected nothing comes
            // out as the plain average of every value instead of NaN — zero those rows explicitly (upstream's
            // convention) (AB-L-0198)
            let live = broadcast(bmg.any(axis: -1).reshaped([n, 1, 1]), to: [n, 128, 1]).reshaped([1, 1, n * 128, 1])
            o = MLX.where(live, o, MLXArray(Float(0)).asType(o.dtype))
            parts.append(o)
        }
        var o = parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
        o = o.transposed(0, 2, 1, 3).reshaped([b, lq, hd])
        if evalEachHead { eval(o) }
        outs.append(o)
    }
    return concatenated(outs, axis: -1)
}

// MARK: - modules

final class SelfAttention: Module {
    let q: Linear, k: Linear, v: Linear, o: Linear
    @ModuleInfo(key: "norm_q") var normQ: WanRMSNorm
    @ModuleInfo(key: "norm_k") var normK: WanRMSNorm
    let heads: Int

    init(_ c: FlashVSRDiTConfig) {
        q = Linear(c.dim, c.dim); k = Linear(c.dim, c.dim); v = Linear(c.dim, c.dim); o = Linear(c.dim, c.dim)
        _normQ.wrappedValue = WanRMSNorm(c.dim, eps: c.eps)
        _normK.wrappedValue = WanRMSNorm(c.dim, eps: c.eps)
        heads = c.numHeads
    }
}

final class CrossAttention: Module {
    let q: Linear, k: Linear, v: Linear, o: Linear
    @ModuleInfo(key: "norm_q") var normQ: WanRMSNorm
    @ModuleInfo(key: "norm_k") var normK: WanRMSNorm
    let heads: Int
    var cacheK: MLXArray?
    var cacheV: MLXArray?

    init(_ c: FlashVSRDiTConfig) {
        q = Linear(c.dim, c.dim); k = Linear(c.dim, c.dim); v = Linear(c.dim, c.dim); o = Linear(c.dim, c.dim)
        _normQ.wrappedValue = WanRMSNorm(c.dim, eps: c.eps)
        _normK.wrappedValue = WanRMSNorm(c.dim, eps: c.eps)
        heads = c.numHeads
    }

    /// Upstream `init_cache(ctx)`: the context is a constant, so K/V are computed once.
    func initCache(_ ctx: MLXArray) {
        cacheK = normK(k(ctx))
        cacheV = v(ctx)
        eval(cacheK!, cacheV!)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2)), hd = d / heads
        let qh = normQ(q(x)).reshaped([b, l, heads, hd]).transposed(0, 2, 1, 3)
        let s = cacheK!.dim(1)
        let kh = cacheK!.reshaped([b, s, heads, hd]).transposed(0, 2, 1, 3)
        let vh = cacheV!.reshaped([b, s, heads, hd]).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh,
                                                    scale: 1 / Float(hd).squareRoot(), mask: nil)
        return o(out.transposed(0, 2, 1, 3).reshaped([b, l, d]))
    }
}

final class DiTBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SelfAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: CrossAttention
    let norm1: LayerNorm, norm2: LayerNorm, norm3: LayerNorm
    @ModuleInfo var ffn: [UnaryLayer]
    let modulation: MLXArray

    init(_ c: FlashVSRDiTConfig) {
        _selfAttn.wrappedValue = SelfAttention(c)
        _crossAttn.wrappedValue = CrossAttention(c)
        norm1 = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
        norm2 = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
        norm3 = LayerNorm(dimensions: c.dim, eps: c.eps, affine: true)
        _ffn.wrappedValue = [Linear(c.dim, c.ffnDim), GELU(approximation: .tanh), Linear(c.ffnDim, c.dim)]
        modulation = MLXArray.zeros([1, 6, c.dim])
    }
}

final class Head: Module {
    let norm: LayerNorm
    let head: Linear
    let modulation: MLXArray
    init(_ c: FlashVSRDiTConfig) {
        norm = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
        head = Linear(c.dim, c.outDim * 4)
        modulation = MLXArray.zeros([1, 2, c.dim])
    }
    /// Upstream `Head.forward(x, t)` — modulated by `t` (not `t_mod`), broadcast over the 2 rows.
    func callAsFunction(_ x: MLXArray, t: MLXArray) -> MLXArray {
        let m = modulation.asType(t.dtype) + t.reshaped([1, 1, -1])
        let shift = m[0..., 0 ..< 1], scale = m[0..., 1 ..< 2]
        return head(norm(x) * (1 + scale) + shift)
    }
}

/// Per-chunk inputs/outputs of the streaming step, plus optional taps for the parity gates.
public struct DiTChunkResult {
    public let pred: MLXArray                   // (B, f, H, W, 16) noise prediction (unpatchified, channels-last)
    public var taps: [String: MLXArray] = [:]
}

public final class FlashVSRDiT: Module {
    public let config: FlashVSRDiTConfig
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    @ModuleInfo(key: "text_embedding") var textEmbedding: [UnaryLayer]
    @ModuleInfo(key: "time_embedding") var timeEmbedding: [UnaryLayer]
    @ModuleInfo(key: "time_projection") var timeProjection: [UnaryLayer]
    let blocks: [DiTBlock]
    let head: Head
    let rope: RopeTables
    /// Constants of a one-step model at t = 1000: `t` (1, dim) and `t_mod` (1, 6, dim) — set by `prepare(context:)`.
    var t: MLXArray?
    var tMod: MLXArray?
    var localCache: [String: [Bool]] = [:]

    public init(_ c: FlashVSRDiTConfig = .init()) {
        config = c
        _patchEmbedding.wrappedValue = Conv3d(inputChannels: c.inDim, outputChannels: c.dim,
                                              kernelSize: [1, 2, 2], stride: [1, 2, 2])
        _textEmbedding.wrappedValue = [Linear(c.textDim, c.dim), GELU(approximation: .tanh), Linear(c.dim, c.dim)]
        _timeEmbedding.wrappedValue = [Linear(c.freqDim, c.dim), SiLU(), Linear(c.dim, c.dim)]
        _timeProjection.wrappedValue = [SiLU(), Linear(c.dim, c.dim * 6)]
        blocks = (0 ..< c.numLayers).map { _ in DiTBlock(c) }
        head = Head(c)
        rope = RopeTables(headDim: c.headDim)
        super.init()
        train(false)
    }

    /// Upstream `sinusoidal_embedding_1d(dim, position)` in Double: [cos(p·ω) ‖ sin(p·ω)], ω_j = 10000^(−j/(dim/2)).
    static func sinusoidal(dim: Int, position: Double) -> MLXArray {
        let half = dim / 2
        var v = [Float](repeating: 0, count: dim)
        for j in 0 ..< half {
            let a = position * pow(10000.0, -Double(j) / Double(half))
            v[j] = Float(cos(a)); v[half + j] = Float(sin(a))
        }
        return MLXArray(v, [1, dim])
    }

    /// Upstream `init_cross_kv`: text-embed the fixed prompt context, cache every block's cross K/V, and compute the
    /// t = 1000 time constants.
    public func prepare(context: MLXArray) {
        let dt = patchEmbedding.weight.dtype
        let ctx = seq(textEmbedding, context.asType(dt))
        for b in blocks { b.crossAttn.initCache(ctx) }
        let tt = seq(timeEmbedding, Self.sinusoidal(dim: config.freqDim, position: 1000).asType(dt))
        t = tt
        tMod = seq(timeProjection, tt).reshaped([1, 6, config.dim])
        eval(t!, tMod!)
    }

    /// The t = 1000 constants and block `i`'s cached cross-attention K/V (for the gates).
    public var timeConstants: (t: MLXArray, tMod: MLXArray)? { t.flatMap { t in tMod.map { (t, $0) } } }
    public func crossKV(block i: Int) -> (k: MLXArray, v: MLXArray)? {
        blocks[i].crossAttn.cacheK.flatMap { k in blocks[i].crossAttn.cacheV.map { (k, $0) } }
    }
    public var blockCount: Int { blocks.count }

    func local(bh: Int, bw: Int, range: Int) -> [Bool] {
        let key = "\(bh)x\(bw)r\(range)"
        if let m = localCache[key] { return m }
        let m = localBlockMask(bh: bh, bw: bw, range: range)
        localCache[key] = m
        return m
    }

    /// Everything a chunk's blocks share: token grid, RoPE tables, LCSA parameters, the time modulation.
    struct StepGeometry {
        let b: Int, f: Int, h: Int, w: Int, seqlen: Int, s: Int, topk: Int, kvLen: Int
        let cos: MLXArray, sin: MLXArray, local: [Bool], tMod: MLXArray
    }

    func geometry(b: Int, f: Int, h: Int, w: Int, dtype: DType, t0: Int, topkRatio: Double, kvLen: Int,
                  localRange: Int) -> StepGeometry {
        let s = (h / 8) * (w / 8)
        let (cosR, sinR) = rope.grid(f: f, h: h, w: w, t0: t0)
        return StepGeometry(b: b, f: f, h: h, w: w, seqlen: f / 2, s: s, topk: Int(Double(s * s) * topkRatio) - 1,
                            kvLen: kvLen, cos: cosR, sin: sinR, local: local(bh: h / 8, bw: w / 8, range: localRange),
                            tMod: tMod!.asType(dtype))
    }

    /// One DiT block — upstream `DiTBlock.forward(..., is_stream=True)`. Updates the block's window K/V cache in place.
    func blockForward(_ i: Int, _ x: MLXArray, _ g: StepGeometry, cacheK: inout MLXArray?, cacheV: inout MLXArray?,
                      taps: inout [String: MLXArray]?, maskOverride: MLXArray? = nil) -> MLXArray {
        let c = config, blk = blocks[i]
        var tok = x
        taps?["b\(i).in"] = tok
        let mod = blk.modulation.asType(g.tMod.dtype) + g.tMod                         // (1, 6, dim)
        let shiftMSA = mod[0..., 0 ..< 1], scaleMSA = mod[0..., 1 ..< 2], gateMSA = mod[0..., 2 ..< 3]
        let shiftMLP = mod[0..., 3 ..< 4], scaleMLP = mod[0..., 4 ..< 5], gateMLP = mod[0..., 5 ..< 6]
        let inX = blk.norm1(tok) * (1 + scaleMSA) + shiftMSA
        // self-attention (streaming, LCSA)
        let sa = blk.selfAttn
        var q = sa.normQ(sa.q(inX)), k = sa.normK(sa.k(inX))
        let v = sa.v(inX)
        q = ropeApply(q, heads: c.numHeads, cos: g.cos, sin: g.sin)
        k = ropeApply(k, heads: c.numHeads, cos: g.cos, sin: g.sin)
        let (b, f, h, w) = (g.b, g.f, g.h, g.w)
        let qW = windowPartition(q.reshaped([b, f, h, w, c.dim]), f: f, h: h, w: w)
        var kW = windowPartition(k.reshaped([b, f, h, w, c.dim]), f: f, h: h, w: w)
        var vW = windowPartition(v.reshaped([b, f, h, w, c.dim]), f: f, h: h, w: w)
        let oneLen = kW.dim(0) / b / g.seqlen
        if let ck = cacheK, let cv = cacheV {
            kW = concatenated([ck, kW], axis: 0)
            vW = concatenated([cv, vW], axis: 0)
        }
        let (mask, probs) = draftBlockMaskAndProbs(qW: qW, kW: kW, heads: c.numHeads, qSlices: g.seqlen,
                                                   spatialBlocks: g.s, local: g.local, topk: g.topk)
        let rq = qW.reshaped([b, -1, c.dim]), rk = kW.reshaped([b, -1, c.dim]), rv = vW.reshaped([b, -1, c.dim])
        let bm = maskOverride ?? mask
        let attn: MLXArray
        switch FlashVSRAttention.current {
        case .dense: attn = blockMaskedAttention(q: rq, k: rk, v: rv, blockMask: bm, heads: c.numHeads)
        case .gathered: attn = gatheredBlockAttention(q: rq, k: rk, v: rv, blockMask: bm, heads: c.numHeads)
        case .kernel: attn = kernelBlockAttention(q: rq, k: rk, v: rv, blockMask: bm, heads: c.numHeads)
        }
        if taps != nil {
            taps?["b\(i).sa_in"] = inX; taps?["b\(i).q"] = rq; taps?["b\(i).k"] = rk; taps?["b\(i).v"] = rv
            taps?["b\(i).mask"] = mask; taps?["b\(i).probs"] = probs; taps?["b\(i).attn"] = attn
        }
        if kW.dim(0) / oneLen > g.kvLen {
            cacheK = kW[oneLen...]
            cacheV = vW[oneLen...]
        } else {
            cacheK = kW
            cacheV = vW
        }
        let saOut = sa.o(windowReverse(attn.reshaped([-1, 128, c.dim]), b: b, f: f, h: h, w: w))
        taps?["b\(i).sa_out"] = saOut
        tok = tok + gateMSA * saOut
        let caIn = blk.norm3(tok)
        let caOut = blk.crossAttn(caIn)
        taps?["b\(i).ca_in"] = caIn; taps?["b\(i).ca_out"] = caOut
        tok = tok + caOut
        let inF = blk.norm2(tok) * (1 + scaleMLP) + shiftMLP
        tok = tok + gateMLP * seq(blk.ffn, inF)
        taps?["b\(i).out"] = tok
        return tok
    }

    /// One streaming step — upstream `model_fn_wan_video` with `is_stream = True`. `x`: (B, f, H, W, 16) latents of
    /// this chunk (f = 6 for chunk 0, 2 after); `lq`: the LQ projector's per-layer token additions; `cacheK/V`: the
    /// per-block window caches from the previous chunk (updated in place). `t0`: temporal RoPE start (0, then
    /// 4 + 2·chunk). `topk = int(S²·topkRatio) − 1` with S the spatial blocks per temporal slice.
    public func step(_ x: MLXArray, lq: [MLXArray], cacheK: inout [MLXArray?], cacheV: inout [MLXArray?],
                     t0: Int, topkRatio: Double, kvLen: Int, localRange: Int,
                     tapBlocks: Set<Int> = []) -> DiTChunkResult {
        let c = config
        let pe = patchEmbedding(x.asType(patchEmbedding.weight.dtype))                // (B, f, h, w, dim)
        let (b, f, h, w) = (pe.dim(0), pe.dim(1), pe.dim(2), pe.dim(3))
        var tok = pe.reshaped([b, f * h * w, c.dim])
        let g = geometry(b: b, f: f, h: h, w: w, dtype: tok.dtype, t0: t0, topkRatio: topkRatio, kvLen: kvLen,
                         localRange: localRange)
        var all: [String: MLXArray] = [:]
        for i in blocks.indices {
            if i < lq.count { tok = tok + lq[i].asType(tok.dtype) }
            var taps: [String: MLXArray]? = tapBlocks.contains(i) ? [:] : nil
            tok = blockForward(i, tok, g, cacheK: &cacheK[i], cacheV: &cacheV[i], taps: &taps)
            if let taps { all.merge(taps) { $1 } }
            eval(tok, cacheK[i]!, cacheV[i]!)       // bound the graph to one block (the family's large-seqLen rule)
        }
        let out = head(tok, t: t!.asType(tok.dtype))                                     // (B, L, 64)
        // unpatchify 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)', x=1 y=2 z=2 c=16 → channels-last
        let pred = out.reshaped([b, f, h, w, 2, 2, c.outDim]).transposed(0, 1, 2, 4, 3, 5, 6)
            .reshaped([b, f, h * 2, w * 2, c.outDim])
        return DiTChunkResult(pred: pred, taps: all)
    }

    /// Gate entry: run block `i` alone on a given token input (B, f·h·w, dim) for an (f, h, w) patch grid, with an
    /// optional incoming window cache. Returns the block's taps (incl. `b<i>.probs`, the draft block probabilities).
    /// `maskOverride` (heads, Nq, Nk) replaces the block's own draft mask in the attention (the mask itself is still
    /// computed and tapped) — isolates the attention kernel from the selection rule.
    public func isolatedBlock(_ i: Int, input: MLXArray, f: Int, h: Int, w: Int, t0: Int, topkRatio: Double,
                              kvLen: Int, localRange: Int, cacheK: MLXArray?, cacheV: MLXArray?,
                              maskOverride: MLXArray? = nil) -> [String: MLXArray] {
        let g = geometry(b: input.dim(0), f: f, h: h, w: w, dtype: input.dtype, t0: t0, topkRatio: topkRatio,
                         kvLen: kvLen, localRange: localRange)
        var ck = cacheK, cv = cacheV
        var taps: [String: MLXArray]? = [:]
        _ = blockForward(i, input, g, cacheK: &ck, cacheV: &cv, taps: &taps, maskOverride: maskOverride)
        taps?["b\(i).cache_k"] = ck; taps?["b\(i).cache_v"] = cv
        return taps!
    }
}
