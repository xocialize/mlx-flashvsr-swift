// Block-sparse flash attention as a Metal kernel — the role of mit-han-lab's CUDA `block_sparse_attn_func` that
// upstream FlashVSR calls. Work and memory scale with the SELECTED blocks only: a threadgroup owns 32 query rows of
// one 128-query block of one head (4 simdgroups × 8 rows), walks that block's selected key blocks by index (no
// gather, no copies, no score matrix), 16 keys at a time through threadgroup memory, with S = Q·Kᵀ and O += P·V on
// 8×8 simdgroup matrices and an fp32 online softmax. A query block that selected nothing writes zeros (upstream's
// convention, the same one the dense path applies).
import Foundation
import MLX

enum BlockSparseKernel {
    static let source = """
        constexpr int D = 128;
        constexpr int BK = 16;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint tid = thread_index_in_threadgroup;
        const uint3 tgp = threadgroup_position_in_grid;
        const int Lq = meta[0], Lk = meta[1], NQ = meta[2], NKB = meta[3];
        const int h = int(tgp.y);
        const int qb = int(tgp.x) / 4, qsub = int(tgp.x) % 4;
        const int row0 = qb * 128 + qsub * 32 + int(sg) * 8;

        threadgroup float Ks[BK * D];
        threadgroup float Vs[BK * D];
        threadgroup float scr[4][8 * 16];
        threadgroup float dg[4][64];
        threadgroup float mrow[4][8];
        threadgroup float lrow[4][8];
        threadgroup float stage[4][64];

        const device T* qh = q + size_t(h) * size_t(Lq) * D;
        const device T* kh = k + size_t(h) * size_t(Lk) * D;
        const device T* vh = v + size_t(h) * size_t(Lk) * D;
        device T* oh = o + size_t(h) * size_t(Lq) * D;

        const int cnt = kcnt[h * NQ + qb];
        if (cnt == 0) {
            for (int e = int(lane); e < 8 * D; e += 32) {
                oh[size_t(row0 + e / D) * D + (e % D)] = T(0);
            }
            return;
        }

        simdgroup_float8x8 Qt[16];
        for (int t = 0; t < 16; ++t) {
            for (int e = int(lane); e < 64; e += 32) {
                stage[sg][e] = float(qh[size_t(row0 + e / 8) * D + t * 8 + (e % 8)]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(Qt[t], stage[sg], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_float8x8 Ot[16];
        for (int t = 0; t < 16; ++t) { Ot[t] = simdgroup_float8x8(0); }
        if (lane < 8) { mrow[sg][lane] = -INFINITY; lrow[sg][lane] = 0.0f; }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float sc = scale;
        const int r = int(lane) / 4, c0 = (int(lane) % 4) * 4;
        for (int j = 0; j < cnt; ++j) {
            const int kb = kidx[(h * NQ + qb) * NKB + j];
            for (int sub = 0; sub < 128 / BK; ++sub) {
                const int key0 = kb * 128 + sub * BK;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int e = int(tid); e < BK * D; e += 128) {
                    const size_t g = size_t(key0 + e / D) * D + (e % D);
                    Ks[e] = float(kh[g]);
                    Vs[e] = float(vh[g]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 S0 = simdgroup_float8x8(0), S1 = simdgroup_float8x8(0);
                for (int dd = 0; dd < 16; ++dd) {
                    simdgroup_float8x8 K0, K1;
                    simdgroup_load(K0, Ks + dd * 8, D, ulong2(0, 0), true);
                    simdgroup_load(K1, Ks + 8 * D + dd * 8, D, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(S0, Qt[dd], K0, S0);
                    simdgroup_multiply_accumulate(S1, Qt[dd], K1, S1);
                }
                simdgroup_store(S0, scr[sg], 16);
                simdgroup_store(S1, scr[sg] + 8, 16);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                float s[4];
                float mx = -INFINITY;
                for (int i = 0; i < 4; ++i) { s[i] = scr[sg][r * 16 + c0 + i] * sc; mx = max(mx, s[i]); }
                mx = max(mx, simd_shuffle_xor(mx, 1));
                mx = max(mx, simd_shuffle_xor(mx, 2));
                const float mold = mrow[sg][r];
                const float lold = lrow[sg][r];
                const float mnew = max(mold, mx);
                float sum = 0.0f;
                for (int i = 0; i < 4; ++i) { s[i] = exp(s[i] - mnew); sum += s[i]; }
                sum += simd_shuffle_xor(sum, 1);
                sum += simd_shuffle_xor(sum, 2);
                const float alpha = exp(mold - mnew);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = 0; i < 4; ++i) { scr[sg][r * 16 + c0 + i] = s[i]; }
                for (int e = int(lane); e < 64; e += 32) { dg[sg][e] = 0.0f; }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if ((lane % 4) == 0) {
                    dg[sg][r * 8 + r] = alpha;
                    mrow[sg][r] = mnew;
                    lrow[sg][r] = lold * alpha + sum;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 A, P0, P1;
                simdgroup_load(A, dg[sg], 8);
                simdgroup_load(P0, scr[sg], 16);
                simdgroup_load(P1, scr[sg] + 8, 16);
                for (int t = 0; t < 16; ++t) {
                    simdgroup_float8x8 tmp;
                    simdgroup_multiply(tmp, A, Ot[t]);
                    simdgroup_float8x8 V0, V1;
                    simdgroup_load(V0, Vs + t * 8, D);
                    simdgroup_load(V1, Vs + 8 * D + t * 8, D);
                    simdgroup_multiply_accumulate(tmp, P0, V0, tmp);
                    simdgroup_multiply_accumulate(Ot[t], P1, V1, tmp);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (int e = int(lane); e < 64; e += 32) { dg[sg][e] = 0.0f; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if ((lane % 4) == 0) { dg[sg][r * 8 + r] = 1.0f / lrow[sg][r]; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 Dinv;
        simdgroup_load(Dinv, dg[sg], 8);
        for (int t = 0; t < 16; ++t) {
            simdgroup_float8x8 res;
            simdgroup_multiply(res, Dinv, Ot[t]);
            simdgroup_store(res, stage[sg], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = int(lane); e < 64; e += 32) {
                oh[size_t(row0 + e / 8) * D + t * 8 + (e % 8)] = T(stage[sg][e]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        """

    static let kernel = MLXFast.metalKernel(
        name: "flashvsr_block_sparse_attention",
        inputNames: ["q", "k", "v", "kidx", "kcnt", "meta", "scale"],
        outputNames: ["o"],
        source: source)
}

/// Block-sparse attention on the GPU kernel. q (1, Lq, H·128), k/v (1, Lk, H·128) in window order (128-token blocks),
/// blockMask (H, Nq, Nk) 0/1. Returns (1, Lq, H·128).
func kernelBlockAttention(q: MLXArray, k: MLXArray, v: MLXArray, blockMask: MLXArray, heads: Int) -> MLXArray {
    let (lq, d) = (q.dim(1), q.dim(2))
    let lk = k.dim(1), hd = d / heads
    precondition(q.dim(0) == 1 && hd == 128, "kernel attention: batch 1, head dim 128")
    let nq = blockMask.dim(1), nk = blockMask.dim(2)
    let sel = blockMask .> 0.5
    let kcnt = sel.asType(.int32).sum(axis: -1)                                             // (H, Nq)
    let kidx = argSort((.!sel).asType(.int32), axis: -1).asType(.int32)                     // selected first
    func heads3(_ x: MLXArray, _ l: Int) -> MLXArray { x.reshaped([l, heads, hd]).transposed(1, 0, 2) }
    let meta = MLXArray([Int32(lq), Int32(lk), Int32(nq), Int32(nk)])
    let out = BlockSparseKernel.kernel(
        [heads3(q[0], lq), heads3(k[0], lk), heads3(v[0], lk), kidx, kcnt, meta,
         MLXArray(1 / Float(hd).squareRoot())],
        template: [("T", q.dtype)],
        grid: (nq * 4 * 128, heads, 1), threadGroup: (128, 1, 1),
        outputShapes: [[heads, lq, hd]], outputDTypes: [q.dtype])[0]
    return out.transposed(1, 0, 2).reshaped([1, lq, d])
}

/// Public entry points for probes and benchmarks (the model calls the implementations directly).
public enum FlashVSRAttentionOps {
    public static func dense(q: MLXArray, k: MLXArray, v: MLXArray, blockMask: MLXArray, heads: Int) -> MLXArray {
        blockMaskedAttention(q: q, k: k, v: v, blockMask: blockMask, heads: heads)
    }
    public static func gathered(q: MLXArray, k: MLXArray, v: MLXArray, blockMask: MLXArray, heads: Int) -> MLXArray {
        gatheredBlockAttention(q: q, k: k, v: v, blockMask: blockMask, heads: heads)
    }
    public static func kernel(q: MLXArray, k: MLXArray, v: MLXArray, blockMask: MLXArray, heads: Int) -> MLXArray {
        kernelBlockAttention(q: q, k: k, v: v, blockMask: blockMask, heads: heads)
    }
}
