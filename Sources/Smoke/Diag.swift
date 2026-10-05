// Diagnostics — where two runs of the DiT on (nearly) the same input part ways.
import Foundation
import FlashVSRMLX
import MLX
import MLXNN

/// Chunk 0 twice on the current device: once with the golden LQ tokens, once with the port's own projector tokens.
/// Reports, per block, the mask flips between the two runs and the draft probabilities at the flips.
func diagMaskSensitivity(weights: URL, golden: URL, gpu: Bool) throws {
    try Device.withDefaultDevice(gpu ? Device(.gpu) : Device(.cpu)) {
        let (g, meta) = try loadArraysAndMetadata(url: golden)
        let p = try FlashVSRPipeline.load(directory: weights)
        let topk = Double(meta["topk_ratio"]!)!, local = Int(meta["local_range"]!)!
        let lq = cl5(g["input.lq"]!)
        p.lqProj.reset()
        var tok: MLXArray?
        for i in 0 ..< 7 {
            guard let cur = p.lqProj.stream(lq[0..., max(0, 4 * i - 3) ..< (i + 1) * 4 - 3]) else { continue }
            tok = tok.map { concatenated([$0, cur[0]], axis: 1) } ?? cur[0]
        }
        let x = cl5(g["c0.x_in"]!)
        let all = Set(0 ..< 30)
        var ck = [MLXArray?](repeating: nil, count: 30), cv = ck
        let a = p.dit.step(x, lq: [g["c0.lq.0"]!], cacheK: &ck, cacheV: &cv, t0: 0, topkRatio: topk, kvLen: 3,
                           localRange: local, tapBlocks: all)
        ck = .init(repeating: nil, count: 30); cv = ck
        let b = p.dit.step(x, lq: [tok!], cacheK: &ck, cacheV: &cv, t0: 0, topkRatio: topk, kvLen: 3,
                           localRange: local, tapBlocks: all)
        note("lq tokens golden vs port: " + compare("lq", tok!, g["c0.lq.0"]!).line)
        for i in 0 ..< 30 {
            let ma = a.taps["b\(i).mask"]!, mb = b.taps["b\(i).mask"]!
            let flips = (abs(ma - mb) .> 0.5)
            let n = flips.asType(.int32).sum().item(Int.self)
            let din = compare("in", b.taps["b\(i).in"]!, a.taps["b\(i).in"]!)
            var s = String(format: "  b%02d in-drift %.2e  flips %d", i, din.rel, n)
            if n > 0 {
                let pa = a.taps["b\(i).probs"]!, pb = b.taps["b\(i).probs"]!
                let fa = MLX.where(flips, pa, MLXArray(Float(0))).max().item(Float.self)
                let fb = MLX.where(flips, pb, MLXArray(Float(0))).max().item(Float.self)
                s += String(format: "  max p at flips %.3e / %.3e", fa, fb)
            }
            note(s)
        }
        note("pred A vs B: " + compare("pred", b.pred, a.pred).line)
        note("pred A vs golden: " + compare("pred", a.pred, cl5(g["c0.pred"]!)).line)
        note("pred B vs golden: " + compare("pred", b.pred, cl5(g["c0.pred"]!)).line)
    }
}

/// CPU-path determinism: run() twice (no colour fix), then the stream (no colour fix), on a golden's inputs.
func diagStream(weights: URL, golden: URL, gpu: Bool) throws {
    try Device.withDefaultDevice(gpu ? Device(.gpu) : Device(.cpu)) {
        let (g, meta) = try loadArraysAndMetadata(url: golden)
        let p = try FlashVSRPipeline.load(directory: weights)
        let opts = FlashVSROptions(topkRatio: Double(meta["topk_ratio"]!)!, localRange: Int(meta["local_range"]!)!,
                                   colorFix: false)
        let lq = cl5(g["input.lq"]!), noise = cl5(g["input.noise"]!)
        let a = try p.run(lq: lq, noise: noise, options: opts)!
        let b = try p.run(lq: lq, noise: noise, options: opts)!
        note(compare("run A vs run B", b, a).line)
        let st = FlashVSRStream(pipeline: p, height: lq.dim(2), width: lq.dim(3), options: opts)
        let pad = concatenated([noise, MLXRandom.normal([1, 8, noise.dim(2), noise.dim(3), 16])], axis: 1)
        st.noiseProvider = { pad[0..., $0] }
        var sf: [MLXArray] = []
        for i in 0 ..< lq.dim(1) { sf += try st.push(lq[0, i]) }
        sf += try st.finish()
        let s = stacked(sf, axis: 0).expandedDimensions(axis: 0)[0..., 0 ..< a.dim(1)]
        note(compare("stream vs run A", s, a).line)
        let fa = FlashVSRPipeline.colorFix(a, lq: lq[0..., 0 ..< a.dim(1)])
        let fs = FlashVSRPipeline.colorFix(s, lq: lq[0..., 0 ..< a.dim(1)])
        let fsc = FlashVSRPipeline.colorFix(s, lq: contiguous(lq[0..., 0 ..< a.dim(1)]))
        note(compare("colourfix(stream) vs colourfix(A)", fs, fa).line)
        note(compare("colourfix(stream, contiguous lq) vs (strided lq)", fsc, fs).line)
        if let gold = g["out.frames"] {
            let ref = gold.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
            note(compare("colourfix(A) vs golden out.frames", fa, ref).line)
            note(compare("colourfix(A, contiguous lq) vs golden", FlashVSRPipeline.colorFix(contiguous(a),
                         lq: contiguous(lq[0..., 0 ..< a.dim(1)])), ref).line)
        }
    }
}


/// The block-sparse kernel against the dense reference on random q/k/v and a random block mask with empty rows
/// (GPU; both paths in fp32 and bf16). Also times both on a 1280×768-chunk-shaped problem.
func probeAttention() throws {
    for (dt, lqB, lkB, density) in [(DType.float32, 4, 6, 0.5), (.float32, 12, 36, 0.3), (.bfloat16, 12, 36, 0.6)] {
        let heads = 2, hd = 128
        let (lq, lk) = (lqB * 128, lkB * 128)
        let q = MLXRandom.normal([1, lq, heads * hd]).asType(dt)
        let k = MLXRandom.normal([1, lk, heads * hd]).asType(dt)
        let v = MLXRandom.normal([1, lk, heads * hd]).asType(dt)
        var m = (MLXRandom.uniform(0 ..< 1, [heads, lqB, lkB]) .< Float(density)).asType(.float32)
        m[0, 1] = MLXArray.zeros([lkB])                                   // an empty query block
        let ref = FlashVSRAttentionOps.dense(q: q, k: k, v: v, blockMask: m, heads: heads)
        let got = FlashVSRAttentionOps.kernel(q: q, k: k, v: v, blockMask: m, heads: heads)
        note(compare("kernel \(dt) \(lqB)x\(lkB) d\(density)", got, ref).line)
    }
    // bf16 accuracy: each bf16 path against the fp32 dense result on the same (bf16-representable) inputs
    do {
        let heads = 2, hd = 128, lqB = 12, lkB = 36
        let q = MLXRandom.normal([1, lqB * 128, heads * hd]).asType(.bfloat16)
        let k = MLXRandom.normal([1, lkB * 128, heads * hd]).asType(.bfloat16)
        let v = MLXRandom.normal([1, lkB * 128, heads * hd]).asType(.bfloat16)
        let m = (MLXRandom.uniform(0 ..< 1, [heads, lqB, lkB]) .< Float(0.5)).asType(.float32)
        let ref = FlashVSRAttentionOps.dense(q: q.asType(.float32), k: k.asType(.float32), v: v.asType(.float32),
                                             blockMask: m, heads: heads)
        note(compare("bf16 dense  vs fp32", FlashVSRAttentionOps.dense(q: q, k: k, v: v, blockMask: m, heads: heads), ref).line)
        note(compare("bf16 kernel vs fp32", FlashVSRAttentionOps.kernel(q: q, k: k, v: v, blockMask: m, heads: heads), ref).line)
    }
    // timing on a 1280×768 chunk-1 shape: Lq = 2·48·80 = 7680 (60 blocks), Lk = 4 slices = 240 blocks, 12 heads
    let heads = 12, hd = 128, lqB = 60, lkB = 240
    for (dt, density) in [(DType.bfloat16, 0.5), (.bfloat16, 0.22)] {
        let q = MLXRandom.normal([1, lqB * 128, heads * hd]).asType(dt)
        let k = MLXRandom.normal([1, lkB * 128, heads * hd]).asType(dt)
        let v = MLXRandom.normal([1, lkB * 128, heads * hd]).asType(dt)
        let m = (MLXRandom.uniform(0 ..< 1, [heads, lqB, lkB]) .< Float(density)).asType(.float32)
        eval(q, k, v, m)
        for (name, f) in [("dense", { FlashVSRAttentionOps.dense(q: q, k: k, v: v, blockMask: m, heads: heads) }),
                          ("kernel", { FlashVSRAttentionOps.kernel(q: q, k: k, v: v, blockMask: m, heads: heads) })] {
            eval(f())
            var ts: [Double] = []
            for _ in 0 ..< 3 { let (o, t) = timed { let o = f(); eval(o); return o }; _ = o; ts.append(t) }
            note(String(format: "  %@ %@ density %.2f: %.1f ms (min of 3)", name as NSString,
                        "\(dt)" as NSString, density, 1000 * ts.min()!))
        }
    }
}

/// The published lanes: the bf16 lane's parameters must equal the fp32 lane cast to bf16 at load, bit for bit.
func checkLanes(fp32Dir: URL, bf16Dir: URL) throws {
    let a = try FlashVSRPipeline.load(directory: fp32Dir, dtypes: .production)   // fp32 files, cast to bf16
    let b = try FlashVSRPipeline.load(directory: bf16Dir, dtypes: .production)   // bf16 files
    var n = 0, diff = 0
    for (ma, mb) in [(a.dit as Module, b.dit as Module), (a.lqProj, b.lqProj), (a.decoder, b.decoder)] {
        let pa = Dictionary(uniqueKeysWithValues: ma.parameters().flattened())
        for (k, vb) in mb.parameters().flattened() {
            n += 1
            guard let va = pa[k], va.shape == vb.shape, va.dtype == vb.dtype,
                  (va .== vb).all().item(Bool.self) else { diff += 1; note("  differs: \(k)"); continue }
        }
    }
    let (ta, tma) = a.dit.timeConstants!, (tb, tmb) = b.dit.timeConstants!
    let consts = (ta .== tb).all().item(Bool.self) && (tma .== tmb).all().item(Bool.self)
    note("lanes: \(n) parameters compared, \(diff) differ; t/t_mod identical: \(consts)")
    if diff > 0 || !consts { throw SmokeError("bf16 lane != fp32 lane cast to bf16") }
}
