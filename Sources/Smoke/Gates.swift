// Parity gates against `oracle/dump_goldens.py` — upstream FlashVSR executed VERBATIM on the CPU in fp32.
// All gates pin the CPU stream (`Device.withDefaultDevice`), so the comparison is fp32-vs-fp32 arithmetic, not a
// Metal-vs-CPU drift measurement. Layout: goldens are upstream's NCDHW / NTCHW; the port is channels-last.

import Foundation
import FlashVSRMLX
import MLX
import MLXNN

struct Cmp {
    let name: String
    let maxAbs: Float, refMax: Float, rel: Float, psnr: Double
    var line: String {
        String(format: "  %-16@ maxAbs %.3e  ref|max| %.3e  rel %.3e  psnr %6.1f dB", name as NSString,
               maxAbs, refMax, rel, psnr)
    }
}

func compare(_ name: String, _ got: MLXArray, _ ref: MLXArray) -> Cmp {
    precondition(got.shape == ref.shape, "\(name): shape \(got.shape) vs golden \(ref.shape)")
    let g = got.asType(.float32), r = ref.asType(.float32)
    let d = abs(g - r)
    let maxAbs = d.max().item(Float.self)
    let refMax = abs(r).max().item(Float.self)
    let mse = (d * d).mean().item(Float.self)
    let peak = max(refMax, 1e-12)
    let psnr = mse == 0 ? Double.infinity : 10 * log10(Double(peak * peak) / Double(mse))
    return Cmp(name: name, maxAbs: maxAbs, refMax: refMax, rel: maxAbs / peak, psnr: psnr)
}

/// NCDHW → NDHWC
func cl5(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 4, 1) }
/// NTCHW → NTHWC
func clT(_ x: MLXArray) -> MLXArray { x.transposed(0, 1, 3, 4, 2) }

struct GateReport {
    var rows: [Cmp] = []
    var notes: [String] = []
    var failures: [String] = []
    mutating func add(_ c: Cmp, relMax: Float) {
        rows.append(c)
        note(c.line + (c.rel > relMax ? "   ✗ (> \(relMax))" : ""))
        if c.rel > relMax { failures.append("\(c.name) rel \(c.rel) > \(relMax)") }
    }
    mutating func say(_ s: String) { notes.append(s); note(s) }
}

/// S0 — key contract: every component loads with `verify: [.all]` (0 missing / 0 unused) and parameter counts match
/// the converted files.
func gateS0(weights: URL) throws {
    let p = try FlashVSRPipeline.load(directory: weights)
    func count(_ m: Module) -> Int { m.parameters().flattened().reduce(0) { $0 + $1.1.size } }
    func keys(_ m: Module) -> Int { m.parameters().flattened().count }
    note("S0 key contract (verify .all): PASS")
    note("  DiT        \(keys(p.dit)) tensors  \(count(p.dit)) params   (golden 825 / 1,418,996,800)")
    note("  LQ proj    \(keys(p.lqProj)) tensors  \(count(p.lqProj)) params   (golden 8 / 287,845,888)")
    note("  TCDecoder  \(keys(p.decoder)) tensors  \(count(p.decoder)) params   (golden 66 / 45,338,371)")
    guard keys(p.dit) == 825, count(p.dit) == 1_418_996_800, keys(p.lqProj) == 8, count(p.lqProj) == 287_845_888,
          keys(p.decoder) == 66, count(p.decoder) == 45_338_371 else { throw SmokeError("S0 count mismatch") }
}

/// S1 — sub-op taps per component, each fed the GOLDEN inputs so a break localises to one component:
///   const (t, t_mod, cross K/V) · LQ projector tokens per chunk · DiT step taps for blocks 0 / 29 (q, k, v, block
///   mask, attention, sub-layer outputs), pred, block-0 K/V cache · TCDecoder · and E2E from (lq, noise).
/// E2E is gated RELATIVE to upstream's own floor (`oracle/drift_floor.py`): the hard top-k block selection lets a
/// 1e-6 input nudge flip a near-tie block, so two correct runs can sit tens of dB apart. Bar = min(90 dB,
/// floor − 6 dB), PSNR at peak 1 on [−1, 1]; 90 dB admits the one systematic difference — fp32 softmax underflow
/// ties (torch CPU keeps denormal block probabilities, MLX flushes them), worth ≤ 1e-3 of the range.
func gateS1(weights: URL, golden: URL, relMax: Float, chainRelMax: Float, e2e: Bool, gpu: Bool,
            floor: URL?) throws {
    try Device.withDefaultDevice(gpu ? Device(.gpu) : Device(.cpu)) {
        let (g, meta) = try loadArraysAndMetadata(url: golden)
        func G(_ k: String) throws -> MLXArray {
            guard let a = g[k] else { throw SmokeError("golden lacks \(k)") }
            return a
        }
        var rep = GateReport()
        let p = try FlashVSRPipeline.load(directory: weights)
        var floorDB = Double.infinity
        if let floor, let j = try JSONSerialization.jsonObject(with: Data(contentsOf: floor)) as? [String: Any],
           let nd = j["nudge_1e-6"] as? [String: Any], let f = nd["psnr"] as? Double { floorDB = f }
        // Where upstream's own 1e-6 nudge floor is below 90 dB (the sparse regime), a chained run from golden inputs
        // is a lottery over near-tie block flips (AB-L-0199): chained rows are reported, not gated. The isolated
        // rows and the floor-relative E2E carry the gate there.
        let chainGated = floorDB >= 90
        func chain(_ c: Cmp) {
            if chainGated { rep.add(c, relMax: chainRelMax) } else { rep.rows.append(c); note(c.line + "   (info: sparse regime)") }
        }
        rep.say("S1 on \(golden.lastPathComponent) — \(gpu ? "GPU" : "CPU") stream, fp32, attention \(FlashVSRAttention.current)"
                + (gpu ? ", MLX_ENABLE_TF32=\(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset")" : ""))

        // const
        let (t, tMod) = p.dit.timeConstants!, (xk, xv) = p.dit.crossKV(block: 0)!
        rep.add(compare("const.t", t, try G("const.t")), relMax: relMax)
        rep.add(compare("const.t_mod", tMod, try G("const.t_mod")), relMax: relMax)
        rep.add(compare("const.cross_k0", xk, try G("const.cross_k0")), relMax: relMax)
        rep.add(compare("const.cross_v0", xv, try G("const.cross_v0")), relMax: relMax)

        // LQ projector (streamed exactly as the pipeline streams it)
        let lq = cl5(try G("input.lq"))
        let fLQ = lq.dim(1), chunks = (fLQ - 1) / 8 - 2
        p.lqProj.reset()
        for k in 0 ..< chunks {
            let clips = k == 0 ? (0 ..< 7).map { (max(0, 4 * $0 - 3), ($0 + 1) * 4 - 3) }
                               : (0 ..< 2).map { (8 * k + 17 + 4 * $0, 8 * k + 21 + 4 * $0) }
            var tok: MLXArray?
            for (a, b) in clips {
                guard let cur = p.lqProj.stream(lq[0..., a ..< b]) else { continue }
                tok = tok.map { concatenated([$0, cur[0]], axis: 1) } ?? cur[0]
            }
            rep.add(compare("c\(k).lq.0", tok!, try G("c\(k).lq.0")), relMax: relMax)
        }

        // DiT — (a) ISOLATED: blocks 0 and 29 each fed the golden block input (and, in chunk 1, the window cache the
        // isolated chunk-0 run of the same block produced) → strict; (b) CHAINED: the whole step from the golden
        // latents + LQ tokens through the port's own caches → fp32 accumulation over 30 blocks, looser bar.
        guard let topk = meta["topk_ratio"].flatMap(Double.init), let local = meta["local_range"].flatMap(Int.init)
        else { throw SmokeError("golden metadata lacks topk_ratio / local_range") }
        rep.say("  topk_ratio \(topk), local_range \(local)")
        var ck = [MLXArray?](repeating: nil, count: 30), cv = ck
        var isoK: [Int: MLXArray] = [:], isoV: [Int: MLXArray] = [:]
        for k in 0 ..< chunks {
            let x = cl5(try G("c\(k).x_in"))
            let (f, h, w) = (x.dim(1), x.dim(2) / 2, x.dim(3) / 2)
            let t0 = k == 0 ? 0 : 4 + 2 * k
            for b in [0, 29] {
                let inK = k == 0 ? nil : (b == 0 ? try G("c0.cache_k0") : isoK[b])
                let inV = k == 0 ? nil : (b == 0 ? try G("c0.cache_v0") : isoV[b])
                // attention under the GOLDEN block mask: the kernel and everything after it, free of the selection
                // rule's underflow ties (the mask itself is compared separately below)
                let gMask = try G("c\(k).b\(b).mask")[0]
                let taps = p.dit.isolatedBlock(b, input: try G("c\(k).b\(b).in"), f: f, h: h, w: w, t0: t0,
                                               topkRatio: topk, kvLen: 3, localRange: local, cacheK: inK, cacheV: inV,
                                               maskOverride: gMask)
                isoK[b] = taps["b\(b).cache_k"]; isoV[b] = taps["b\(b).cache_v"]
                for t in ["sa_in", "q", "k", "v", "attn", "sa_out", "ca_in", "ca_out", "out"] {
                    let key = "c\(k).b\(b).\(t)"
                    rep.add(compare("iso " + key, taps["b\(b).\(t)"]!, try G(key)), relMax: relMax)
                }
                maskReport(&rep, "iso c\(k).b\(b)", taps, b, gMask, seqlen: f / 2, strict: true)
            }
            let r = p.dit.step(x, lq: [try G("c\(k).lq.0")], cacheK: &ck, cacheV: &cv, t0: t0, topkRatio: topk,
                               kvLen: 3, localRange: local, tapBlocks: [0, 29])
            for b in [0, 29] {
                for t in ["in", "attn", "out"] {
                    let key = "c\(k).b\(b).\(t)"
                    chain(compare("chain " + key, r.taps["b\(b).\(t)"]!, try G(key)))
                }
                maskReport(&rep, "chain c\(k).b\(b)", r.taps, b, try G("c\(k).b\(b).mask")[0], seqlen: f / 2,
                           strict: false)
            }
            chain(compare("chain c\(k).pred", r.pred, cl5(try G("c\(k).pred"))))
            chain(compare("chain c\(k).cache_k0", ck[0]!, try G("c\(k).cache_k0")))
            chain(compare("chain c\(k).cache_v0", cv[0]!, try G("c\(k).cache_v0")))
        }

        // TCDecoder on the golden latents + cond, decoded in one call (upstream's call shape)
        p.decoder.reset()
        let dec = p.decoder.decode(clT(try G("dec.latents")), lq: cl5(try G("dec.cond")))
        rep.add(compare("dec.out", dec, clT(try G("dec.out"))), relMax: relMax)
        p.decoder.reset()

        if e2e {
            let noise = cl5(try G("input.noise"))
            // stage by stage: each chunk's latents vs the golden decoder input, frames before the colour fix vs
            // the golden decoder output, then the final frames (per-frame PSNR on failure)
            let gLat = clT(try G("dec.latents")), gDec = clT(try G("dec.out")) * 2 - 1
            var lat: [MLXArray] = []
            var chunkRows: [Cmp] = []
            let raw = try p.run(lq: lq, noise: noise,
                                options: FlashVSROptions(topkRatio: topk, localRange: local, colorFix: false)) { ch in
                lat.append(ch.latents)
                if let gl = g["c\(ch.index).lq.0"] { chunkRows.append(compare("E2E c\(ch.index).lq.0", ch.lqTokens[0], gl)) }
                if let gp = g["c\(ch.index).pred"] { chunkRows.append(compare("E2E c\(ch.index).pred", ch.dit.pred, cl5(gp))) }
                if let gx = g["c\(ch.index).x_in"] {
                    let k = ch.index, (n0, n1) = k == 0 ? (0, 6) : (4 + 2 * k, 6 + 2 * k)
                    chunkRows.append(compare("E2E c\(k).x_in", noise[0..., n0 ..< n1], cl5(gx)))
                }
            }!
            for r in chunkRows { rep.rows.append(r); note(r.line + "   (info)") }
            for r in [compare("E2E latents", concatenated(lat, axis: 1), gLat),
                      compare("E2E pre-colourfix", raw, gDec)] { rep.rows.append(r); note(r.line + "   (info)") }
            let out = try p.run(lq: lq, noise: noise,
                                options: FlashVSROptions(topkRatio: topk, localRange: local))!
            // the streaming API on the same frames + noise: one output per pushed frame, and on the frames upstream
            // emits it must equal the whole-clip run
            let st = FlashVSRStream(pipeline: p, height: lq.dim(2), width: lq.dim(3),
                                    options: FlashVSROptions(topkRatio: topk, localRange: local))
            let extra = MLXRandom.normal([1, 8, noise.dim(2), noise.dim(3), 16], key: MLXRandom.key(99))
            let padded = concatenated([noise, extra], axis: 1)        // the stream runs one chunk past upstream
            st.noiseProvider = { padded[0..., $0] }
            var stLat: [MLXArray] = [], stTok: [MLXArray] = []
            st.onChunk = { stLat.append($0.latents); stTok.append($0.lqTokens[0]) }
            var sf: [MLXArray] = []
            for i in 0 ..< lq.dim(1) { sf += try st.push(lq[0, i]) }
            sf += try st.finish()
            let streamed = stacked(sf, axis: 0).expandedDimensions(axis: 0)
            rep.say("  stream: \(lq.dim(1)) pushed → \(sf.count) out (whole-clip run emits \(out.dim(1)))")
            if sf.count != lq.dim(1) { rep.failures.append("stream emitted \(sf.count) for \(lq.dim(1)) pushed") }
            for k in 0 ..< min(2, stLat.count) {
                note(compare("stream c\(k).lq.0", stTok[k], try G("c\(k).lq.0")).line + "   (info, vs golden)")
                note(compare("stream c\(k).latents", stLat[k], lat[k]).line + "   (info, vs run)")
            }
            let sc = compare("stream vs run", streamed[0..., 0 ..< out.dim(1)], out)
            if sc.rel > 1e-6 {
                let per = (0 ..< out.dim(1)).map { i in
                    String(format: "%d:%.1e", i, abs(streamed[0, i] - out[0, i]).max().item(Float.self))
                }
                note("  stream vs run per-frame max: " + per.joined(separator: " "))
            }
            rep.add(sc, relMax: 1e-6)
            let ref = try G("out.frames").transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
            let d = out - ref
            let psnr = 10 * log10(1 / Double(max((d * d).mean().item(Float.self), 1e-30)))
            let bar = min(90, floorDB - 6)
            let maxAbs = abs(d).max().item(Float.self)
            rep.say(String(format: "  E2E out.frames      psnr %.1f dB (peak 1)  maxAbs %.3e   upstream nudge floor %.1f dB → bar %.1f dB%@",
                           psnr, maxAbs, floorDB, bar, (psnr < bar ? "   ✗" : "") as NSString))
            if psnr < bar { rep.failures.append("E2E \(psnr) dB < bar \(bar) dB") }
            if psnr < bar || floor == nil {
                let per = (0 ..< out.dim(1)).map { i -> String in
                    let d = out[0, i] - ref[0, i]
                    let mse = (d * d).mean().item(Float.self)
                    return String(format: "%d:%.1f", i, 10 * log10(1 / Double(max(mse, 1e-20))))
                }
                rep.say("  per-frame PSNR (peak 1): " + per.joined(separator: " "))
            }
        }
        if rep.failures.isEmpty {
            note("S1 PASS (\(rep.rows.count) comparisons; isolated rel ≤ \(relMax), chained rel ≤ \(chainRelMax); "
                 + "every isolated mask flip a near-tie" + (e2e ? "; E2E at or above the upstream-relative bar)" : ")"))
        } else {
            note("S1 FAIL:\n  " + rep.failures.joined(separator: "\n  "))
            throw SmokeError("S1 failed")
        }
    }
}

/// Block-mask agreement. A flip is a NEAR-TIE when the port's draft probability sits within 1e-3 (relative) of its
/// row threshold, or below 1e-30 (both sides of fp32 softmax underflow) — sub-ulp drift legitimately moves those.
/// Isolated (golden-fed) blocks must have only near-tie flips; chained blocks report them.
func maskReport(_ rep: inout GateReport, _ name: String, _ taps: [String: MLXArray], _ b: Int, _ golden: MLXArray,
                seqlen: Int, strict: Bool) {
    let mine = taps["b\(b).mask"]!, probs = taps["b\(b).probs"]!
    let (heads, nq, nk) = (probs.dim(0), probs.dim(1), probs.dim(2))
    let flat = probs.reshaped([heads * seqlen, (nq / seqlen) * nk])
    // the row threshold exactly as the model computes it is not needed: a flip at p means p is the boundary value
    // on one side — recompute each row's sorted neighbours of p instead
    let flips = (abs(mine - golden) .> 0.5).reshaped([heads * seqlen, (nq / seqlen) * nk])
    let nFlip = flips.asType(.int32).sum().item(Int.self)
    let density = golden.mean().item(Float.self)
    let empty = (golden.sum(axis: -1) .== 0).asType(.int32).sum().item(Int.self)
    var line = String(format: "  %-22@ %d / %d blocks differ (density %.3f, empty q-rows %d)", name as NSString,
                      nFlip, golden.size, density, empty)
    if nFlip > 0 {
        let rows = flips.asType(.float32).asArray(Float.self), pv = flat.asArray(Float.self)
        let n = flat.dim(1)
        var worstGap: Float = 0, maxP: Float = 0, underflow = 0
        for r in 0 ..< flat.dim(0) {
            let row = Array(pv[(r * n) ..< ((r + 1) * n)])
            let sortedRow = row.sorted()
            for c in 0 ..< n where rows[r * n + c] > 0.5 {
                let p = row[c]
                maxP = max(maxP, p)
                if p < 1e-30 { underflow += 1; continue }
                // nearest distinct neighbour in sorted order = how far p is from swapping rank
                let idx = sortedRow.firstIndex(of: p)!
                let lo = idx > 0 ? sortedRow[idx - 1] : p, hi = idx + 1 < n ? sortedRow[idx + 1] : p
                worstGap = max(worstGap, min(abs(p - lo), abs(hi - p)) / max(p, 1e-38))
            }
        }
        line += String(format: "  flipped: %d underflow ties (p < 1e-30), max p %.3e, worst neighbour gap %.2e",
                       underflow, maxP, worstGap)
        if strict && worstGap > 1e-3 { rep.failures.append("\(name) mask: a flip at relative gap \(worstGap)") }
    }
    rep.say(line)
}
