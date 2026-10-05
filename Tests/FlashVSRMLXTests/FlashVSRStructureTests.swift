import Foundation
import MLX
import MLXNN
import XCTest

@testable import FlashVSRMLX

/// Structure suite — CPU stream, no weights, no metallib: the key contract of each network, the layout helpers
/// against brute-force definitions of upstream's rearranges, the LCSA masks, and the chunk bookkeeping.
/// Numerical parity against upstream lives in the smoke CLI (`flashvsr-smoke s1`), which needs the 6 GB checkpoint.
final class FlashVSRStructureTests: XCTestCase {
    override func invokeTest() { Device.withDefaultDevice(Device(.cpu)) { super.invokeTest() } }

    func testKeyContracts() {
        let dit = FlashVSRDiT(), lq = LQProjector(), dec = TCDecoder()
        let dk = dit.parameters().flattened().map(\.0), lk = lq.parameters().flattened().map(\.0)
        let ck = dec.parameters().flattened().map(\.0)
        XCTAssertEqual(dk.count, 825)
        XCTAssertEqual(lk.count, 8)
        XCTAssertEqual(ck.count, 66)
        XCTAssertTrue(dk.contains("blocks.29.cross_attn.norm_k.weight"))
        XCTAssertTrue(dk.contains("blocks.0.ffn.2.bias"))
        XCTAssertTrue(dk.contains("head.modulation"))
        XCTAssertTrue(dk.contains("time_projection.1.weight"))
        XCTAssertFalse(dk.contains { $0.contains("norm1") || $0.contains("norm2") }, "norm1/norm2 carry no affine")
        XCTAssertEqual(Set(lk), ["conv1.weight", "conv1.bias", "norm1.gamma", "conv2.weight", "conv2.bias",
                                 "norm2.gamma", "linear_layers.0.weight", "linear_layers.0.bias"])
        XCTAssertTrue(ck.contains("decoder.9.conv.weight"))          // TGrow(512, 1)
        XCTAssertTrue(ck.contains("decoder.19.conv.4.bias"))         // last MemBlock(128)
        XCTAssertFalse(ck.contains("decoder.3.bias"), "IdentityConv2d has no bias")
        XCTAssertFalse(dit.training || lq.training || dec.training)
    }

    func testWindowPartitionRoundTripAndOrder() {
        let (f, h, w, c) = (4, 16, 24, 3)
        let x = MLXArray(0 ..< Int32(f * h * w * c)).asType(.float32).reshaped([1, f, h, w, c])
        let p = windowPartition(x, f: f, h: h, w: w)
        XCTAssertEqual(p.shape, [(f / 2) * (h / 8) * (w / 8), 128, c])
        // window (nf=1, nh=1, nw=2), token (wf=1, wh=3, ww=5) ↔ x[f=3, h=11, w=21]
        let win = (1 * (h / 8) + 1) * (w / 8) + 2, tok = (1 * 8 + 3) * 8 + 5
        XCTAssertEqual(p[win, tok].asArray(Float.self), x[0, 3, 11, 21].asArray(Float.self))
        let r = windowReverse(p, b: 1, f: f, h: h, w: w)
        XCTAssertEqual(r.reshaped(x.shape).asArray(Float.self), x.asArray(Float.self))
    }

    func testLocalBlockMaskMatchesDefinition() {
        // upstream: key (r', c') visible iff r' ∈ [r − R//2, r − R//2 + R − 1] (same for columns), unclamped
        for (bh, bw, r) in [(3, 4, 11), (6, 10, 3), (5, 5, 4)] {
            let m = localBlockMask(bh: bh, bw: bw, range: r)
            for q in 0 ..< bh * bw { for k in 0 ..< bh * bw {
                let (qr, qc, kr, kc) = (q / bw, q % bw, k / bw, k % bw)
                let lo = { (v: Int) in v - r / 2 }
                let want = kr >= lo(qr) && kr <= lo(qr) + r - 1 && kc >= lo(qc) && kc <= lo(qc) + r - 1
                XCTAssertEqual(m[q * bh * bw + k], want)
            } }
        }
        XCTAssertTrue(localBlockMask(bh: 3, bw: 4, range: 11).allSatisfy { $0 }, "range ≥ grid → dense")
    }

    func testDraftMaskKeepsStrictlyAboveThreshold() {
        // one head, one temporal slice, 2 spatial blocks: scores fixed so the softmax ranks are known
        let s = 2
        let q = MLXArray([Float](repeating: 1, count: 2 * 128 * 128), [2, 128, 128])
        var kv = [Float](repeating: 0, count: 2 * 128 * 128)
        for i in 0 ..< 128 * 128 { kv[i] = 2 }                       // key block 0 scores higher than block 1
        let k = MLXArray(kv, [2, 128, 128])
        let m = draftBlockMask(qW: q, kW: k, heads: 1, qSlices: 1, spatialBlocks: s,
                               local: [Bool](repeating: true, count: 4), topk: 1)
        // flat = [p0, p1, p0, p1], p0 > p1; topk 1 → threshold = 2nd largest = p0 → nothing strictly above
        XCTAssertEqual(m.sum().item(Float.self), 0)
        let m0 = draftBlockMask(qW: q, kW: k, heads: 1, qSlices: 1, spatialBlocks: s,
                                local: [Bool](repeating: true, count: 4), topk: 2)
        // threshold = 3rd largest = p1 → both p0 entries kept
        XCTAssertEqual(m0.asArray(Float.self), [1, 0, 1, 0])
    }

    func testPixelUnshuffleOrders() {
        // LQ projector: channel = c·256 + hh·16 + ww
        let x = MLXArray(0 ..< Int32(32 * 32 * 3)).asType(.float32).reshaped([1, 1, 32, 32, 3])
        let u = LQProjector.unshuffle16(x)
        XCTAssertEqual(u.shape, [1, 1, 2, 2, 768])
        XCTAssertEqual(u[0, 0, 1, 0, 2 * 256 + 5 * 16 + 7].item(Float.self), x[0, 0, 16 + 5, 7, 2].item(Float.self))
        // decoder cond: channel = c·256 + ff·64 + hh·8 + ww, front-padded with frame 0 to a multiple of 4
        let y = MLXArray(0 ..< Int32(5 * 16 * 8 * 3)).asType(.float32).reshaped([1, 5, 16, 8, 3])
        let cnd = TCDecoder.condition(y)
        XCTAssertEqual(cnd.shape, [1, 2, 2, 1, 768])
        XCTAssertEqual(cnd[0, 0, 1, 0, 1 * 256 + 2 * 64 + 3 * 8 + 4].item(Float.self),
                       y[0, 0, 8 + 3, 4, 1].item(Float.self), "padded slot 2 is frame 0")
        XCTAssertEqual(cnd[0, 1, 0, 0, 0 * 256 + 3 * 64 + 1 * 8 + 6].item(Float.self),
                       y[0, 4, 1, 6, 0].item(Float.self), "step 1 slot 3 is frame 4")
    }

    func testChunkPlan() {
        let p = FlashVSRPipeline.plan(sourceFrames: 33)
        XCTAssertEqual(p.lqFrames, 33); XCTAssertEqual(p.chunks, 2); XCTAssertEqual(p.latents, 8)
        XCTAssertEqual(p.outFrames, 29)
        let q = FlashVSRPipeline.plan(sourceFrames: 21)
        XCTAssertEqual(q.lqFrames, 25); XCTAssertEqual(q.chunks, 1); XCTAssertEqual(q.outFrames, 21)
    }

    func testColorFixMatchesLQStatistics() {
        let hq = MLXRandom.normal([1, 2, 8, 8, 3], key: MLXRandom.key(1)) * 0.3
        let lq = MLXRandom.normal([1, 2, 8, 8, 3], key: MLXRandom.key(2)) * 0.2 + 0.1
        let out = FlashVSRPipeline.colorFix(hq, lq: lq)
        let mo = out.mean(axes: [2, 3]), ml = lq.mean(axes: [2, 3])
        XCTAssertLessThan(abs(mo - ml).max().item(Float.self), 1e-5)
    }
}
