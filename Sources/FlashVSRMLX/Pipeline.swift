// The streaming driver — port of FlashVSR `diffsynth/pipelines/flashvsr_tiny.py` `FlashVSRTinyPipeline.__call__`
// (Apache-2.0) with `if_buffer = True`, `cfg = 1`, one step at t = 1000.
//
// Chunk schedule (latent frames; one latent = 4 LQ frames, the first latent = 1):
//   chunk 0  LQ clips [0:1] (primes the projector) then [1:5] … [21:25] → 6 latents, noise[0:6],  RoPE t0 = 0
//   chunk k  LQ clips [8k+17 : 8k+21] and [8k+21 : 8k+25]            → 2 latents, noise[4+2k : 6+2k], t0 = 4+2k
// latents = noise − pred (one Euler step from σ = 1). The decoder is sequential with carried state, so decoding each
// chunk as it lands equals upstream's decode-at-the-end: chunk 0 decodes against LQ[0:21] (front-padded by 3 to 24),
// chunk k against LQ[8k+13 : 8k+21]; out frame i ↔ LQ frame i. Upstream emits 8(P−1)+21 frames for an input of
// F = 8(P+2)+1 (P chunks) — the host pads the tail (`FlashVSRPipeline.plan`).
//
// Colour fix: upstream's AdaIN branch (per frame, per channel; biased variance + 1e-5) against the LQ frame, clamped
// to [−1, 1].

import Foundation
import MLX
import MLXNN

public struct FlashVSROptions: Sendable {
    /// nil → upstream's example default 2·768·1280 / (H·W) at the OUTPUT size.
    public var topkRatio: Double?
    public var kvRatio: Double = 3
    public var localRange: Int = 11
    public var colorFix: Bool = true
    public init(topkRatio: Double? = nil, kvRatio: Double = 3, localRange: Int = 11, colorFix: Bool = true) {
        self.topkRatio = topkRatio; self.kvRatio = kvRatio; self.localRange = localRange; self.colorFix = colorFix
    }
}

/// What one chunk produced (also the gate taps when requested).
public struct FlashVSRChunk {
    public let index: Int
    public let lqTokens: [MLXArray]
    public let latents: MLXArray       // (1, f, h, w, 16)
    public let dit: DiTChunkResult
    public var frames: MLXArray?       // (1, n, H, W, 3) in [−1, 1] when decoded
}

public final class FlashVSRPipeline {
    /// Per-stage memory/time trace (FLASHVSR_PROFILE=1): stage → (seconds, MLX peak bytes since the stage began).
    public var profile: [(String, Double, Int)] = []
    public var profiling = ProcessInfo.processInfo.environment["FLASHVSR_PROFILE"] == "1"
    func stage<R>(_ name: String, _ body: () throws -> R) rethrows -> R {
        guard profiling else { return try body() }
        Memory.peakMemory = 0
        let t0 = DispatchTime.now().uptimeNanoseconds
        let r = try body()
        profile.append((name, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9, Memory.peakMemory))
        return r
    }
    public let dit: FlashVSRDiT
    public let lqProj: LQProjector
    public let decoder: TCDecoder

    public init(dit: FlashVSRDiT, lqProj: LQProjector, decoder: TCDecoder) {
        self.dit = dit; self.lqProj = lqProj; self.decoder = decoder
    }

    /// Frame bookkeeping for `n` source frames (upstream's example: append 4 copies of the last frame, then take
    /// F = 8⌊(n+3)/8⌋ + 1). Returns the LQ length F the pipeline consumes, the chunk count P, the latent count and the
    /// number of output frames 8(P−1)+21.
    public static func plan(sourceFrames n: Int) -> (lqFrames: Int, chunks: Int, latents: Int, outFrames: Int) {
        let f = ((n + 4 - 1) / 8) * 8 + 1
        let p = (f - 1) / 8 - 2
        return (f, p, (f - 1) / 4, 8 * (p - 1) + 21)
    }

    public static func defaultTopkRatio(height: Int, width: Int) -> Double { 2.0 * 768 * 1280 / Double(height * width) }

    /// AdaIN colour fix: hq, lq (B, T, H, W, 3) → per (frame, channel) re-standardised to the LQ statistics.
    /// The statistics reduce over H·W as the INNERMOST contiguous axis: with channels-last data, a reduction over
    /// (H, W) strides past C and MLX's CPU backend accumulates it far less precisely (65.6 dB vs upstream on the
    /// gates, every pixel moved by up to 1.2e-3); transposed to (B, T, C, H·W) it is a contiguous last-axis reduction.
    public static func colorFix(_ hq: MLXArray, lq: MLXArray) -> MLXArray {
        func stats(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let (b, t, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
            let xf = contiguous(x.asType(.float32).transposed(0, 1, 4, 2, 3)).reshaped([b, t, c, h * w])
            let mean = xf.mean(axis: -1, keepDims: true)
            let v = ((xf - mean) * (xf - mean)).mean(axis: -1, keepDims: true) + 1e-5
            return (mean.reshaped([b, t, 1, 1, c]), sqrt(v).reshaped([b, t, 1, 1, c]))
        }
        let (sm, ss) = stats(lq), (cm, cs) = stats(hq)
        let out = (hq.asType(.float32) - cm) / cs * ss + sm
        return clip(out, min: -1, max: 1).asType(hq.dtype)
    }

    /// The per-run state the DiT carries between chunks (the projector and decoder hold their own).
    public struct ChunkState {
        var cacheK: [MLXArray?], cacheV: [MLXArray?]
        init(blocks: Int) { cacheK = .init(repeating: nil, count: blocks); cacheV = cacheK }
    }

    /// LQ frame windows chunk `k` reads: the projector clips, and the decoder/colour-fix condition (whose frame i is
    /// output frame i). `needed` = exclusive end of the LQ frames the chunk touches.
    public static func schedule(chunk k: Int) -> (clips: [Range<Int>], cond: Range<Int>, latents: Range<Int>,
                                                  needed: Int) {
        if k == 0 {
            return ((0 ..< 7).map { i in max(0, 4 * i - 3) ..< (i + 1) * 4 - 3 }, 0 ..< 21, 0 ..< 6, 25)
        }
        return ((0 ..< 2).map { j in (8 * k + 17 + 4 * j) ..< (8 * k + 21 + 4 * j) }, (8 * k + 13) ..< (8 * k + 21),
                (4 + 2 * k) ..< (6 + 2 * k), 8 * k + 25)
    }

    /// Reset the stateful modules for a new clip.
    public func begin() -> ChunkState {
        lqProj.reset()
        decoder.reset()
        return ChunkState(blocks: dit.blocks.count)
    }

    /// One chunk of the streaming schedule. `frames(r)` returns LQ frames r as (1, |r|, H, W, 3) in [−1, 1];
    /// `noise`: this chunk's latents' noise (1, f, H/8, W/8, 16).
    public func chunk(_ k: Int, frames: (Range<Int>) -> MLXArray, noise x: MLXArray, topkRatio: Double,
                      options: FlashVSROptions, decode: Bool = true, tapBlocks: Set<Int> = [],
                      state: inout ChunkState) throws -> FlashVSRChunk {
        try Task.checkCancellation()
        let sch = Self.schedule(chunk: k)
        let tokens: [MLXArray]? = stage("c\(k).lqproj") {
            var tokens: [MLXArray]? = nil
            for r in sch.clips {
                guard let cur = lqProj.stream(frames(r)) else { continue }
                tokens = tokens.map { zip($0, cur).map { concatenated([$0, $1], axis: 1) } } ?? cur
            }
            return tokens
        }
        let t0 = sch.latents.lowerBound
        let r = stage("c\(k).dit") {
            let r = dit.step(x, lq: tokens ?? [], cacheK: &state.cacheK, cacheV: &state.cacheV, t0: t0,
                             topkRatio: topkRatio, kvLen: Int(options.kvRatio), localRange: options.localRange,
                             tapBlocks: tapBlocks)
            eval(r.pred)
            return r
        }
        let latents = x.asType(r.pred.dtype) - r.pred
        eval(latents)
        var out = FlashVSRChunk(index: k, lqTokens: tokens ?? [], latents: latents, dit: r)
        if decode {
            let lqc = frames(sch.cond)
            out.frames = stage("c\(k).decode") {
                var f = decoder.decode(latents.asType(decoder.dtype), lq: lqc.asType(decoder.dtype))
                f = f.asType(.float32) * 2 - 1
                if options.colorFix { f = Self.colorFix(f, lq: lqc[0..., 0 ..< f.dim(1)]) }
                eval(f)
                return f
            }
        }
        return out
    }

    /// Run every chunk of a whole clip (upstream's call shape — the parity gates use it). `lq`: (1, F, H, W, 3) in
    /// [−1, 1] (the ×4-bicubic upscale, H and W multiples of 128, F = 8k+1 ≥ 25); `noise`: (1, (F−1)/4, H/8, W/8, 16).
    /// `onChunk` sees each chunk as it lands. Returns all output frames (1, 8(P−1)+21, H, W, 3) in [−1, 1].
    @discardableResult
    public func run(lq: MLXArray, noise: MLXArray, options: FlashVSROptions = .init(), decode: Bool = true,
                    tapBlocks: Set<Int> = [], onChunk: ((FlashVSRChunk) throws -> Void)? = nil) throws -> MLXArray? {
        let (fLQ, hO, wO) = (lq.dim(1), lq.dim(2), lq.dim(3))
        precondition(hO % 128 == 0 && wO % 128 == 0, "LQ size \(wO)×\(hO) must be a multiple of 128")
        precondition(fLQ % 8 == 1 && fLQ >= 25, "LQ frame count \(fLQ) must be 8k+1 ≥ 25")
        let topkRatio = options.topkRatio ?? Self.defaultTopkRatio(height: hO, width: wO)
        var state = begin()
        var outs: [MLXArray] = []
        for k in 0 ..< (fLQ - 1) / 8 - 2 {
            let ch = try chunk(k, frames: { lq[0..., $0] }, noise: noise[0..., Self.schedule(chunk: k).latents],
                               topkRatio: topkRatio, options: options, decode: decode, tapBlocks: tapBlocks,
                               state: &state)
            if let f = ch.frames { outs.append(f) }
            try onChunk?(ch)
        }
        return decode ? concatenated(outs, axis: 1) : nil
    }
}

/// Frame-in / frame-out streaming over `FlashVSRPipeline` — the shape an engine host drives. Push LQ frames
/// (already ×4-upscaled, H×W multiples of 128, [−1, 1]); chunks run as soon as their frames are present
/// (25 for the first, then every 8) and return their output frames in order; `finish()` replicates the last frame to
/// complete the final chunk and returns the rest, so EVERY pushed frame gets exactly one output frame (upstream
/// drops the source's last up-to-8 frames; on the frames both emit, the two are identical for the same noise).
/// Noise is drawn per latent index from `seed`, so a stream is reproducible. LQ memory is bounded: frames below
/// the next chunk's first use are released. One stream per pipeline at a time (the modules are stateful).
public final class FlashVSRStream {
    public let pipeline: FlashVSRPipeline
    public let height: Int, width: Int
    let options: FlashVSROptions
    let topkRatio: Double
    let seed: UInt64
    let noiseDType: DType
    var state: FlashVSRPipeline.ChunkState
    var buffer: [Int: MLXArray] = [:]       // absolute frame index → (1, 1, H, W, 3)
    var pushed = 0, nextChunk = 0, emitted = 0
    var total: Int?                         // set by finish()

    public init(pipeline: FlashVSRPipeline, height: Int, width: Int, options: FlashVSROptions = .init(),
                seed: UInt64 = 0) {
        precondition(height % 128 == 0 && width % 128 == 0, "LQ size \(width)×\(height) must be a multiple of 128")
        self.pipeline = pipeline; self.height = height; self.width = width; self.options = options
        self.topkRatio = options.topkRatio ?? FlashVSRPipeline.defaultTopkRatio(height: height, width: width)
        self.seed = seed
        self.noiseDType = .float32
        state = pipeline.begin()
    }

    /// Overrides the noise source (gates replay a golden's noise tensor): latent range → (1, |r|, H/8, W/8, 16).
    public var noiseProvider: ((Range<Int>) -> MLXArray)?
    /// Sees each chunk as it completes (progress, gates).
    public var onChunk: ((FlashVSRChunk) -> Void)?

    /// Noise for latents r: N(0, 1), latent j from its own key (seed ⊕ golden-ratio hash of j), so any chunk can be
    /// regenerated alone.
    func noise(_ r: Range<Int>) -> MLXArray {
        if let noiseProvider { return noiseProvider(r) }
        let parts = r.map { j in
            MLXRandom.normal([1, 1, height / 8, width / 8, 16],
                             key: MLXRandom.key(seed ^ (UInt64(j + 1) &* 0x9E37_79B9_7F4A_7C15)))
        }
        return concatenated(parts, axis: 1)
    }

    func frame(_ i: Int) -> MLXArray {
        if let f = buffer[i] { return f }
        precondition(total != nil && i >= pushed, "LQ frame \(i) was released or never pushed")
        return buffer[pushed - 1]!                     // past the end: replicate the last frame
    }

    /// Push one LQ frame (H, W, 3) or (1, H, W, 3). Returns any output frames now complete, each (H, W, 3).
    public func push(_ f: MLXArray) throws -> [MLXArray] {
        precondition(total == nil, "push after finish")
        let x = f.ndim == 3 ? f.expandedDimensions(axis: 0) : f
        precondition(x.dim(1) == height && x.dim(2) == width, "frame \(x.shape) is not \(height)×\(width)")
        buffer[pushed] = x.expandedDimensions(axis: 0)
        pushed += 1
        return try drain()
    }

    /// End of input: complete the last chunk(s) by replicating the final frame; returns the remaining output frames.
    public func finish() throws -> [MLXArray] {
        precondition(pushed > 0, "finish with no frames")
        total = pushed
        return try drain()
    }

    func drain() throws -> [MLXArray] {
        var out: [MLXArray] = []
        while true {
            let sch = FlashVSRPipeline.schedule(chunk: nextChunk)
            if let total { if emitted >= total { break } } else if pushed < sch.needed { break }
            let ch = try pipeline.chunk(nextChunk, frames: { r in concatenated(r.map { self.frame($0) }, axis: 1) },
                                        noise: noise(sch.latents), topkRatio: topkRatio, options: options,
                                        state: &state)
            onChunk?(ch)
            let f = ch.frames!
            let keep = min(f.dim(1), (total ?? Int.max) - emitted)
            for i in 0 ..< keep { out.append(f[0, i]) }
            emitted += keep
            nextChunk += 1
            // the next chunk's earliest LQ use is its condition window start (8(k+1)+13)
            let low = FlashVSRPipeline.schedule(chunk: nextChunk).cond.lowerBound
            for i in buffer.keys where i < low && i != pushed - 1 { buffer[i] = nil }
        }
        return out
    }
}

extension TCDecoder {
    var dtype: DType { (decoder[1] as! ExactConv2d).weight.dtype }
}
