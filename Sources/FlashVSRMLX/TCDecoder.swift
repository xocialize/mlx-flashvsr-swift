// TCDecoder — port of FlashVSR `examples/WanVSR/utils/TCDecoder.py` (Apache-2.0): a widened TAEHV decoder
// (madebyollin's tiny video autoencoder shape) CONDITIONED on the LQ video — each latent frame is concatenated with a
// (4, 8, 8) pixel-unshuffle of four LQ frames (768 ch) before decoding, so the input is 16 + 768 = 784 channels.
//
// Decoded SEQUENTIALLY (upstream `parallel=False`): each MemBlock sees the previous time step's INPUT to itself as
// `past` (zeros on the first step); TGrow(stride) fans one step out into `stride` steps (channel groups in order),
// processed depth-first exactly like upstream's work queue. Two TGrow×2 stages → 4 frames per latent; the first
// `framesToTrim = 3` outputs of a clip are dropped.
//
// Layer indices are upstream's `decoder.N` (the identity-deepened Sequential):
//   0 Clamp · 1 conv 784→512 · 2 ReLU · 3 IdConv 512 · 4 ReLU · 5–7 MemBlock 512 · 8 Upsample ×2 · 9 TGrow(512, 1) ·
//   10 conv 512→256 · 11–13 MemBlock 256 · 14 Upsample · 15 TGrow(256, 2) · 16 conv 256→128 · 17–19 MemBlock 128 ·
//   20 Upsample · 21 TGrow(128, 2) · 22 conv 128→128 · 23 ReLU · 24 IdConv 128 · 25 ReLU · 26 conv 128→3
//
// GPU numerics: mlx's fp32 3×3 conv2d loses ~7 bits at ≥ 192 input channels on Metal (AB-L-0163); every 3×3 conv
// here at ≥ 192 in-channels is routed through conv3d with kT = 1, which is exact.

import Foundation
import MLX
import MLXNN

final class ExactConv2d: Module, UnaryLayer {
    let weight: MLXArray       // (O, kH, kW, I)
    let bias: MLXArray?
    let padding: Int
    init(_ i: Int, _ o: Int, kernel: Int = 3, bias: Bool = true) {
        weight = MLXArray.zeros([o, kernel, kernel, i])
        self.bias = bias ? MLXArray.zeros([o]) : nil
        padding = kernel / 2
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = weight.asType(x.dtype)
        var y: MLXArray
        if weight.dim(1) == 3 && weight.dim(3) >= 192 {
            // conv3d route (kT = 1): (B, H, W, C) → (B, 1, H, W, C), padding (0, p, p) — exact on Metal
            y = conv3d(x.expandedDimensions(axis: 1), w.expandedDimensions(axis: 1),
                       padding: [0, padding, padding]).squeezed(axis: 1)
        } else {
            y = conv2d(x, w, padding: .init(padding))
        }
        if let bias { y = y + bias.asType(x.dtype) }
        return y
    }
}

final class MemBlock: Module {
    @ModuleInfo var conv: [UnaryLayer]
    init(_ n: Int) {
        _conv.wrappedValue = [ExactConv2d(2 * n, n), ReLU(), ExactConv2d(n, n), ReLU(), ExactConv2d(n, n)]
    }
    func callAsFunction(_ x: MLXArray, past: MLXArray) -> MLXArray {
        relu(seq(conv, concatenated([x, past], axis: -1)) + x)
    }
}

final class TGrow: Module {
    let conv: ExactConv2d
    let stride: Int
    init(_ n: Int, stride: Int) {
        conv = ExactConv2d(n, n * stride, kernel: 1, bias: false)
        self.stride = stride
    }
}

final class Clamp: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { tanh(x / 3) * 3 }
}

final class Upsample2x: Module, UnaryLayer {   // nn.Upsample(scale_factor=2), mode nearest
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return broadcast(x.reshaped([b, h, 1, w, 1, c]), to: [b, h, 2, w, 2, c]).reshaped([b, 2 * h, 2 * w, c])
    }
}

public final class TCDecoder: Module {
    @ModuleInfo var decoder: [Module]
    var mem: [MLXArray?]
    var trimmed = false
    public static let framesToTrim = 3

    public init(channels ch: [Int] = [512, 256, 128, 128]) {
        let layers: [Module] = [
            Clamp(), ExactConv2d(784, ch[0]), ReLU(), ExactConv2d(ch[0], ch[0], bias: false), ReLU(),
            MemBlock(ch[0]), MemBlock(ch[0]), MemBlock(ch[0]), Upsample2x(), TGrow(ch[0], stride: 1),
            ExactConv2d(ch[0], ch[1], bias: false),
            MemBlock(ch[1]), MemBlock(ch[1]), MemBlock(ch[1]), Upsample2x(), TGrow(ch[1], stride: 2),
            ExactConv2d(ch[1], ch[2], bias: false),
            MemBlock(ch[2]), MemBlock(ch[2]), MemBlock(ch[2]), Upsample2x(), TGrow(ch[2], stride: 2),
            ExactConv2d(ch[2], ch[3], bias: false), ReLU(), ExactConv2d(ch[3], ch[3], bias: false), ReLU(),
            ExactConv2d(ch[3], 3),
        ]
        _decoder.wrappedValue = layers
        mem = Array(repeating: nil, count: layers.count)
        super.init()
        train(false)
    }

    public func reset() { mem = Array(repeating: nil, count: decoder.count); trimmed = false }

    /// (4, 8, 8) pixel-unshuffle of LQ frames (B, T, H, W, 3), front-padded with the first frame to a multiple of 4:
    /// 'b c (f 4) (h 8) (w 8) -> b f (c 4 8 8) h w' → channel index c·256 + ff·64 + hh·8 + ww.
    public static func condition(_ lq: MLXArray) -> MLXArray {
        var x = lq
        let t = x.dim(1)
        if t % 4 != 0 { x = concatenated([repeated(x[0..., 0 ..< 1], count: 4 - t % 4, axis: 1), x], axis: 1) }
        let (b, tt, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        return x.reshaped([b, tt / 4, 4, h / 8, 8, w / 8, 8, c]).transposed(0, 1, 3, 5, 7, 2, 4, 6)
            .reshaped([b, tt / 4, h / 8, w / 8, c * 256])
    }

    /// FLASHVSR_DEC_TRACE=1: evaluate after every layer and print the active / peak MLX memory (diagnostics only).
    static let trace = ProcessInfo.processInfo.environment["FLASHVSR_DEC_TRACE"] == "1"

    func run(_ x: MLXArray, from i: Int, into out: inout [MLXArray]) {
        if Self.trace, i > 0 {
            eval(x)
            FileHandle.standardError.write(String(format: "    dec[%2d] %@ %@  active %.2f GB  peak %.2f GB  cache %.2f GB\n",
                i - 1, String(describing: type(of: decoder[i - 1])) as NSString, x.shape.description as NSString,
                Double(Memory.activeMemory) / 1e9, Double(Memory.peakMemory) / 1e9,
                Double(Memory.cacheMemory) / 1e9).data(using: .utf8)!)
        }
        if i == decoder.count {
            // one finished frame: materialise it and the memory it updated, so the next frame's graph starts from
            // concrete arrays (left lazy, a time step's four full-resolution frames become one graph whose
            // intermediates MLX keeps alive together — 42 GB at 1280×768 instead of ~3)
            eval([x] + mem.compactMap { $0 })
            out.append(x)
            return
        }
        switch decoder[i] {
        case let m as MemBlock:
            let past = mem[i] ?? MLXArray.zeros(like: x)
            mem[i] = x
            run(m(x, past: past), from: i + 1, into: &out)
        case let g as TGrow:
            let c = x.dim(-1)
            // each branch is a contiguous copy, so the stride·C-wide TGrow output is not held across the subtrees
            let y = g.conv(x)
            let parts = (0 ..< g.stride).map { s in contiguous(y[.ellipsis, (s * c) ..< ((s + 1) * c)]) }
            eval(parts)
            for p in parts { run(p, from: i + 1, into: &out) }
        case let u as UnaryLayer:
            run(u(x), from: i + 1, into: &out)
        default:
            fatalError("TCDecoder: unexpected layer \(type(of: decoder[i]))")
        }
    }

    /// Upstream `decode_video(latents, parallel=False, cond=lq)`: latents (B, T, h, w, 16), lq (B, 4T−?, H, W, 3).
    /// Returns frames (B, T', H, W, 3) in ~[0, 1] (the pipeline maps to [−1, 1]).
    public func decode(_ latents: MLXArray, lq: MLXArray) -> MLXArray {
        let cond = Self.condition(lq).asType(latents.dtype)
        precondition(cond.dim(1) == latents.dim(1), "cond steps \(cond.dim(1)) != latent steps \(latents.dim(1))")
        let x = concatenated([cond, latents], axis: -1)
        var frames: [MLXArray] = []
        for t in 0 ..< x.dim(1) {
            var out: [MLXArray] = []
            run(x[0..., t], from: 0, into: &out)
            eval(out + mem.compactMap { $0 })
            frames.append(contentsOf: out)
        }
        var y = stacked(frames, axis: 1)
        if !trimmed { y = y[0..., Self.framesToTrim...]; trimmed = true }
        return y
    }
}
