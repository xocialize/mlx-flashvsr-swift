// Causal_LQ4x_Proj — port of FlashVSR `examples/WanVSR/utils/utils.py` (Apache-2.0): the LQ conditioning path.
// The ×4-bicubic LQ video is pixel-unshuffled 16×16 (3 → 768 ch), run through two STREAMING causal Conv3d (k (4,3,3),
// stride (2,1,1)) with RMS norm + SiLU, and projected by a Linear into the DiT width; the result is ADDED to the DiT
// tokens at block 0 (layer_num = 1 in v1.1). Every 4 LQ frames yield one latent frame of tokens.
//
// Upstream's CausalConv3d pads with REPLICATE (not zeros): spatial 1 on each side, temporal 2 at the front — reduced
// by the 2 cached frames once a cache exists (CACHE_T = 2, the last two frames of the conv's own INPUT).
// RMS_norm(images=False): F.normalize over channels × √C × γ.

import Foundation
import MLX
import MLXNN

final class CausalConv3d: Module {
    let weight: MLXArray   // (O, 4, 3, 3, I)
    let bias: MLXArray
    init(_ i: Int, _ o: Int) {
        weight = MLXArray.zeros([o, 4, 3, 3, i])
        bias = MLXArray.zeros([o])
    }
    /// x: (B, T, H, W, C). Replicate-pad (time front 2 − cache, space 1), then a VALID conv, stride (2, 1, 1).
    func callAsFunction(_ x: MLXArray, cache: MLXArray?) -> MLXArray {
        var x = x
        var padT = 2
        if let cache {
            x = concatenated([cache.asType(x.dtype), x], axis: 1)
            padT -= cache.dim(1)
        }
        if padT > 0 { x = concatenated([repeated(x[0..., 0 ..< 1], count: padT, axis: 1), x], axis: 1) }
        x = padded(x, widths: [0, 0, .init((1, 1)), .init((1, 1)), 0], mode: .edge)
        return conv3d(x, weight.asType(x.dtype), stride: [2, 1, 1]) + bias.asType(x.dtype)
    }
}

final class RMSNormChannels: Module {
    let gamma: MLXArray
    init(_ c: Int) { gamma = MLXArray.ones([c]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = x.dim(-1)
        let n = sqrt((x.asType(.float32) * x.asType(.float32)).sum(axis: -1, keepDims: true))
        let normed = x.asType(.float32) / maximum(n, MLXArray(Float(1e-12)))
        return (normed * Float(c).squareRoot()).asType(x.dtype) * gamma.asType(x.dtype)
    }
}

public final class LQProjector: Module {
    let conv1: CausalConv3d, conv2: CausalConv3d
    let norm1: RMSNormChannels, norm2: RMSNormChannels
    @ModuleInfo(key: "linear_layers") var linearLayers: [Linear]
    var cacheConv1: MLXArray?
    var cacheConv2: MLXArray?
    var clipIndex = 0

    public init(outDim: Int = 1536, layers: Int = 1, hidden: (Int, Int) = (2048, 3072)) {
        conv1 = CausalConv3d(768, hidden.0)
        conv2 = CausalConv3d(hidden.0, hidden.1)
        norm1 = RMSNormChannels(hidden.0)
        norm2 = RMSNormChannels(hidden.1)
        _linearLayers.wrappedValue = (0 ..< layers).map { _ in Linear(hidden.1, outDim) }
        super.init()
        train(false)
    }

    public func reset() { cacheConv1 = nil; cacheConv2 = nil; clipIndex = 0 }

    /// 'b c (f 1) (h 16) (w 16) -> b (c 16 16) f h w' in channels-last: channel index c·256 + hh·16 + ww.
    static func unshuffle16(_ x: MLXArray) -> MLXArray {
        let (b, t, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        return x.reshaped([b, t, h / 16, 16, w / 16, 16, c]).transposed(0, 1, 2, 4, 6, 3, 5)
            .reshaped([b, t, h / 16, w / 16, c * 256])
    }

    /// Upstream `stream_forward(video_clip)`. clip: (B, T, H, W, 3) in [−1, 1]. The first call (1 frame, repeated to
    /// 4) only primes the caches and returns nil; every later 4-frame clip returns one latent frame of tokens per layer.
    public func stream(_ clip: MLXArray) -> [MLXArray]? {
        var x = clip.asType(conv1.weight.dtype)
        if clipIndex == 0 {
            x = concatenated([repeated(x[0..., 0 ..< 1], count: 3, axis: 1), x], axis: 1)
            x = Self.unshuffle16(x)
            let c1 = x[0..., (x.dim(1) - 2)...]
            x = conv1(x, cache: cacheConv1)
            cacheConv1 = c1
            x = silu(norm1(x))
            cacheConv2 = x[0..., (x.dim(1) - 2)...]
            eval(cacheConv1!, cacheConv2!)
            clipIndex += 1
            return nil
        }
        x = Self.unshuffle16(x)
        let c1 = x[0..., (x.dim(1) - 2)...]
        x = conv1(x, cache: cacheConv1)
        cacheConv1 = c1
        x = silu(norm1(x))
        let c2 = x[0..., (x.dim(1) - 2)...]
        x = conv2(x, cache: cacheConv2)
        cacheConv2 = c2
        x = silu(norm2(x))
        let (b, t, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let tokens = x.reshaped([b, t * h * w, c])
        clipIndex += 1
        let outs = linearLayers.map { $0(tokens) }
        eval(outs + [cacheConv1!, cacheConv2!])
        return outs
    }
}
