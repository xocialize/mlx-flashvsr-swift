// The LQ preparation upstream's example performs before the model sees a frame: PIL `Image.resize(BICUBIC)` to the
// target scale on 8-bit RGB, then [0, 255] → [−1, 1]. The model was trained on bicubic-upscaled input, so the resize
// kernel is part of the contract, not a free choice.
import Foundation
import MLX

public enum FlashVSRPreprocess {
    /// PIL bicubic weights for one axis (a = −0.5; support 2 × max(1, in/out); taps clipped to the image and
    /// renormalised — PIL does not replicate edges). (outN, inN), row-stochastic.
    public static func bicubicMatrix(inN: Int, outN: Int) -> MLXArray {
        func cubic(_ x: Double) -> Double {
            let a = -0.5, x = abs(x)
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        }
        let scale = Double(inN) / Double(outN), fs = max(scale, 1.0), support = 2.0 * fs
        var m = [Float](repeating: 0, count: outN * inN)
        for o in 0 ..< outN {
            let center = (Double(o) + 0.5) * scale
            let lo = max(0, Int((center - support + 0.5).rounded(.down)))
            let hi = min(inN, Int((center + support + 0.5).rounded(.down)))
            var ws: [Double] = []
            for i in lo ..< hi { ws.append(cubic((Double(i) - center + 0.5) / fs)) }
            let sum = ws.reduce(0, +)
            for (j, i) in (lo ..< hi).enumerated() { m[o * inN + i] = Float(ws[j] / sum) }
        }
        return MLXArray(m, [outN, inN])
    }

    /// PIL's two-pass 8-bit resize of (H, W, 3) values in 0…255: horizontal pass rounded and clipped to 8 bits, then
    /// vertical. Returns (outH, outW, 3) in 0…255 (float32).
    public static func bicubic8(_ img: MLXArray, outH: Int, outW: Int, matrices: (y: MLXArray, x: MLXArray)? = nil)
        -> MLXArray {
        let (h, w) = (img.dim(0), img.dim(1))
        let my = matrices?.y ?? bicubicMatrix(inN: h, outN: outH)
        let mx = matrices?.x ?? bicubicMatrix(inN: w, outN: outW)
        let x = img.asType(.float32).transposed(2, 0, 1)                                // (3, H, W)
        let tmp = clip(matmul(x, mx.transposed()).round(), min: 0, max: 255)            // (3, H, outW)
        let out = clip(matmul(my, tmp).round(), min: 0, max: 255)                      // (3, outH, outW)
        return out.transposed(1, 2, 0)
    }

    /// 0…255 → [−1, 1]
    public static func toModel(_ x255: MLXArray) -> MLXArray { x255 / 255 * 2 - 1 }
}
