import CoreMedia
import CoreVideo
import FlashVSRMLX
import Foundation
import FrameStreamNative
import MLX
import MLXNN
import MLXToolKit

public enum FlashVSRPackageError: Error, CustomStringConvertible {
    /// The converted weights are not on this machine.
    case weightsNotProvisioned(expected: String)
    case unsupportedScale(Int)
    case pixelBuffer(String)

    public var description: String {
        switch self {
        case .weightsNotProvisioned(let p):
            return "FlashVSR weights not found at \(p). The engine downloads them from mlx-community/FlashVSR-v1.1-bf16 "
                + "(or -fp32) on first use; to run without the network, point FlashVSRConfiguration.weightsDirectory at a "
                + "lane directory (`oracle/convert_weights.py <dir> --lane bf16|fp32`)."
        case .unsupportedScale(let s): return "FlashVSR supports ×4 (native) and ×2 (got ×\(s))"
        case .pixelBuffer(let d): return "FlashVSR pixel buffer: \(d)"
        }
    }
}

/// An MLXEngine `videoUpscale` package over **FlashVSR v1.1** (OpenImagingLab, Apache-2.0 code and weights) — a
/// ONE-STEP streaming diffusion video super-resolution model: a Wan2.1-1.3B-shaped DiT with locality-constrained
/// block-sparse attention, a causal LQ projector and a conditioned tiny decoder. Generative: it invents texture at the
/// target resolution. **Recommended for live-action footage only — not anime, cartoons or graphics:** on drawn content
/// it renders flat colour and line art as photographic texture and pushes the drawing toward realism.
///
/// The run streams: AVFoundation decode (frame-stream-native, no FFmpeg) → PIL-bicubic ×scale (the input the model was
/// trained on) → `FlashVSRStream` → HEVC encode. The first chunk needs 25 source frames, then every 8 frames yield 8;
/// every source frame gets exactly one output frame carrying its own PTS (`timedTransform` / `timedFlush`).
///
/// Scale: ×4 native; ×2 = the same model on a ×2 bicubic input (upstream's `scale` argument — 4× fewer tokens).
/// Sizes: the upscaled frame is edge-padded to multiples of 128 and the output cropped back (upstream centre-crops
/// instead — padding keeps every pixel).
@InferenceActor
public final class FlashVSRUpscalePackage: ModelPackage {
    public typealias Configuration = FlashVSRConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: FlashVSR-v1.1 weights are Apache-2.0 (HF model card). C8: the port is derived from upstream's
            // Apache-2.0 code (wan_video_dit.py, flashvsr_tiny.py, utils.py, TCDecoder.py) → Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            // the lane repos are converted from JunhaoZhuang/FlashVSR-v1.1 @ 27561b18 (their config.json pins it)
            provenance: Provenance(sourceRepo: FlashVSRConfiguration.bf16Repo, revision: "main", tier: 2),
            requirements: RequirementsManifest(
                footprints: FlashVSRUpscalePackage.footprints,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                VideoUpscaleContract.descriptor(
                    name: "flashvsr-upscale",
                    summary: "FlashVSR v1.1 one-step diffusion video super-resolution (x4 native, x2). Generative: "
                        + "invents plausible texture at the target resolution; streaming, every frame kept. "
                        + "Recommended for live action only — not anime, cartoons or graphics."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: FlashVSRPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        guard let dir = configuration.resolvedWeightsDirectory(),
              configuration.fileNames.allSatisfy({
                  FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
              }) else {
            throw FlashVSRPackageError.weightsNotProvisioned(
                expected: configuration.resolvedWeightsDirectory()?.path ?? "<no weightsDirectory and no store root>")
        }
        pipeline = try FlashVSRPipeline.load(directory: dir, dtypes: configuration.precision.dtypes)
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()
    }

    /// C14 seam: the loaded pipeline (nil until `load()`).
    var loadedPipeline: FlashVSRPipeline? { pipeline }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()                       // CAN-1: first act, before notLoaded validation
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .videoUpscale, let req = request as? VideoUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let scale = req.scale ?? 4
        guard scale == 2 || scale == 4 else { throw FlashVSRPackageError.unsupportedScale(scale) }

        let tmp = FileManager.default.temporaryDirectory
        let inURL = tmp.appendingPathComponent(UUID().uuidString).appendingPathExtension(req.video.format.rawValue)
        let outURL = tmp.appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
        try req.video.data.write(to: inURL)
        defer {
            try? FileManager.default.removeItem(at: inURL)
            try? FileManager.default.removeItem(at: outURL)
        }
        let info = try await NativeFrameStream.probe(url: inURL)
        let expected = max(1, Int((info.duration * info.frameRate).rounded()))

        let driver = FlashVSRDriver(pipeline: pipeline, configuration: configuration, scale: scale)
        let meta = try await NativeFrameStream.run(
            input: inURL, output: outURL, timing: .preserveSource,
            timedTransform: { frame, pts in
                try Task.checkCancellation()               // CAN cadence: once per decoded source frame
                let out = try driver.ingest(frame, pts: pts)
                RunProgress.report(.upsample, step: driver.emitted, totalSteps: expected)
                return out
            },
            timedFlush: {
                let out = try driver.flush()
                RunProgress.report(.upsample, step: driver.emitted, totalSteps: expected)
                return out
            })
        let data = try Data(contentsOf: outURL)
        return VideoUpscaleResponse(
            video: Video(format: .mp4, data: data, durationSeconds: meta.sourceDuration,
                         frameRate: meta.sourceFrameRate),
            appliedScale: scale)
    }
}

/// One clip: BGRA pixel buffers in, (SR pixel buffer, PTS) out — in order, lagging up to 25 frames.
final class FlashVSRDriver {
    let pipeline: FlashVSRPipeline
    let configuration: FlashVSRConfiguration
    let scale: Int
    private var stream: FlashVSRStream?
    private var pendingPTS: [CMTime] = []
    private var geometry: (srcW: Int, srcH: Int, outW: Int, outH: Int, padW: Int, padH: Int)?
    private var matrices: (y: MLXArray, x: MLXArray)?
    private(set) var ingested = 0, emitted = 0

    init(pipeline: FlashVSRPipeline, configuration: FlashVSRConfiguration, scale: Int) {
        self.pipeline = pipeline
        self.configuration = configuration
        self.scale = scale
    }

    func ingest(_ pb: CVPixelBuffer, pts: CMTime) throws -> [(CVPixelBuffer, CMTime)] {
        let x = try Self.rgb255(from: pb)                                       // (H, W, 3) 0…255
        if geometry == nil {
            let (w, h) = (x.dim(1), x.dim(0))
            let (ow, oh) = (w * scale, h * scale)
            let (pw, ph) = ((ow + 127) / 128 * 128, (oh + 127) / 128 * 128)
            geometry = (w, h, ow, oh, pw, ph)
            matrices = (FlashVSRPreprocess.bicubicMatrix(inN: h, outN: oh),
                        FlashVSRPreprocess.bicubicMatrix(inN: w, outN: ow))
            stream = FlashVSRStream(pipeline: pipeline, height: ph, width: pw, options: configuration.options,
                                    seed: configuration.seed)
        }
        let g = geometry!
        var up = FlashVSRPreprocess.bicubic8(x, outH: g.outH, outW: g.outW, matrices: matrices)
        if g.padH > g.outH || g.padW > g.outW {
            up = padded(up, widths: [.init((0, g.padH - g.outH)), .init((0, g.padW - g.outW)), 0], mode: .edge)
        }
        pendingPTS.append(pts)
        ingested += 1
        return try emit(stream!.push(FlashVSRPreprocess.toModel(up)))
    }

    func flush() throws -> [(CVPixelBuffer, CMTime)] {
        guard let stream, ingested > 0 else { return [] }
        return try emit(stream.finish())
    }

    private func emit(_ frames: [MLXArray]) throws -> [(CVPixelBuffer, CMTime)] {
        guard let g = geometry else { return [] }
        return try frames.map { f in
            let y = (f[0 ..< g.outH, 0 ..< g.outW] + 1) / 2
            emitted += 1
            return (try Self.pixelBuffer(from: y), pendingPTS.removeFirst())
        }
    }

    /// 32BGRA → (H, W, 3) float32 RGB in 0…255.
    static func rgb255(from pb: CVPixelBuffer) throws -> MLXArray {
        guard CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA else {
            throw FlashVSRPackageError.pixelBuffer("expected 32BGRA")
        }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let (w, h, bpr) = (CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), CVPixelBufferGetBytesPerRow(pb))
        guard let base = CVPixelBufferGetBaseAddress(pb) else { throw FlashVSRPackageError.pixelBuffer("no base") }
        var packed = [UInt8](repeating: 0, count: w * h * 4)
        packed.withUnsafeMutableBytes { dst in
            for row in 0 ..< h { memcpy(dst.baseAddress! + row * w * 4, base + row * bpr, w * 4) }
        }
        let bgra = MLXArray(packed, [h, w, 4])
        return concatenated([bgra[.ellipsis, 2 ..< 3], bgra[.ellipsis, 1 ..< 2], bgra[.ellipsis, 0 ..< 1]], axis: -1)
            .asType(.float32)
    }

    /// (H, W, 3) float RGB in [0, 1] → 32BGRA (clamped, rounded, opaque).
    static func pixelBuffer(from x: MLXArray) throws -> CVPixelBuffer {
        let (h, w) = (x.dim(0), x.dim(1))
        let q = (clip(x.asType(.float32), min: 0, max: 1) * 255).round().asType(.uint8)
        let bgra = concatenated([q[.ellipsis, 2 ..< 3], q[.ellipsis, 1 ..< 2], q[.ellipsis, 0 ..< 1],
                                 MLXArray.full([h, w, 1], values: MLXArray(UInt8(255)))], axis: -1)
        let bytes = bgra.reshaped([h * w * 4]).asArray(UInt8.self)
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { throw FlashVSRPackageError.pixelBuffer("allocate \(w)x\(h)") }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let dst = CVPixelBufferGetBaseAddress(buffer)!
        let bpr = CVPixelBufferGetBytesPerRow(buffer)
        bytes.withUnsafeBytes { src in
            for row in 0 ..< h { memcpy(dst + row * bpr, src.baseAddress! + row * w * 4, w * 4) }
        }
        return buffer
    }
}

extension FlashVSRUpscalePackage {
    public nonisolated static var registration: PackageRegistration { .of(FlashVSRUpscalePackage.self) }

    public nonisolated static func requiredBytes(_ precision: FlashVSRPrecision) -> UInt64 {
        guard let fp = footprints.first(where: { $0.quant == precision.quant }) else { return .max }
        return fp.residentBytes + fp.peakActivationBytes
    }

    /// Split footprint — MEASURED through the real MLXServeEngine (`flashvsr-smoke engine`, 2026-10-04, M5 Max,
    /// thermal .fair; one clip per process; basis = process lifetime-peak `phys_footprint`; block-sparse kernel):
    ///
    ///   input → output (×)          lane   phys peak   MLX peak
    ///   320×192 → 640×384 (×2)      bf16    9.18 GB     6.86 GB   (dense attention)
    ///   320×192 → 1280×768 (×4)     bf16   19.11 GB    15.95 GB
    ///   480×288 → 1920×1152 (×4)    bf16   33.68 GB    31.37 GB
    ///   320×192 → 1280×768 (×4)     fp32   34.38 GB    31.55 GB
    ///
    /// Streaming makes memory independent of clip length, so activation scales with OUTPUT PIXELS per frame. bf16:
    /// the line through the 1280×768 and 1920×1152 points, 7.45 GB + 11.86 KB/px of phys (it sits ABOVE the ×2
    /// point: 10.36 vs 9.18 GB), less the 3.51 GB of bf16 weights, ×1.2 headroom → 4.73 GB + 14.23 KB/px, measured
    /// ceiling 1920×1152; the scalar is the representative 1280×768 case, (19.11 − 3.51) × 1.2 + 0.256 ≈ 19.0 GB.
    /// fp32 (the parity lane) was measured at 1280×768 only: a flat line at its scalar, (34.38 − 7.01) × 1.2 +
    /// 0.256 ≈ 33.1 GB, ceiling 1280×768 — the engine refuses fp32 above it.
    nonisolated static var footprints: [QuantFootprint] {
        [
            QuantFootprint(quant: .bf16, residentBytes: 3_600_000_000, peakActivationBytes: 19_000_000_000,
                           activationScaling: ActivationScaling(axis: .outputPixels, baseBytes: 4_730_000_000,
                                                                bytesPerUnit: 14_230, measuredCeiling: 1920 * 1152)),
            QuantFootprint(quant: .fp32, residentBytes: 7_100_000_000, peakActivationBytes: 33_100_000_000,
                           activationScaling: ActivationScaling(axis: .outputPixels, baseBytes: 33_100_000_000,
                                                                bytesPerUnit: 0, measuredCeiling: 1280 * 768)),
        ]
    }
}
