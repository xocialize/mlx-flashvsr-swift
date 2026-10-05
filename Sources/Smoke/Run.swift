// `flashvsr-smoke run` — a frame directory through the port on the GPU, the way upstream's example script does it:
// PIL-bicubic ×4 to 8 bits, centre-crop to multiples of 128, 4 copies of the last frame appended, F = 8⌊(n+3)/8⌋+1.
import CoreGraphics
import Foundation
import FlashVSRMLX
import ImageIO
import MLX
import UniformTypeIdentifiers

// MARK: - images

/// A PNG/JPEG as (H, W, 3) uint8-valued float32 (0…255).
func loadRGB255(_ path: String) throws -> MLXArray {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError("cannot read \(path)") }
    let (w, h) = (cg.width, cg.height)
    var px = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { throw SmokeError("ctx") }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return MLXArray(px, [h, w, 4])[0..., 0..., 0 ..< 3].asType(.float32)
}

/// (H, W, 3) float in [0, 1] → 8-bit sRGB PNG (round, clamp).
func writePNG(_ x: MLXArray, _ path: String) throws {
    let (h, w) = (x.dim(0), x.dim(1))
    let rgb = (clip(x.asType(.float32), min: 0, max: 1) * 255).round().asType(.uint8).reshaped([h * w * 3])
        .asArray(UInt8.self)
    var px = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0 ..< w * h { px[4 * i] = rgb[3 * i]; px[4 * i + 1] = rgb[3 * i + 1]; px[4 * i + 2] = rgb[3 * i + 2] }
    let cg: CGImage? = px.withUnsafeMutableBytes { p in
        CGContext(data: p.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)?.makeImage()
    }
    guard let cg, let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                            UTType.png.identifier as CFString, 1, nil)
    else { throw SmokeError("png write \(path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SmokeError("png finalize \(path)") }
}

// MARK: - runner

func runFrames(inDir: String, outDir: String, weights: URL, dtypes: FlashVSRDTypes, limit: Int?, seed: UInt64,
               options: FlashVSROptions, cacheMB: Int?) throws {
    // the freed-buffer cache otherwise grows to the working set's high water (14–18 GB on top of the live set)
    if let cacheMB, cacheMB >= 0 { Memory.cacheLimit = cacheMB << 20 }
    var paths = try FileManager.default.contentsOfDirectory(atPath: inDir)
        .filter { $0.hasSuffix(".png") || $0.hasSuffix(".jpg") }.sorted().map { inDir + "/" + $0 }
    if let limit { paths = Array(paths.prefix(limit)) }
    guard paths.count >= 21 else { throw SmokeError("need ≥ 21 frames, have \(paths.count)") }
    try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let plan = FlashVSRPipeline.plan(sourceFrames: paths.count)
    let first = try loadRGB255(paths[0])
    let (sH, sW) = (first.dim(0) * 4, first.dim(1) * 4)
    let (tH, tW) = ((sH / 128) * 128, (sW / 128) * 128)
    let (top, left) = ((sH - tH) / 2, (sW - tW) / 2)
    note("in \(paths.count) frames \(first.dim(1))×\(first.dim(0)) → \(tW)×\(tH); F=\(plan.lqFrames), "
         + "\(plan.chunks) chunks, \(plan.outFrames) out")
    let (pipe, loadS) = try timed { try FlashVSRPipeline.load(directory: weights, dtypes: dtypes) }
    // LQ: the source frames, then copies of the last one, up to F
    var lqFrames: [MLXArray] = []
    for i in 0 ..< plan.lqFrames {
        let img = try loadRGB255(paths[min(i, paths.count - 1)])
        let up = FlashVSRPreprocess.bicubic8(img, outH: sH, outW: sW)[top ..< (top + tH), left ..< (left + tW)]
        lqFrames.append(FlashVSRPreprocess.toModel(up))
    }
    let lq = stacked(lqFrames, axis: 0).expandedDimensions(axis: 0)                  // (1, F, H, W, 3)
    let noise = MLXRandom.normal([1, plan.latents, tH / 8, tW / 8, 16], key: MLXRandom.key(seed))
    eval(lq, noise)
    let phys0 = physFootprint()
    let gpuBefore = agxUtilization(), thermalBefore = ProcessInfo.processInfo.thermalState
    var written = 0, nanCount = 0
    var chunkS: [Double] = []
    var produced: [MLXArray] = []
    var t = DispatchTime.now().uptimeNanoseconds
    // The timed region is the pipeline alone (chunks are evaluated inside `run`): PNG encoding happens after it, so
    // run_s compares like with like against the upstream torch runner, whose run_s also excludes writing frames.
    let (_, runS) = try timed {
        try pipe.run(lq: lq, noise: noise, options: options) { ch in
            let now = DispatchTime.now().uptimeNanoseconds
            chunkS.append(Double(now - t) / 1e9)
            let fr = ch.frames!
            produced.append(fr)
            note(String(format: "  chunk %d: %d frames in %.2f s  (phys %.2f GB, MLX peak %.2f GB)", ch.index,
                        fr.dim(1), chunkS.last!, Double(physFootprint().current) / 1e9,
                        Double(Memory.peakMemory) / 1e9))
            t = DispatchTime.now().uptimeNanoseconds
        }
    }
    for fr in produced {
        nanCount += isNaN(fr).asType(.int32).sum().item(Int.self)
        for i in 0 ..< fr.dim(1) {
            try writePNG((fr[0, i] + 1) / 2, String(format: "%@/f%04d.png", outDir, written))
            written += 1
        }
    }
    let phys = physFootprint()
    for (n, t, pk) in pipe.profile { note(String(format: "  %-12@ %6.2f s  MLX peak %6.2f GB", n as NSString, t, Double(pk) / 1e9)) }
    let rec: [String: Any] = [
        "frames_in": paths.count, "frames_out": written, "target": "\(tW)x\(tH)",
        "dtypes": "dit \(dtypes.dit) lq \(dtypes.lqProj) dec \(dtypes.decoder)",
        "load_s": loadS, "run_s": runS, "s_per_frame": runS / Double(written), "chunk_s": chunkS,
        "peak_phys_gb": Double(phys.peak) / 1e9, "phys_before_run_gb": Double(phys0.current) / 1e9,
        "mlx_peak_gb": Double(Memory.peakMemory) / 1e9,
        "topk_ratio": options.topkRatio ?? FlashVSRPipeline.defaultTopkRatio(height: tH, width: tW),
        "local_range": options.localRange, "seed": seed, "cache_limit_mb": cacheMB ?? -1, "nan_values": nanCount,
        // timing validity (shared machine): the GPU must be idle before the run and the machine .nominal
        "gpu_util_before_pct": gpuBefore ?? -1, "thermal_before": "\(thermalBefore.rawValue)",
        "thermal_after": "\(ProcessInfo.processInfo.thermalState.rawValue)",
        "profile": pipe.profile.map { ["stage": $0.0, "s": $0.1, "mlx_peak_gb": Double($0.2) / 1e9] },
    ]
    let json = try JSONSerialization.data(withJSONObject: rec, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: URL(fileURLWithPath: outDir + "/run.json"))
    note(String(data: json, encoding: .utf8)!)
}

/// (current, lifetime-peak) phys_footprint of this process, bytes — proc_pid_rusage(RUSAGE_INFO_V4).
func physFootprint() -> (current: UInt64, peak: UInt64) {
    var info = rusage_info_v4()
    let rc = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
    }
    return rc == 0 ? (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint) : (0, 0)
}

func timed<R>(_ body: () throws -> R) rethrows -> (R, Double) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let r = try body()
    return (r, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
}

/// AGX "Device Utilization %" (the unfiltered `ioreg -r -d 1 -c AGXAccelerator` — the `-k PerformanceStatistics`
/// form reads 0 under load, AB-L-0196).
func agxUtilization() -> Int? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
    p.arguments = ["-r", "-d", "1", "-c", "AGXAccelerator"]
    let pipe = Pipe()
    p.standardOutput = pipe
    guard (try? p.run()) != nil else { return nil }
    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    p.waitUntilExit()
    guard let r = out.range(of: #""Device Utilization %"=(\d+)"#, options: .regularExpression) else { return nil }
    return Int(out[r].split(separator: "=").last ?? "")
}
