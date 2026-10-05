import Foundation
import MLX
import MLXFlashVSR
import MLXServeCore
import MLXToolKit

// flashvsr-smoke engine <in.mov|mp4> <out.mp4> <laneDir | --store storeRoot> [--scale 2|4] [--fp32] [--seed S]
//                       [--warmup clip]
//   With `--store`, no weightsDirectory is set: the engine materializes the lane from mlx-community into that model
//   store before load (the fresh-machine path).
//   FlashVSRUpscalePackage through the REAL MLXServeEngine (register → run; the engine constructs and loads the
//   package, C13). Reports load + run wall time, s/frame, and the memory the manifest's split footprint is declared
//   from: the post-load phys floor (MLX cache cleared after an optional warm-up of the same size) and the process's
//   lifetime-peak phys_footprint during the run. One clip per process.
func runEngine(_ args: [String]) async throws {
    guard args.count >= 3 else { throw SmokeError("engine <in> <out.mp4> <laneDir | --store dir> [--scale N] [--fp32] [--seed S] [--warmup clip]") }
    let scale = flag("--scale", args).flatMap(Int.init) ?? 4
    let store = flag("--store", args).map { URL(fileURLWithPath: $0) }
    let cfg = FlashVSRConfiguration(precision: args.contains("--fp32") ? .fp32 : .bf16,
                                    seed: flag("--seed", args).flatMap(UInt64.init) ?? 0,
                                    weightsDirectory: store == nil ? URL(fileURLWithPath: args[3]) : nil)
    let input = try Data(contentsOf: URL(fileURLWithPath: args[1]))
    // --store is the fresh-machine path: anonymous Hub access (the lanes are public), so no Keychain read — an
    // unsigned CLI touching a Keychain item blocks on an OS access prompt
    let engine = store == nil ? MLXServeEngine() : MLXServeEngine(hfTokenProvider: { nil })
    if let store {
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        await engine.useModelStore(ModelStore(root: store))
        note("model store \(store.path): missing before run = \(cfg.missingWeightSources(storeRoot: store).map(\.repo))")
    }
    let (id, tReg) = try await timedAsync {
        try await engine.register(FlashVSRUpscalePackage.registration, configuration: cfg)
    }
    note("registered in \(String(format: "%.3f", tReg)) s; licence advisories: \(await engine.licenseAdvisories.map(\.summary))")
    func fmt(_ p: String) -> Video.Format { URL(fileURLWithPath: p).pathExtension.lowercased() == "mov" ? .mov : .mp4 }
    if let warm = flag("--warmup", args) {
        let w = try Data(contentsOf: URL(fileURLWithPath: warm))
        let (_, tw) = try await timedAsync {
            try await engine.run(VideoUpscaleRequest(video: Video(format: fmt(warm), data: w), scale: scale), package: id)
        }
        note(String(format: "warm-up (load + first run) %.2f s", tw))
    }
    MLX.Memory.clearCache()
    let floor = physFootprint().current
    let (r, t) = try await timedAsync {
        try await engine.run(VideoUpscaleRequest(video: Video(format: fmt(args[1]), data: input), scale: scale),
                             package: id)
    }
    let peak = physFootprint().peak
    if let store {
        note("model store after run: missing = \(cfg.missingWeightSources(storeRoot: store).map(\.repo)); "
             + "lane dir = \(cfg.resolvedWeightsDirectory(storeRoot: store)?.path ?? "nil")")
    }
    guard let resp = r as? VideoUpscaleResponse else { throw SmokeError("bad response") }
    try resp.video.data.write(to: URL(fileURLWithPath: args[2]))
    let frames = Int(((resp.video.durationSeconds ?? 0) * (resp.video.frameRate ?? 0)).rounded())
    print(String(format: "engine: ×%d %@ — %.2f s for ~%d frames (%.0f ms/frame incl. decode+encode); "
                 + "phys floor %.2f GB, lifetime peak %.2f GB (activation ≈ %.2f GB); MLX peak %.2f GB; %d bytes out",
                 resp.appliedScale, cfg.precision.rawValue, t, frames, 1000 * t / Double(max(frames, 1)),
                 Double(floor) / 1e9, Double(peak) / 1e9, Double(peak - floor) / 1e9,
                 Double(Memory.peakMemory) / 1e9, resp.video.data.count))
}

func timedAsync<R>(_ body: () async throws -> R) async rethrows -> (R, Double) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let r = try await body()
    return (r, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
}
