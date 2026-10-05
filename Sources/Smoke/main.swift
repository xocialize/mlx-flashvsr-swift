// flashvsr-smoke — parity gates and runners for the FlashVSR MLX port.
//   flashvsr-smoke s0 <weightsDir>
//   flashvsr-smoke s1 <weightsDir> <golden.safetensors> [--rel 1e-4] [--e2e]
import Foundation
import FlashVSRMLX
import MLX

struct SmokeError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func flag(_ name: String, _ args: [String]) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let args = Array(CommandLine.arguments.dropFirst())
// gate modes compare fp32 to fp32: keep M5's TF32 matmul off unless asked (must precede the first MLX op)
if ["s1", "diag-mask", "diag-stream", "probe-attn"].contains(args.first ?? "") && !args.contains("--tf32") { setenv("MLX_ENABLE_TF32", "0", 1) }
do {
    switch args.first {
    case "s0":
        try gateS0(weights: URL(fileURLWithPath: args[1]))
    case "s1":
        let rel = flag("--rel", args).flatMap(Float.init) ?? 1e-4
        let chainRel = flag("--chain-rel", args).flatMap(Float.init) ?? 1e-3
        try gateS1(weights: URL(fileURLWithPath: args[1]), golden: URL(fileURLWithPath: args[2]), relMax: rel,
                   chainRelMax: chainRel, e2e: args.contains("--e2e"), gpu: args.contains("--gpu"),
                   floor: flag("--floor", args).map { URL(fileURLWithPath: $0) })
    case "diag-mask":
        try diagMaskSensitivity(weights: URL(fileURLWithPath: args[1]), golden: URL(fileURLWithPath: args[2]),
                                gpu: args.contains("--gpu"))
    case "diag-stream":
        try diagStream(weights: URL(fileURLWithPath: args[1]), golden: URL(fileURLWithPath: args[2]),
                       gpu: args.contains("--gpu"))
    case "check-lanes":
        try checkLanes(fp32Dir: URL(fileURLWithPath: args[1]), bf16Dir: URL(fileURLWithPath: args[2]))
    case "probe-attn":
        try probeAttention()
    case "engine":   // top-level await — never block the main thread on a semaphore (it deadlocks the engine)
        try await runEngine(args)
    case "run":
        func dt(_ f: String, _ d: DType) -> DType {
            switch flag(f, args) { case "bf16": .bfloat16; case "fp16": .float16; case "fp32": .float32; default: d }
        }
        let dtypes = FlashVSRDTypes(dit: dt("--dtype", .float32), lqProj: dt("--lq-dtype", .float32),
                                    decoder: dt("--dec-dtype", .float32))
        var opts = FlashVSROptions()
        opts.topkRatio = flag("--topk", args).flatMap(Double.init)
        if let l = flag("--local", args).flatMap(Int.init) { opts.localRange = l }
        try runFrames(inDir: args[1], outDir: args[2], weights: URL(fileURLWithPath: args[3]), dtypes: dtypes,
                      limit: flag("--limit", args).flatMap(Int.init),
                      seed: flag("--seed", args).flatMap(UInt64.init) ?? 0, options: opts,
                      cacheMB: flag("--cache-mb", args).flatMap(Int.init) ?? 2048)
    default:
        note("""
        usage: flashvsr-smoke s0 <weightsDir>
               flashvsr-smoke s1 <weightsDir> <golden> [--rel R] [--chain-rel R] [--e2e [--floor drift.json]] [--gpu [--tf32]]
               flashvsr-smoke diag-mask <weightsDir> <golden> [--gpu]
               flashvsr-smoke engine <in.mov|mp4> <out.mp4> <weightsDir> [--scale 2|4] [--fp32] [--seed S] [--warmup clip]
               flashvsr-smoke run <lrFramesDir> <outDir> <weightsDir> [--dtype|--lq-dtype|--dec-dtype fp32|bf16|fp16]
                              [--limit N] [--seed S] [--topk R] [--local N] [--cache-mb 2048]
        """)
        exit(2)
    }
} catch {
    note("ERROR: \(error)")
    exit(1)
}
