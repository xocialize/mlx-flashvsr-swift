// Weight loading — one precision lane per directory, exactly as `oracle/convert_weights.py` writes it and as the
// published repos hold it (mlx-community/FlashVSR-v1.1-bf16, -fp32): `<component>_<lane>.safetensors` + config.json.
// Keys are upstream's verbatim, conv weights channels-last. Each component loads with `verify: [.all]` (no missing,
// no unused key — the S0 contract) and is cast to the requested compute dtype.

import Foundation
import MLX
import MLXNN

public enum FlashVSRComponent: String, CaseIterable, Sendable {
    case dit
    case lqProj = "lq_proj"
    case decoder = "tcdecoder"
    case prompt
}

/// A published precision lane. bf16 = every tensor rounded to bf16 (bit-identical to casting the fp32 lane at
/// load); fp32 = the release precision, with the two bf16-released parts (LQ projector, prompt) upcast exactly.
public enum FlashVSRWeightLane: String, CaseIterable, Sendable {
    case bf16, fp32
    public func fileName(_ c: FlashVSRComponent) -> String { "\(c.rawValue)_\(rawValue).safetensors" }
    /// Everything a lane directory holds.
    public var files: [String] { FlashVSRComponent.allCases.map(fileName) + ["config.json"] }
}

/// Per-component compute dtypes. The released files are DiT fp32, LQ projector bf16, decoder fp32.
///   • `parity` — fp32 everywhere: the gate lane (S1/E2E against upstream's CPU-fp32 goldens).
///   • `production` — bf16 everywhere, upstream's own CUDA dtype. On the ×4 cells it sits inside the seed spread of
///     fp32 by every full-reference metric (bench/x4_quality.csv: SSIMULACRA2 −44.13 vs −44.28/−44.24 on X4-Wp,
///     −16.36 vs −16.11/−16.88 on X4-Hp) at about half the memory (MLX peak 17.3 vs 32.9 GB at 1280×768).
public struct FlashVSRDTypes: Sendable {
    public var dit: DType, lqProj: DType, decoder: DType
    public init(dit: DType = .float32, lqProj: DType = .float32, decoder: DType = .float32) {
        self.dit = dit; self.lqProj = lqProj; self.decoder = decoder
    }
    public static let parity = FlashVSRDTypes()
    public static let production = FlashVSRDTypes(dit: .bfloat16, lqProj: .bfloat16, decoder: .bfloat16)
}

public enum FlashVSRLoadError: Error, CustomStringConvertible {
    case missing(String)
    public var description: String {
        switch self { case .missing(let p): "FlashVSR weights missing: \(p)" }
    }
}

public extension FlashVSRPipeline {
    /// Load a lane directory. Each component is read from the lane matching its compute dtype when present, else
    /// from the other lane, then cast — so either published repo serves either `FlashVSRDTypes`.
    static func load(directory: URL, dtypes: FlashVSRDTypes = .parity) throws -> FlashVSRPipeline {
        func arrays(_ c: FlashVSRComponent, _ dt: DType?) throws -> [String: MLXArray] {
            let preferred: FlashVSRWeightLane = (dt ?? .bfloat16) == .float32 ? .fp32 : .bf16
            let lanes = [preferred] + FlashVSRWeightLane.allCases.filter { $0 != preferred }
            guard let url = lanes.map({ directory.appendingPathComponent($0.fileName(c)) })
                .first(where: { FileManager.default.fileExists(atPath: $0.path) })
            else { throw FlashVSRLoadError.missing(directory.appendingPathComponent(preferred.fileName(c)).path) }
            let a = try loadArrays(url: url)
            guard let dt else { return a }
            return a.mapValues { $0.asType(dt) }
        }
        let dit = FlashVSRDiT()
        try dit.update(parameters: ModuleParameters.unflattened(arrays(.dit, dtypes.dit)), verify: [.all])
        let lq = LQProjector()
        try lq.update(parameters: ModuleParameters.unflattened(arrays(.lqProj, dtypes.lqProj)), verify: [.all])
        let dec = TCDecoder()
        try dec.update(parameters: ModuleParameters.unflattened(arrays(.decoder, dtypes.decoder)), verify: [.all])
        eval(dit, lq, dec)
        guard let ctx = try arrays(.prompt, dtypes.dit)["context"] else { throw FlashVSRLoadError.missing("prompt context") }
        dit.prepare(context: ctx)
        return FlashVSRPipeline(dit: dit, lqProj: lq, decoder: dec)
    }
}
