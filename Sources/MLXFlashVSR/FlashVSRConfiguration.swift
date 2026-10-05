import FlashVSRMLX
import Foundation
import MLX
import MLXToolKit

/// Compute precision. `bf16` is the default (upstream's own CUDA dtype; inside the fp32 seed spread on every
/// full-reference metric of the ×4 cells at half the memory). `fp32` is the parity lane the S1/E2E gates run.
public enum FlashVSRPrecision: String, Codable, Sendable, CaseIterable {
    case bf16
    case fp32

    var quant: Quant { self == .bf16 ? .bf16 : .fp32 }
    var dtypes: FlashVSRDTypes { self == .bf16 ? .production : .parity }
}

/// Init-time configuration for `FlashVSRUpscalePackage` (C9).
///
/// Materialization (contract 1.24): one Hub repo per precision lane in mlx-community's `<UpstreamRepoName>-<precision>`
/// grammar — `mlx-community/FlashVSR-v1.1-bf16` (3.5 GB, the default) and `-fp32` (7.0 GB, the parity lane) — so a
/// fresh machine's engine downloads exactly the lane it runs into its model store before `load()`. Both are this
/// repo's `oracle/convert_weights.py` output from HF `JunhaoZhuang/FlashVSR-v1.1` (Apache-2.0). `weightsDirectory`
/// (a local lane directory) always wins and never touches the network.
public struct FlashVSRConfiguration: PackageConfiguration, ModelStorable {
    public static let bf16Repo = "mlx-community/FlashVSR-v1.1-bf16"
    public static let fp32Repo = "mlx-community/FlashVSR-v1.1-fp32"
    public static func repo(for precision: FlashVSRPrecision) -> String { precision == .fp32 ? fp32Repo : bf16Repo }

    public var precision: FlashVSRPrecision
    /// Noise seed (the model is one-step generative: the seed picks the texture it invents).
    public var seed: UInt64
    /// LCSA sparsity: nil → upstream's default 2·768·1280 / (H·W) at the output size.
    public var topkRatio: Double?
    /// LCSA local window in 128-token blocks (upstream example: 11).
    public var localRange: Int
    /// AdaIN colour fix against the LQ frame (upstream default: on).
    public var colorFix: Bool
    /// Explicit directory holding the four files; host-specific → excluded from `Codable`.
    public var weightsDirectory: URL?
    /// Stamped by the engine from its `ModelStore`; excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(precision: FlashVSRPrecision = .bf16, seed: UInt64 = 0, topkRatio: Double? = nil,
                localRange: Int = 11, colorFix: Bool = true, weightsDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.precision = precision
        self.seed = seed
        self.topkRatio = topkRatio
        self.localRange = localRange
        self.colorFix = colorFix
        self.weightsDirectory = weightsDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case precision, seed, topkRatio, localRange, colorFix }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        precision = try c.decodeIfPresent(FlashVSRPrecision.self, forKey: .precision) ?? .bf16
        seed = try c.decodeIfPresent(UInt64.self, forKey: .seed) ?? 0
        topkRatio = try c.decodeIfPresent(Double.self, forKey: .topkRatio)
        localRange = try c.decodeIfPresent(Int.self, forKey: .localRange) ?? 11
        colorFix = try c.decodeIfPresent(Bool.self, forKey: .colorFix) ?? true
    }

    /// The lane a precision runs on, and its files.
    public var lane: FlashVSRWeightLane { precision == .fp32 ? .fp32 : .bf16 }
    public var fileNames: [String] { lane.files }
    /// The repo this configuration's lane materializes from.
    public var repo: String { Self.repo(for: precision) }

    /// Where `load()` reads the lane: the explicit directory if set, else the store's flat repo directory
    /// (`<root>/models--mlx-community--FlashVSR-v1.1-bf16/`, where the engine's materializer lands files).
    public func resolvedWeightsDirectory(storeRoot: URL? = nil) -> URL? {
        if let weightsDirectory { return weightsDirectory }
        return ModelStore(root: storeRoot ?? modelsRootDirectory).directory(for: repo)
    }

    var options: FlashVSROptions {
        FlashVSROptions(topkRatio: topkRatio, localRange: localRange, colorFix: colorFix)
    }
}

/// Fresh-machine sources: one role per lane, so a bf16 configuration never pulls the 7 GB fp32 set.
extension FlashVSRConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "flashvsr-v1.1-\(lane.rawValue)", repo: repo, revision: "main", matching: fileNames)]
    }

    /// Explicit `weightsDirectory` first (complete → nothing missing, incomplete → the lane is missing), then the
    /// default probe over the store.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsDirectory {
            let present = fileNames.allSatisfy {
                FileManager.default.fileExists(atPath: weightsDirectory.appendingPathComponent($0).path)
            }
            return present ? [] : weightSources
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}

extension FlashVSRConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolvedWeightsDirectory() else { return [] }
        return fileNames.filter { $0.hasSuffix(".safetensors") }.map { dir.appendingPathComponent($0) }
    }
}

extension FlashVSRConfiguration: QuantConfigured {
    public var quant: Quant { precision.quant }
}

/// FIT-3: map a request to output pixels per frame (the declared `ActivationScaling` axis), so the engine can refuse a
/// run past the measured ceiling BEFORE loading 4–7 GB of weights. The canonical `Video` carries no geometry, so the
/// dimensions are read from the container's video `tkhd` box (16.16 fixed-point width/height) — synchronous, a few
/// microseconds, no AVFoundation. Output = scale × source, padded up to multiples of 128 (what the model computes on).
/// `nil` (unparseable container, other request types) is never refused.
extension FlashVSRConfiguration: WorkloadDeclaring {
    public func workloadUnits(for request: any CapabilityRequest) -> Double? {
        guard let req = request as? VideoUpscaleRequest, let (w, h) = Self.videoDimensions(req.video.data) else {
            return nil
        }
        let s = req.scale ?? 4
        let pad = { (n: Int) in (n * s + 127) / 128 * 128 }
        return Double(pad(w) * pad(h))
    }

    /// Width × height of the first `tkhd` with a non-zero size (the video track) in an ISO-BMFF / QuickTime file.
    static func videoDimensions(_ data: Data) -> (Int, Int)? {
        let bytes = [UInt8](data)
        let tag: [UInt8] = Array("tkhd".utf8)
        var i = 4
        while i + 4 <= bytes.count {
            if bytes[i] == tag[0] && bytes[i + 1] == tag[1] && bytes[i + 2] == tag[2] && bytes[i + 3] == tag[3] {
                let body = i + 4                                   // version(1) flags(3) …
                guard body < bytes.count else { return nil }
                let off = body + (bytes[body] == 1 ? 88 : 76)      // width, height: 16.16 fixed, big-endian
                if off + 8 <= bytes.count {
                    func u32(_ o: Int) -> Int {
                        (Int(bytes[o]) << 24) | (Int(bytes[o + 1]) << 16) | (Int(bytes[o + 2]) << 8) | Int(bytes[o + 3])
                    }
                    let (w, h) = (u32(off) >> 16, u32(off + 4) >> 16)
                    if w > 0 && h > 0 { return (w, h) }
                }
                i = body
            } else {
                i += 1
            }
        }
        return nil
    }
}
