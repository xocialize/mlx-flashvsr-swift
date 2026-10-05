// FlashVSRPackageTests — the package gates, offline: MAT-1..5 (local-only weights until publication), the licence
// posture (C7/C8 Apache-2.0, admitted by a `.blocking` permissive-only host), CAN-1..3, the honest load refusal, and
// the driver's frame/PTS plumbing through the streaming schedule (a miniature random-weight pipeline: plumbing, not
// pixels).

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import MLX
import MLXNN
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import FlashVSRMLX
@testable import MLXFlashVSR

final class FlashVSRPackageTests: XCTestCase {

    // MARK: - MAT

    func testMATGatePerPrecision() throws {
        for p in FlashVSRPrecision.allCases {
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for f in FlashVSRConfiguration(precision: p).fileNames { try Data([0]).write(to: dir.appendingPathComponent(f)) }
            defer { try? FileManager.default.removeItem(at: dir) }
            let report = MaterializationConformance.check(
                freshConfiguration: FlashVSRConfiguration(precision: p),
                satisfiedConfiguration: FlashVSRConfiguration(precision: p, weightsDirectory: dir))
            XCTAssertTrue(report.passed, "\(p):\n\(report.summary)")
        }
    }

    func testSourcesAreTheMLXCommunityLanes() {
        let bf16 = FlashVSRConfiguration(), fp32 = FlashVSRConfiguration(precision: .fp32)
        XCTAssertEqual(bf16.weightSources.map(\.repo), ["mlx-community/FlashVSR-v1.1-bf16"])
        XCTAssertEqual(fp32.weightSources.map(\.repo), ["mlx-community/FlashVSR-v1.1-fp32"])
        XCTAssertEqual(bf16.weightSources.first?.matching?.sorted(),
                       ["config.json", "dit_bf16.safetensors", "lq_proj_bf16.safetensors", "prompt_bf16.safetensors",
                        "tcdecoder_bf16.safetensors"])
        XCTAssertFalse((bf16 as Any) is SelfMaterializing, "the engine materializes the lane from the Hub")
        XCTAssertEqual(bf16.missingWeightSources(storeRoot: nil).count, 1, "fresh machine: missing")
    }

    func testPartialWeightsDirectoryIsMissing() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data([0]).write(to: dir.appendingPathComponent(FlashVSRWeightLane.bf16.fileName(.dit)))
        XCTAssertEqual(FlashVSRConfiguration(weightsDirectory: dir).missingWeightSources(storeRoot: nil).count, 1)
    }

    // MARK: - licence

    func testLicenceIsApacheOnBothLayers() {
        let lic = FlashVSRUpscalePackage.manifest.license
        XCTAssertEqual(lic.weightLicense, .apache2)
        XCTAssertEqual(lic.portCodeLicense, .apache2)
        XCTAssertTrue(LicensePolicy.permissiveOnly.admits(.apache2))
    }

    func testBlockingPermissiveHostRegisters() async throws {
        let engine = MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)
        _ = try await engine.register(FlashVSRUpscalePackage.registration, configuration: FlashVSRConfiguration())
    }

    func testDeclaredFootprintsCoverBothPrecisions() {
        // bf16: ceiling 1920×1152; fp32 (parity lane): 1280×768
        let fp = FlashVSRUpscalePackage.manifest.requirements.footprints
        XCTAssertEqual(fp.first { $0.quant == .bf16 }?.activationScaling?.measuredCeiling, 1920 * 1152)
        XCTAssertEqual(fp.first { $0.quant == .fp32 }?.activationScaling?.measuredCeiling, 1280 * 768)
        let q = Set(FlashVSRUpscalePackage.manifest.requirements.footprints.map(\.quant))
        XCTAssertEqual(q, [.bf16, .fp32])
        XCTAssertLessThan(FlashVSRUpscalePackage.requiredBytes(.bf16), FlashVSRUpscalePackage.requiredBytes(.fp32))
    }

    // MARK: - FIT

    func testFITGate() {
        let report = FootprintConformance.check(manifest: FlashVSRUpscalePackage.manifest,
                                                configuration: FlashVSRConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWorkloadIsOutputPixelsFromTheContainer() throws {
        // a real QuickTime file written by AVFoundation (the test fixture is generated, 2 frames 200×120)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
        defer { try? FileManager.default.removeItem(at: url) }
        try FlashVSRPackageTests.writeTinyMovie(to: url, width: 200, height: 120)
        let data = try Data(contentsOf: url)
        XCTAssertEqual(FlashVSRConfiguration.videoDimensions(data)?.0, 200)
        XCTAssertEqual(FlashVSRConfiguration.videoDimensions(data)?.1, 120)
        let cfg = FlashVSRConfiguration()
        // ×4: 800×480 → padded 896×512
        XCTAssertEqual(cfg.workloadUnits(for: VideoUpscaleRequest(video: Video(format: .mov, data: data))), 896 * 512)
        XCTAssertEqual(cfg.workloadUnits(for: VideoUpscaleRequest(video: Video(format: .mov, data: data), scale: 2)),
                       512 * 256)
        XCTAssertNil(cfg.workloadUnits(for: VideoUpscaleRequest(video: Video(format: .mp4, data: Data([1, 2, 3])))))
    }

    static func writeTinyMovie(to url: URL, width: Int, height: Int) throws {
        let w = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        w.add(input)
        guard w.startWriting() else { throw w.error ?? NSError(domain: "writer", code: 1) }
        w.startSession(atSourceTime: .zero)
        for i in 0 ..< 2 {
            let pb = try FlashVSRDriver.pixelBuffer(from: MLXArray.zeros([height, width, 3]) + 0.5)
            while !input.isReadyForMoreMediaData { usleep(1000) }
            adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 24))
        }
        input.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        w.finishWriting { sem.signal() }
        sem.wait()
        guard w.status == .completed else { throw w.error ?? NSError(domain: "writer", code: 2) }
    }

    // MARK: - CAN

    func testCANPreCancelledRun() async {
        let report = await CancellationConformance.checkRun(
            package: FlashVSRUpscalePackage(configuration: FlashVSRConfiguration()),
            request: VideoUpscaleRequest(video: Video(format: .mp4, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: FlashVSRUpscalePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: FlashVSRUpscalePackage.manifest,
            // once per decoded source frame in the NativeFrameStream transform closure, and once per chunk inside
            // FlashVSRPipeline.chunk; RunProgress(.upsample) at the frame seam
            posture: .cadence([.init(phase: .upsample, unit: .frame, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - load refusal + driver plumbing

    @InferenceActor
    func testLoadRefusesInsteadOfDownloading() async {
        let pkg = FlashVSRUpscalePackage(configuration: FlashVSRConfiguration(
            weightsDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)))
        do {
            try await pkg.load()
            XCTFail("load must refuse when the converted weights are absent")
        } catch let e as FlashVSRPackageError {
            if case .weightsNotProvisioned = e {} else { XCTFail("\(e)") }
        } catch { XCTFail("\(error)") }
    }

    static func tinyPipeline() -> FlashVSRPipeline {
        let c = FlashVSRDiTConfig.tiny()
        let dit = FlashVSRDiT(c)
        // random weights so nothing degenerates to exact zeros/NaN; plumbing, not pixels
        let ps = dit.parameters().flattened().map { ($0.0, MLXRandom.normal($0.1.shape) * 0.02) }
        dit.update(parameters: ModuleParameters.unflattened(ps))
        dit.prepare(context: MLXRandom.normal([1, 8, c.textDim]))
        let lq = LQProjector(outDim: c.dim, hidden: (32, 48))
        lq.update(parameters: ModuleParameters.unflattened(lq.parameters().flattened().map {
            ($0.0, $0.0.hasSuffix("gamma") ? $0.1 : MLXRandom.normal($0.1.shape) * 0.02) }))
        let dec = TCDecoder(channels: [32, 16, 16, 16])
        dec.update(parameters: ModuleParameters.unflattened(dec.parameters().flattened().map {
            ($0.0, MLXRandom.normal($0.1.shape) * 0.05) }))
        return FlashVSRPipeline(dit: dit, lqProj: lq, decoder: dec)
    }

    func testDriverEmitsEveryFrameInOrderWithItsOwnPTS() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            let driver = FlashVSRDriver(pipeline: Self.tinyPipeline(), configuration: FlashVSRConfiguration(),
                                        scale: 4)
            var out: [(CVPixelBuffer, CMTime)] = []
            var counts: [Int] = []
            for i in 0 ..< 27 {
                let px = MLXRandom.uniform(0 ..< 1, [10, 14, 3])                // 40×56 out → padded to 128×128
                out += try driver.ingest(try FlashVSRDriver.pixelBuffer(from: px),
                                         pts: CMTime(value: CMTimeValue(i * 1001), timescale: 24000))
                counts.append(out.count)
            }
            XCTAssertEqual(counts[23], 0, "nothing before the first chunk's 25 frames")
            XCTAssertEqual(counts[24], 21, "the first chunk emits 21 (25 in)")
            out += try driver.flush()
            XCTAssertEqual(out.count, 27, "every source frame gets one output")
            XCTAssertEqual(out.map { $0.1.value }, (0 ..< 27).map { CMTimeValue($0 * 1001) })
            XCTAssertEqual(CVPixelBufferGetWidth(out[0].0), 56)
            XCTAssertEqual(CVPixelBufferGetHeight(out[0].0), 40)
        }
    }

    func testPixelBufferRoundTripIsExactOn8BitLevels() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            let levels = MLXRandom.randInt(0 ..< 256, [6, 9, 3]).asType(.float32)
            let back = try FlashVSRDriver.rgb255(from: FlashVSRDriver.pixelBuffer(from: levels / 255))
            XCTAssertEqual(abs(back - levels).max().item(Float.self), 0)
        }
    }
}
