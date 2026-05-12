/**
 * @file test_inference_matmul_routing.cpp
 * @brief Validate SDRInferenceEngine::matmulForLayer routes HSDR-encoded
 * segments through the fused-inference path, and falls back to dense matmul
 * for non-HSDR segments. Closes the loop for the inference-engine migration.
 */
#include "ai_compression/SparseInferenceEngine.hpp"
#include "ai_compression/core/AICompressor.hpp"
#include "ai_compression/core/AIModelParser.hpp"
#include "ai_compression/core/ModelSegment.hpp"
#include "ai_compression/strategies/GzipStrategy.hpp"
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace CortexAICompression;

namespace {

class StubParser : public IAIModelParser {
public:
    explicit StubParser(std::vector<ModelSegment> segs) : segs_(std::move(segs)) {}
    std::vector<ModelSegment> parse(const std::string&) const override { return segs_; }
private:
    std::vector<ModelSegment> segs_;
};

ModelSegment makeFP32(const std::string& name, uint32_t R, uint32_t C, uint32_t seed) {
    ModelSegment s;
    s.type = SegmentType::WEIGHTS_FP32;
    s.name = name;
    s.original_size = static_cast<size_t>(R) * C * sizeof(float);
    s.data.resize(s.original_size);
    s.data_format = "f32";
    TensorMetadata m;
    m.dimensions = {R, C};
    s.tensor_metadata = m;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    float* w = reinterpret_cast<float*>(s.data.data());
    for (size_t i = 0; i < R * C; ++i) w[i] = dist(rng);
    return s;
}

}  // namespace

int main() {
    std::cout << "Inference matmul-routing test\n";

    // Build TWO segments: one will go through HSDR (priority 1), the other
    // gets Gzip (priority 2 lossless fallback because HSDR's 1D-row mode
    // requires shape divisibility we deliberately violate).
    auto hsdr_seg = makeFP32("layer.hsdr.weight", /*R=*/256, /*C=*/512, 0xA1);
    auto gzip_seg = makeFP32("layer.gzip.weight", /*R=*/96,  /*C=*/256, 0xB2);

    // For the HSDR segment, use a 1-D row-tile config so matmulForLayer can
    // route it through fused matmul.
    HierarchicalSDRConfig cfg = HierarchicalSDRConfig::forRow1D(
        /*row_width=*/512, /*n_atoms=*/64, /*k_per_stage=*/6);
    cfg.ksvd_iters = 6;
    auto hsdrStrategy = std::make_shared<HierarchicalSDRStrategy>(cfg, cfg);

    // Compress via the standard AICompressor pipeline so the resulting archive
    // exercises the real loader path (not just direct strategy calls).
    auto parser = std::make_unique<StubParser>(std::vector<ModelSegment>{hsdr_seg, gzip_seg});
    AICompressor compressor(std::move(parser));
    constexpr uint8_t HSDR_STRATEGY_ID = 5;
    constexpr uint8_t GZIP_STRATEGY_ID = 3;
    compressor.registerStrategy(SegmentType::WEIGHTS_FP32, /*priority=*/1,
                                HSDR_STRATEGY_ID, hsdrStrategy);
    compressor.registerStrategy(SegmentType::WEIGHTS_FP32, /*priority=*/2,
                                GZIP_STRATEGY_ID, std::make_shared<GzipStrategy>());

    const std::string archive_path = (std::filesystem::temp_directory_path()
                                      / "matmul_routing_test.sdr").string();
    {
        std::ofstream out(archive_path, std::ios::binary | std::ios::trunc);
        if (!out) { std::cerr << "FAIL: cannot open " << archive_path << "\n"; return 1; }
        compressor.compressModel("", out);
    }
    std::cout << "  wrote archive: " << archive_path << "\n";

    SDRModelLoader loader(archive_path);
    SDRInferenceEngine engine(loader);

    // Confirm strategy assignment.
    const auto* hsdr_hdr = loader.findSegmentHeader("layer.hsdr.weight");
    const auto* gzip_hdr = loader.findSegmentHeader("layer.gzip.weight");
    if (!hsdr_hdr || !gzip_hdr) { std::cerr << "FAIL: segments missing\n"; return 1; }
    std::cout << "  layer.hsdr.weight  strategy_id=" << int(hsdr_hdr->compression_strategy_id)
              << " (expect 5=HSDR)\n";
    std::cout << "  layer.gzip.weight  strategy_id=" << int(gzip_hdr->compression_strategy_id)
              << " (expect 3=Gzip)\n";
    if (hsdr_hdr->compression_strategy_id != HSDR_STRATEGY_ID
        || gzip_hdr->compression_strategy_id != GZIP_STRATEGY_ID) {
        std::cerr << "FAIL: unexpected strategy assignment\n";
        return 1;
    }

    // ----------------------------------------------------------------------
    // 1. HSDR path: matmulForLayer should call fused matmulHSDR.
    // ----------------------------------------------------------------------
    const size_t batch = 4;
    std::vector<float> x_hsdr(512 * batch);
    {
        std::mt19937 rng(0xDEAD);
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (auto& v : x_hsdr) v = d(rng);
    }
    auto t0 = std::chrono::steady_clock::now();
    auto Y_hsdr = engine.matmulForLayer("layer.hsdr.weight", x_hsdr.data(), batch);
    auto t1 = std::chrono::steady_clock::now();
    if (Y_hsdr.size() != 256 * batch) {
        std::cerr << "FAIL: hsdr output size " << Y_hsdr.size() << " != " << (256 * batch) << "\n";
        return 1;
    }
    // Reference via loader.matmulHSDR directly — should produce identical bytes.
    auto Y_ref = loader.matmulHSDR("layer.hsdr.weight", x_hsdr.data(), batch);
    double max_d = 0.0;
    for (size_t i = 0; i < Y_ref.size(); ++i) {
        max_d = std::max(max_d, static_cast<double>(std::abs(Y_ref[i] - Y_hsdr[i])));
    }
    std::cout << "  HSDR path:   " << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms, max-vs-direct-matmulHSDR diff = " << max_d << "\n";
    if (max_d > 1e-6) {
        std::cerr << "FAIL: matmulForLayer(HSDR) drifted from direct matmulHSDR\n";
        return 1;
    }

    // ----------------------------------------------------------------------
    // 2. Gzip path: matmulForLayer should fall back to dense matmul on the
    //    decompressed FP32 weights. Verify by comparing to the reference dense.
    // ----------------------------------------------------------------------
    std::vector<float> x_gzip(256 * batch);
    {
        std::mt19937 rng(0xBEEF);
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (auto& v : x_gzip) v = d(rng);
    }
    auto Y_gzip = engine.matmulForLayer("layer.gzip.weight", x_gzip.data(), batch);
    if (Y_gzip.size() != 96 * batch) {
        std::cerr << "FAIL: gzip output size " << Y_gzip.size() << " != " << (96 * batch) << "\n";
        return 1;
    }
    // Reference: load + dense matmul.
    auto seg = loader.loadSegmentByName("layer.gzip.weight");
    const float* W = reinterpret_cast<const float*>(seg.data.data());
    std::vector<float> Y_dense(96 * batch, 0.0f);
    for (size_t r = 0; r < 96; ++r) {
        for (size_t c = 0; c < 256; ++c) {
            const float w = W[r * 256 + c];
            for (size_t b = 0; b < batch; ++b) {
                Y_dense[r * batch + b] += w * x_gzip[c * batch + b];
            }
        }
    }
    double max_g = 0.0;
    for (size_t i = 0; i < Y_dense.size(); ++i) {
        max_g = std::max(max_g, static_cast<double>(std::abs(Y_dense[i] - Y_gzip[i])));
    }
    std::cout << "  Gzip path:   max-vs-reference-dense diff = " << max_g << "\n";
    if (max_g > 1e-5) {
        std::cerr << "FAIL: Gzip-fallback matmul drifted from reference dense\n";
        return 1;
    }

    // Missing-name should throw cleanly.
    try {
        engine.matmulForLayer("does.not.exist", x_hsdr.data(), batch);
        std::cerr << "FAIL: missing-name should have thrown\n";
        return 1;
    } catch (const std::exception& e) {
        std::cout << "  missing-name throws as expected: " << e.what() << "\n";
    }

    std::error_code ec;
    std::filesystem::remove(archive_path, ec);
    std::cout << "PASS\n";
    return 0;
}
