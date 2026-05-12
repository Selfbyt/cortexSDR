/**
 * @file test_hsdr_archive_e2e.cpp
 * @brief End-to-end: compress a synthetic FP32 weight via HSDR through the
 * real AICompressor pipeline → write a .sdr archive → open via SDRModelLoader →
 * call matmulHSDR → compare to a dense reference.
 *
 * Exercises the strategy registration, the archive format, the segment index,
 * and the new loader API. Confirms the HSDR fused path works through the same
 * channels production inference will use.
 */
#include "ai_compression/core/AICompressor.hpp"
#include "ai_compression/core/AIModelParser.hpp"
#include "ai_compression/core/ModelSegment.hpp"
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/SparseInferenceEngine.hpp"

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

/// Stub parser that returns one pre-built FP32 weight segment. Lets us drive
/// the AICompressor pipeline without any real model on disk.
class StubParser : public IAIModelParser {
public:
    explicit StubParser(ModelSegment seg) : seg_(std::move(seg)) {}
    std::vector<ModelSegment> parse(const std::string& /*modelPath*/) const override {
        return {seg_};
    }
private:
    ModelSegment seg_;
};

}  // namespace

int main() {
    std::cout << "HSDR archive end-to-end test\n";

    // ----------------------------------------------------------------------
    // 1. Build a deterministic FP32 weight matrix shaped for 1D row tiles.
    // ----------------------------------------------------------------------
    const uint32_t R = 128;     // output dim
    const uint32_t C = 512;     // input dim (= tile_cols in 1D row mode)
    std::vector<float> weight(static_cast<size_t>(R) * C);
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (auto& v : weight) v = dist(rng);

    // ----------------------------------------------------------------------
    // 2. Wrap it as a ModelSegment with TensorMetadata (HSDR needs dims).
    // ----------------------------------------------------------------------
    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_FP32;
    seg.name = "test.layer.weight";
    seg.data.resize(weight.size() * sizeof(float));
    std::memcpy(seg.data.data(), weight.data(), seg.data.size());
    seg.original_size = seg.data.size();
    seg.data_format = "f32";
    TensorMetadata meta;
    meta.dimensions = {R, C};
    meta.sparsity_ratio = 0.0f;
    seg.tensor_metadata = meta;

    // ----------------------------------------------------------------------
    // 3. Run the segment through AICompressor with HSDR registered at priority 1.
    // ----------------------------------------------------------------------
    auto parser = std::make_unique<StubParser>(seg);
    AICompressor compressor(std::move(parser));

    HierarchicalSDRConfig cfg = HierarchicalSDRConfig::forRow1D(
        /*row_width=*/static_cast<uint16_t>(C),
        /*n_atoms=*/64,
        /*k_per_stage=*/6);
    cfg.ksvd_iters = 6;
    auto strategy = std::make_shared<HierarchicalSDRStrategy>(cfg, cfg);
    constexpr uint8_t HSDR_STRATEGY_ID = 5;
    compressor.registerStrategy(SegmentType::WEIGHTS_FP32, /*priority=*/1,
                                HSDR_STRATEGY_ID, strategy);

    const std::string archive_path = (std::filesystem::temp_directory_path()
                                      / "hsdr_e2e_test.sdr").string();
    {
        std::ofstream archive(archive_path, std::ios::binary | std::ios::trunc);
        if (!archive) {
            std::cerr << "FAIL: cannot open " << archive_path << " for write\n";
            return 1;
        }
        // compressModel ignores the path argument here; the StubParser returns our segment regardless.
        compressor.compressModel("", archive);
    }
    std::cout << "  wrote archive: " << archive_path << "\n";
    auto archive_size = std::filesystem::file_size(archive_path);
    std::cout << "  archive size: " << archive_size << " bytes\n";

    // ----------------------------------------------------------------------
    // 4. Open via SDRModelLoader; verify HSDR strategy id stuck.
    // ----------------------------------------------------------------------
    SDRModelLoader loader(archive_path);
    const auto* header = loader.findSegmentHeader(seg.name);
    if (!header) {
        std::cerr << "FAIL: segment not found in archive\n";
        return 1;
    }
    std::cout << "  archive segment '" << header->name
              << "'  strategy_id=" << static_cast<int>(header->compression_strategy_id)
              << "  compressed_size=" << header->compressed_size << "\n";
    if (header->compression_strategy_id != HSDR_STRATEGY_ID) {
        std::cerr << "FAIL: expected HSDR strategy (5), got "
                  << static_cast<int>(header->compression_strategy_id) << "\n";
        return 1;
    }

    // ----------------------------------------------------------------------
    // 5. Build an activation x (C, batch) and compare matmulHSDR vs dense.
    // ----------------------------------------------------------------------
    const size_t batch = 4;
    std::vector<float> x(static_cast<size_t>(C) * batch);
    std::normal_distribution<float> xd(0.0f, 1.0f);
    for (auto& v : x) v = xd(rng);

    // Dense reference: load the segment (decompresses), then do W·x.
    auto t0 = std::chrono::steady_clock::now();
    ModelSegment recon = loader.loadSegmentByName(seg.name);
    const float* W_recon = reinterpret_cast<const float*>(recon.data.data());
    std::vector<float> Y_dense(static_cast<size_t>(R) * batch, 0.0f);
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t c = 0; c < C; ++c) {
            const float w = W_recon[r * C + c];
            for (size_t b = 0; b < batch; ++b) {
                Y_dense[r * batch + b] += w * x[c * batch + b];
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    auto t2 = std::chrono::steady_clock::now();
    auto Y_fused = loader.matmulHSDR(seg.name, x.data(), batch);
    auto t3 = std::chrono::steady_clock::now();

    double dense_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double fused_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "  dense (loadSegmentByName + matmul): " << dense_ms << " ms\n";
    std::cout << "  fused (loader.matmulHSDR):          " << fused_ms << " ms\n";

    if (Y_fused.size() != Y_dense.size()) {
        std::cerr << "FAIL: output size mismatch\n";
        return 1;
    }
    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (size_t i = 0; i < Y_dense.size(); ++i) {
        double d = std::abs(static_cast<double>(Y_dense[i]) - Y_fused[i]);
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    mean_abs /= Y_dense.size();
    std::cout << "  max |dense - fused| = " << max_abs
              << ",  mean |..| = " << mean_abs << "\n";

    bool ok = (max_abs < 1e-3);
    std::cout << (ok ? "  PASS\n" : "  FAIL: numeric drift > 1e-3\n");

    // Cleanup temporary archive.
    std::error_code ec;
    std::filesystem::remove(archive_path, ec);

    return ok ? 0 : 2;
}
