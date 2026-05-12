/**
 * @file bench_hsdr_fit_speed.cpp
 * @brief Measure fit wall-time + storage ratio + NMSE for a few representative
 * HSDR configs on a synthetic LLM-MLP-shaped tensor. Helps choose between the
 * default config (slow + tight quality) and the new fast preset (fast + looser
 * quality) without burning hours on a 7B model.
 */
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/core/ModelSegment.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace CortexAICompression;

namespace {

ModelSegment makeFP32Segment(uint32_t R, uint32_t C, uint32_t seed) {
    ModelSegment s;
    s.type = SegmentType::WEIGHTS_FP32;
    s.name = "bench.weight";
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

double nmse(const float* orig, const float* recon, size_t n) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += orig[i];
    mean /= n;
    double mse = 0.0, var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = orig[i] - recon[i]; mse += d * d;
        double dv = orig[i] - mean; var += dv * dv;
    }
    return (mse / n) / std::max(var / n, 1e-30);
}

struct Run {
    std::string label;
    HierarchicalSDRConfig cfg;
};

void runOne(const Run& r, const ModelSegment& seg) {
    const auto& dims = seg.tensor_metadata.value().dimensions;
    const size_t R = dims[0];
    const size_t C = dims[1];
    HierarchicalSDRStrategy strat(r.cfg, r.cfg);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::byte> bytes;
    try {
        bytes = strat.compress(seg);
    } catch (const std::exception& e) {
        std::cout << "  " << r.label << ": THREW " << e.what() << "\n";
        return;
    }
    auto t1 = std::chrono::steady_clock::now();
    auto out = strat.decompress(bytes, seg.type, seg.data.size());
    auto t2 = std::chrono::steady_clock::now();

    const float* orig = reinterpret_cast<const float*>(seg.data.data());
    const float* rec  = reinterpret_cast<const float*>(out.data());
    const double q = nmse(orig, rec, R * C);
    const double compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decompress_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double ratio = static_cast<double>(seg.data.size()) / bytes.size();

    std::cout << "  " << r.label << "\n"
              << "    compress:   " << compress_ms << " ms\n"
              << "    decompress: " << decompress_ms << " ms\n"
              << "    bytes:      " << bytes.size() << "  (ratio " << ratio << "x vs FP32)\n"
              << "    NMSE:       " << q << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    // Default: small (256, 512) so smoke runs are fast.
    // Pass `--big` to use a (1024, 4096) tensor (closer to attention-layer scale).
    uint32_t R = 256, C = 512;
    if (argc >= 2 && std::string(argv[1]) == "--big") {
        R = 1024; C = 4096;
    }

    std::cout << "HSDR fit-speed benchmark   shape=(" << R << ", " << C << ")  FP32\n";

    auto seg = makeFP32Segment(R, C, 0xBEEF);

    // 1) Default 1-D row-tile (high quality, slow)
    HierarchicalSDRConfig default_cfg = HierarchicalSDRConfig::forRow1D(
        /*row_width=*/static_cast<uint16_t>(C), /*n_atoms=*/256, /*k_per_stage=*/6);
    default_cfg.ksvd_iters = 8;

    // 2) Fast preset (small K, few iters, no tile cap needed at this size)
    HierarchicalSDRConfig fast_cfg = HierarchicalSDRConfig::forFast(
        /*row_width=*/static_cast<uint16_t>(C));

    // 3) Same as fast but with explicit tile cap (no-op at this scale, but
    //    proves the wiring works).
    HierarchicalSDRConfig capped_cfg = fast_cfg;
    capped_cfg.max_tiles_for_fit = 64;  // R is small; this triggers subsampling.

    std::vector<Run> runs = {
        {"default 1D row-tile (K=256, k=6x3, 8 iters)", default_cfg},
        {"--fast preset (K=128, k=4x3, 4 iters)",        fast_cfg},
        {"--fast + max_tiles_for_fit=64",                 capped_cfg},
    };

    for (const auto& r : runs) runOne(r, seg);
    return 0;
}
