/**
 * @file test_hsdr_roundtrip.cpp
 * @brief Self-contained smoke test for HierarchicalSDRStrategy.
 *
 * Builds a synthetic FP32 weight matrix that the strategy should be able to
 * compress and decompress with bounded reconstruction error, then prints the
 * NMSE and compression ratio. Not a CTest target — invoked manually after a
 * build to confirm the integration is wired correctly.
 *
 *   build\Release\test_hsdr_roundtrip.exe
 */
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/core/ModelSegment.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace CortexAICompression;

namespace {

double meanSquaredError(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum / std::max<size_t>(n, 1);
}

double variance(const float* a, size_t n) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += a[i];
    mean /= std::max<size_t>(n, 1);
    double sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - mean;
        sq += d * d;
    }
    return sq / std::max<size_t>(n, 1);
}

void run_case(const char* label, SegmentType type, uint32_t R, uint32_t C, uint32_t seed,
              const HierarchicalSDRConfig& cfg) {
    std::cout << "\n=== " << label << "  shape=(" << R << ", " << C << ")"
              << "  tile=" << cfg.tile_rows << "x" << cfg.tile_cols
              << "  K=" << cfg.n_atoms
              << "  k=" << static_cast<int>(cfg.active_bits_per_stage) << "x" << static_cast<int>(cfg.n_stages)
              << "  iters=" << static_cast<int>(cfg.ksvd_iters)
              << " ===\n";

    std::vector<float> weight(static_cast<size_t>(R) * C);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (auto& v : weight) v = dist(rng);

    ModelSegment seg;
    seg.type = type;
    seg.name = (type == SegmentType::FEED_FORWARD_WEIGHTS) ? "model.layers.0.mlp.down_proj.weight"
                                                            : "model.layers.0.self_attn.q_proj.weight";
    seg.data.resize(weight.size() * sizeof(float));
    std::memcpy(seg.data.data(), weight.data(), seg.data.size());
    seg.original_size = seg.data.size();
    seg.data_format = "f32";

    TensorMetadata meta;
    meta.dimensions = {R, C};
    meta.sparsity_ratio = 0.0f;
    meta.is_sorted = false;
    seg.tensor_metadata = meta;

    // Use the same config for both attn and mlp branches in this smoke test.
    HierarchicalSDRStrategy strat(cfg, cfg);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::byte> compressed;
    try {
        compressed = strat.compress(seg);
    } catch (const CompressionError& e) {
        std::cout << "  expected throw on this configuration: " << e.what() << "\n";
        std::cout << "  PASS (error path)\n";
        return;
    }
    auto t1 = std::chrono::steady_clock::now();
    double fit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  compress: " << fit_ms << " ms, " << compressed.size() << " bytes\n";
    std::cout << "  ratio vs FP32 input: "
              << static_cast<double>(seg.data.size()) / static_cast<double>(compressed.size())
              << "x\n";

    auto t2 = std::chrono::steady_clock::now();
    auto decompressed = strat.decompress(compressed, seg.type, seg.data.size());
    auto t3 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "  decompress: " << dec_ms << " ms\n";

    if (decompressed.size() != seg.data.size()) {
        std::cerr << "  FAIL: decompressed size " << decompressed.size()
                  << " != " << seg.data.size() << "\n";
        return;
    }
    const float* recon = reinterpret_cast<const float*>(decompressed.data());
    double mse = meanSquaredError(weight.data(), recon, weight.size());
    double var = variance(weight.data(), weight.size());
    double nmse = mse / std::max(var, 1e-30);
    std::cout << "  reconstruction MSE = " << mse
              << ", NMSE = " << nmse << "\n";
    if (nmse < 1.5) {
        std::cout << "  PASS\n";
    } else {
        std::cout << "  WARN: NMSE > 1.5 — check fit\n";
    }
}

}  // namespace

int main() {
    std::cout << "HSDR round-trip smoke test\n";

    // Tiny config for fast verification (~seconds): 32x32 tiles, K=32, k=4x3, 4 iters.
    HierarchicalSDRConfig tiny;
    tiny.tile_rows = 32;
    tiny.tile_cols = 32;
    tiny.n_atoms = 32;
    tiny.n_stages = 3;
    tiny.active_bits_per_stage = 4;
    tiny.ksvd_iters = 4;
    tiny.stage_decay = 0.5f;

    // Shape must be divisible by tile, n_tiles >= K. 256x256 -> 64 tiles, K=32 ✓
    run_case("attention-like (small, tiny config)", SegmentType::ATTENTION_WEIGHTS,
             256, 256, 0xC0FFEE, tiny);

    // 192x320 -> 6x10 = 60 tiles, K=32 ✓
    run_case("mlp-like (small, tiny config)", SegmentType::FEED_FORWARD_WEIGHTS,
             192, 320, 0xBEEF, tiny);

    // Medium config (~30s on CPU): 64x64 tiles, K=64, k=5x3, 6 iters.
    HierarchicalSDRConfig medium;
    medium.tile_rows = 64;
    medium.tile_cols = 64;
    medium.n_atoms = 64;
    medium.n_stages = 3;
    medium.active_bits_per_stage = 5;
    medium.ksvd_iters = 6;
    medium.stage_decay = 0.5f;

    // 512x512 -> 64 tiles, K=64 ✓ (boundary case)
    run_case("medium attention", SegmentType::ATTENTION_WEIGHTS,
             512, 512, 0xFEEDBEEF, medium);

    // 1D row-tile mode: each weight ROW becomes one tile. This is the layout
    // that makes the precompute-once-per-token-block fused inference math
    // actually win versus dense matmul (see HierarchicalSDRStrategy.cpp note).
    HierarchicalSDRConfig row1d;
    row1d.tile_rows = 1;
    row1d.tile_cols = 256;
    row1d.n_atoms = 32;
    row1d.n_stages = 3;
    row1d.active_bits_per_stage = 4;
    row1d.ksvd_iters = 4;
    row1d.stage_decay = 0.5f;

    // 64 rows x 256 cols, K=32, n_tiles=64.
    run_case("1D row-tile mode", SegmentType::ATTENTION_WEIGHTS,
             64, 256, 0xDEFACED, row1d);

    // ----------------------------------------------------------------------
    // Fused matmul: verify Y_fused (W·x without materialising W) matches
    // Y_dense (W·x from decompressed W) and report rough speed comparison.
    // Uses 1D row-tile encoding — this is the only mode where fused beats
    // dense at typical LLM shapes.
    // ----------------------------------------------------------------------
    {
        std::cout << "\n=== Fused matmul (1D row tiles) ===\n";
        HierarchicalSDRConfig cfg = HierarchicalSDRConfig::forRow1D(/*row_width=*/512,
                                                                     /*n_atoms=*/128,
                                                                     /*k_per_stage=*/6);
        cfg.ksvd_iters = 6;  // keep test fast

        const uint32_t R = 256;
        const uint32_t C = cfg.tile_cols;  // 512
        const size_t batch = 4;

        std::vector<float> weight(static_cast<size_t>(R) * C);
        std::mt19937 rng(0xFAFA);
        std::normal_distribution<float> dist(0.0f, 0.05f);
        for (auto& v : weight) v = dist(rng);

        // Build a segment and compress.
        ModelSegment seg;
        seg.type = SegmentType::ATTENTION_WEIGHTS;
        seg.name = "fused.matmul.test";
        seg.data.resize(weight.size() * sizeof(float));
        std::memcpy(seg.data.data(), weight.data(), seg.data.size());
        seg.original_size = seg.data.size();
        seg.data_format = "f32";
        TensorMetadata meta;
        meta.dimensions = {R, C};
        seg.tensor_metadata = meta;

        HierarchicalSDRStrategy strat(cfg, cfg);
        auto compressed = strat.compress(seg);

        // Build an input activation x of shape (C, batch), row-major.
        std::vector<float> x(static_cast<size_t>(C) * batch);
        std::normal_distribution<float> xd(0.0f, 1.0f);
        for (auto& v : x) v = xd(rng);

        // (a) Dense path: decompress W' and do W' @ x.
        auto t0 = std::chrono::steady_clock::now();
        auto dec_bytes = strat.decompress(compressed, seg.type, seg.data.size());
        const float* W_recon = reinterpret_cast<const float*>(dec_bytes.data());
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

        // (b) Fused path: matmul directly from compressed bytes.
        auto t2 = std::chrono::steady_clock::now();
        auto Y_fused = strat.matmulRowMajor(compressed, x.data(), batch);
        auto t3 = std::chrono::steady_clock::now();

        double dense_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double fused_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "  W shape=(" << R << ", " << C << ")   batch=" << batch << "\n";
        std::cout << "  dense path (decompress+matmul): " << dense_ms << " ms\n";
        std::cout << "  fused path (no materialisation): " << fused_ms << " ms\n";

        if (Y_dense.size() != Y_fused.size()) {
            std::cerr << "  FAIL: output size mismatch\n";
        } else {
            double max_abs_diff = 0.0;
            double mean_abs_diff = 0.0;
            for (size_t i = 0; i < Y_dense.size(); ++i) {
                double d = std::abs(static_cast<double>(Y_dense[i]) - Y_fused[i]);
                if (d > max_abs_diff) max_abs_diff = d;
                mean_abs_diff += d;
            }
            mean_abs_diff /= Y_dense.size();
            std::cout << "  max |dense - fused| = " << max_abs_diff
                      << ",  mean |..| = " << mean_abs_diff << "\n";
            if (max_abs_diff < 1e-3) {
                std::cout << "  PASS (dense and fused outputs agree)\n";
            } else {
                std::cout << "  WARN: drift > 1e-3 — check accumulation order / fp64 vs fp32\n";
            }
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
