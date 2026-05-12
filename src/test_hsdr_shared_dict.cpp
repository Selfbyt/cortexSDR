/**
 * @file test_hsdr_shared_dict.cpp
 * @brief Validate the shared-dictionary path: pool tiles from several
 * synthetic FP32 weight tensors, fit ONE shared dictionary, encode each
 * tensor as codes-only bytes, verify round-trip + fused matmul, and report
 * total storage vs the per-tensor mode.
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

ModelSegment makeFP32Segment(const std::string& name, uint32_t R, uint32_t C, uint32_t seed) {
    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_FP32;
    seg.name = name;
    seg.original_size = static_cast<size_t>(R) * C * sizeof(float);
    seg.data.resize(seg.original_size);
    seg.data_format = "f32";
    TensorMetadata meta;
    meta.dimensions = {R, C};
    seg.tensor_metadata = meta;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    float* w = reinterpret_cast<float*>(seg.data.data());
    for (size_t i = 0; i < R * C; ++i) w[i] = dist(rng);
    return seg;
}

// Flatten a single FP32 weight tensor into 1D row tiles (each row = one tile).
void appendTilesFromSegment(const ModelSegment& seg, std::vector<float>& out) {
    const auto& dims = seg.tensor_metadata.value().dimensions;
    const size_t R = dims[0];
    const size_t C = dims[1];
    const float* w = reinterpret_cast<const float*>(seg.data.data());
    const size_t old = out.size();
    out.resize(old + R * C);
    std::memcpy(out.data() + old, w, R * C * sizeof(float));
}

}  // namespace

int main() {
    std::cout << "HSDR shared-dictionary test\n";

    // Three synthetic tensors with identical row width C — required so they
    // can all share one dictionary in 1D row-tile mode.
    const uint32_t C = 256;
    std::vector<ModelSegment> segs = {
        makeFP32Segment("layer0.attn", 96,  C, 0xA1),
        makeFP32Segment("layer1.attn", 128, C, 0xB2),
        makeFP32Segment("layer2.attn", 80,  C, 0xC3),
    };

    // ----------------------------------------------------------------------
    // 1. Pool tiles from all three segments and fit one shared dictionary.
    // ----------------------------------------------------------------------
    HierarchicalSDRConfig cfg = HierarchicalSDRConfig::forRow1D(
        /*row_width=*/static_cast<uint16_t>(C),
        /*n_atoms=*/64,
        /*k_per_stage=*/6);
    cfg.ksvd_iters = 6;

    std::vector<float> pooled;
    uint32_t pooled_count = 0;
    for (const auto& s : segs) {
        appendTilesFromSegment(s, pooled);
        pooled_count += static_cast<uint32_t>(s.tensor_metadata.value().dimensions[0]);
    }
    std::cout << "  pooled tiles from " << segs.size() << " tensors: "
              << pooled_count << " rows total\n";

    HierarchicalSDRStrategy strat(cfg, cfg);

    auto t_fit_0 = std::chrono::steady_clock::now();
    auto shared = strat.fitSharedDictionary(pooled, pooled_count, cfg);
    auto t_fit_1 = std::chrono::steady_clock::now();
    double fit_ms = std::chrono::duration<double, std::milli>(t_fit_1 - t_fit_0).count();
    std::cout << "  fit shared dict (" << shared.atoms.size() / cfg.tileSize()
              << " atoms × " << cfg.tileSize() << " dim) in " << fit_ms << " ms\n";

    // ----------------------------------------------------------------------
    // 2. Encode each segment against the shared dictionary.
    // ----------------------------------------------------------------------
    std::vector<std::vector<std::byte>> codes_streams;
    size_t total_codes_bytes = 0;
    for (const auto& s : segs) {
        auto codes = strat.compressWithExternalDictionary(s, shared);
        std::cout << "    encode '" << s.name << "' → " << codes.size() << " codes bytes\n";
        total_codes_bytes += codes.size();
        codes_streams.push_back(std::move(codes));
    }

    const size_t shared_dict_bytes = shared.atoms.size() * sizeof(float)
                                     + shared.stage_scales.size() * sizeof(float);
    const size_t shared_total_bytes = shared_dict_bytes + total_codes_bytes;
    std::cout << "  SHARED-DICT total: " << shared_dict_bytes << " (dict) + "
              << total_codes_bytes << " (codes) = " << shared_total_bytes << " bytes\n";

    // ----------------------------------------------------------------------
    // 3. Comparison baseline: same segments encoded per-tensor (each carries
    //    its own dictionary in the bytes).
    // ----------------------------------------------------------------------
    size_t per_tensor_total = 0;
    for (const auto& s : segs) {
        auto bytes = strat.compress(s);
        per_tensor_total += bytes.size();
    }
    std::cout << "  PER-TENSOR total: " << per_tensor_total << " bytes\n";
    std::cout << "  shared / per-tensor = "
              << static_cast<double>(shared_total_bytes) / per_tensor_total << "\n";

    // ----------------------------------------------------------------------
    // 4. Round-trip + fused matmul verification on the first segment.
    // ----------------------------------------------------------------------
    const ModelSegment& probe = segs.front();
    const auto& dims = probe.tensor_metadata.value().dimensions;
    const uint32_t R = dims[0];

    // 4a. Decompress via shared path, ensure FP32 layout is preserved.
    auto decompressed = strat.decompressWithExternalDictionary(
        codes_streams[0], shared, probe.data.size());
    if (decompressed.size() != probe.data.size()) {
        std::cerr << "FAIL: decompressed size mismatch\n";
        return 1;
    }

    // 4b. Reconstruction quality vs original.
    const float* w_orig = reinterpret_cast<const float*>(probe.data.data());
    const float* w_recon = reinterpret_cast<const float*>(decompressed.data());
    double mse = 0.0, var_sum = 0.0;
    double mean = 0.0;
    for (size_t i = 0; i < probe.data.size() / sizeof(float); ++i) mean += w_orig[i];
    mean /= (probe.data.size() / sizeof(float));
    for (size_t i = 0; i < probe.data.size() / sizeof(float); ++i) {
        double d = w_orig[i] - w_recon[i];
        mse += d * d;
        double dv = w_orig[i] - mean;
        var_sum += dv * dv;
    }
    size_t n_elems = probe.data.size() / sizeof(float);
    double nmse = (mse / n_elems) / std::max(var_sum / n_elems, 1e-12);
    std::cout << "  reconstruction NMSE (probe) = " << nmse << "\n";

    // 4c. Fused matmul vs dense matmul on the decompressed weight.
    const size_t batch = 4;
    std::vector<float> x(static_cast<size_t>(C) * batch);
    std::mt19937 rng(0xFEED);
    std::normal_distribution<float> xd(0.0f, 1.0f);
    for (auto& v : x) v = xd(rng);

    std::vector<float> Y_dense(static_cast<size_t>(R) * batch, 0.0f);
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t c = 0; c < C; ++c) {
            const float w = w_recon[r * C + c];
            for (size_t b = 0; b < batch; ++b) {
                Y_dense[r * batch + b] += w * x[c * batch + b];
            }
        }
    }

    auto Y_fused = strat.matmulWithExternalDictionary(
        codes_streams[0], shared, x.data(), batch);
    if (Y_fused.size() != Y_dense.size()) {
        std::cerr << "FAIL: matmul output size mismatch\n";
        return 1;
    }
    double max_abs = 0.0;
    for (size_t i = 0; i < Y_dense.size(); ++i) {
        double d = std::abs(static_cast<double>(Y_dense[i]) - Y_fused[i]);
        if (d > max_abs) max_abs = d;
    }
    std::cout << "  max |dense - fused matmul| = " << max_abs << "\n";

    if (nmse < 1.0 && max_abs < 1e-3 && shared_total_bytes < per_tensor_total) {
        std::cout << "  PASS\n";
        std::cout << "  → shared-dict saved "
                  << per_tensor_total - shared_total_bytes
                  << " bytes (" << (1.0 - double(shared_total_bytes)/per_tensor_total) * 100.0
                  << "%) on " << segs.size() << " tensors\n";
        return 0;
    }
    std::cout << "  FAIL\n";
    return 2;
}
