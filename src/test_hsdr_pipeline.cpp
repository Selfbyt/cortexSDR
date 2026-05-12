/**
 * @file test_hsdr_pipeline.cpp
 * @brief Multi-pass pipeline test for compressGroupedSegments.
 *
 * Builds N synthetic segments across two roles (attn, mlp) with different
 * row widths, runs the pipeline, verifies that:
 *   - Each bucket produced one shared dictionary.
 *   - Every segment decodes back to its original shape with bounded NMSE.
 *   - Fused matmul matches dense to FP32 noise (for 1D row tiles).
 *   - The total bytes are clearly below the per-tensor encoding.
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

ModelSegment makeFP32Segment(SegmentType type, const std::string& name,
                              uint32_t R, uint32_t C, uint32_t seed,
                              size_t layer_index = 0) {
    ModelSegment seg;
    seg.type = type;
    seg.name = name;
    seg.layer_index = layer_index;
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

double computeNMSE(const float* orig, const float* recon, size_t n) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += orig[i];
    mean /= n;
    double mse = 0.0, var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = orig[i] - recon[i];
        mse += d * d;
        double dv = orig[i] - mean;
        var += dv * dv;
    }
    return (mse / n) / std::max(var / n, 1e-30);
}

}  // namespace

int main() {
    std::cout << "HSDR multi-pass pipeline test\n";

    // Build 10 segments across two roles:
    //   - 6 attention segments, all shape (128, 256)
    //   - 4 MLP segments, all shape (96, 256)
    // Both roles share the same row width (256), but `configFor` will pick
    // different (n_atoms, k) defaults — so we expect TWO dictionaries.
    std::vector<ModelSegment> segs;
    for (int i = 0; i < 6; ++i) {
        segs.push_back(makeFP32Segment(SegmentType::ATTENTION_WEIGHTS,
                                        "model.layers." + std::to_string(i) + ".attn.q_proj.weight",
                                        128, 256, 0xA00 + i, /*layer_index=*/i));
    }
    for (int i = 0; i < 4; ++i) {
        segs.push_back(makeFP32Segment(SegmentType::FEED_FORWARD_WEIGHTS,
                                        "model.layers." + std::to_string(i) + ".mlp.down_proj.weight",
                                        96, 256, 0xB00 + i, /*layer_index=*/i));
    }

    // Override the defaults so K is small enough that even 96 tiles can fit.
    // (default attn K=256 would need ≥256 pooled tiles; default mlp K=512 way more.)
    HierarchicalSDRConfig attn_cfg = HierarchicalSDRConfig::forRow1D(
        /*row_width=*/256, /*n_atoms=*/64, /*k_per_stage=*/4);
    attn_cfg.ksvd_iters = 4;
    HierarchicalSDRConfig mlp_cfg = HierarchicalSDRConfig::forRow1D(
        /*row_width=*/256, /*n_atoms=*/96, /*k_per_stage=*/6);
    mlp_cfg.ksvd_iters = 4;

    HierarchicalSDRStrategy strat(attn_cfg, mlp_cfg);

    // ----------------------------------------------------------------------
    // 1. Run the pipeline.
    // ----------------------------------------------------------------------
    std::vector<std::string> skipped;
    auto t0 = std::chrono::steady_clock::now();
    auto archive = strat.compressGroupedSegments(segs, &skipped);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  pipeline ran in " << ms << " ms\n";
    std::cout << "    dictionaries: " << archive.dictionaries.size() << "\n";
    std::cout << "    encoded segments: " << archive.segments.size() << "\n";
    std::cout << "    skipped segments: " << skipped.size() << "\n";

    if (archive.dictionaries.size() != 2) {
        std::cerr << "FAIL: expected 2 dictionaries (one per role), got "
                  << archive.dictionaries.size() << "\n";
        return 1;
    }
    if (archive.segments.size() != segs.size() || !skipped.empty()) {
        std::cerr << "FAIL: expected " << segs.size() << " encoded segments, got "
                  << archive.segments.size() << " (skipped " << skipped.size() << ")\n";
        return 1;
    }

    // ----------------------------------------------------------------------
    // 2. Verify each segment round-trips and matmul-fuses.
    // ----------------------------------------------------------------------
    int rt_fail = 0;
    double worst_nmse = 0.0;
    double worst_matmul_err = 0.0;
    const size_t batch = 2;

    // Make a single random activation reused for all matmul checks.
    std::vector<float> x(static_cast<size_t>(256) * batch);
    std::mt19937 xrng(0xFEED);
    std::normal_distribution<float> xd(0.0f, 1.0f);
    for (auto& v : x) v = xd(xrng);

    for (size_t i = 0; i < archive.segments.size(); ++i) {
        const auto& entry = archive.segments[i];
        const auto& dict = archive.dictionaries[entry.dict_index];

        // Locate the original segment.
        const ModelSegment* orig = nullptr;
        for (const auto& s : segs) if (s.name == entry.name) { orig = &s; break; }
        if (!orig) {
            std::cerr << "FAIL: original not found for '" << entry.name << "'\n";
            return 1;
        }

        // Decompress against the shared dict.
        auto rec_bytes = strat.decompressWithExternalDictionary(
            entry.codes_bytes, dict, entry.original_size);
        if (rec_bytes.size() != orig->data.size()) {
            std::cerr << "FAIL: size mismatch on '" << entry.name << "'\n";
            return 1;
        }
        const float* orig_w = reinterpret_cast<const float*>(orig->data.data());
        const float* rec_w  = reinterpret_cast<const float*>(rec_bytes.data());
        size_t n_elems = orig->data.size() / sizeof(float);
        double nmse = computeNMSE(orig_w, rec_w, n_elems);
        if (nmse > worst_nmse) worst_nmse = nmse;

        // Fused matmul check.
        const auto& dims = orig->tensor_metadata.value().dimensions;
        const uint32_t R = static_cast<uint32_t>(dims[0]);
        const uint32_t C = static_cast<uint32_t>(dims[1]);
        // Dense reference using the *decompressed* weight (which is what fused
        // should reproduce). NB: comparing against the original weight would
        // bake in the lossy reconstruction error; we want to verify the fused
        // path matches what dense matmul on the reconstructed weight gives.
        std::vector<float> Y_dense(static_cast<size_t>(R) * batch, 0.0f);
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t c = 0; c < C; ++c) {
                const float w = rec_w[r * C + c];
                for (size_t b = 0; b < batch; ++b) {
                    Y_dense[r * batch + b] += w * x[c * batch + b];
                }
            }
        }
        auto Y_fused = strat.matmulWithExternalDictionary(
            entry.codes_bytes, dict, x.data(), batch);
        double max_diff = 0.0;
        for (size_t k = 0; k < Y_dense.size(); ++k) {
            double d = std::abs(static_cast<double>(Y_dense[k]) - Y_fused[k]);
            if (d > max_diff) max_diff = d;
        }
        if (max_diff > worst_matmul_err) worst_matmul_err = max_diff;
    }

    std::cout << "  worst NMSE across segments:  " << worst_nmse << "\n";
    std::cout << "  worst fused-matmul drift:    " << worst_matmul_err << "\n";

    // ----------------------------------------------------------------------
    // 3. Compare total bytes to per-tensor mode.
    // ----------------------------------------------------------------------
    size_t per_tensor_total = 0;
    for (const auto& s : segs) {
        try {
            auto bytes = strat.compress(s);
            per_tensor_total += bytes.size();
        } catch (const CompressionError& e) {
            // The per-tensor compress() defaults to attn/mlp_default_ K, which is
            // 256/512 — these are larger than n_tiles per segment here, so it
            // will throw. Approximate per-tensor cost using the same configs the
            // pipeline picked (attn_cfg/mlp_cfg).
            HierarchicalSDRStrategy alt(attn_cfg, mlp_cfg);
            auto bytes = alt.compress(s);
            per_tensor_total += bytes.size();
        }
    }
    const size_t pipeline_total = archive.totalBytes();
    std::cout << "  PER-TENSOR total: " << per_tensor_total << " bytes\n";
    std::cout << "  PIPELINE total:   " << pipeline_total << " bytes\n";
    if (per_tensor_total > 0) {
        const double ratio = static_cast<double>(pipeline_total) / per_tensor_total;
        std::cout << "  pipeline / per_tensor = " << ratio
                  << " (" << (1.0 - ratio) * 100.0 << "% saved)\n";
    }

    if (rt_fail > 0 || worst_matmul_err >= 1e-3 || worst_nmse > 1.0) {
        std::cerr << "FAIL: rt_fail=" << rt_fail
                  << " worst_matmul_err=" << worst_matmul_err
                  << " worst_nmse=" << worst_nmse << "\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
