/**
 * @file test_hsdr_protection.cpp
 * @brief Validate the hybrid FP16 protection predicates: boundary-MLP and
 * by-name. A protected segment should throw CompressionError so the
 * AICompressor strategy chain falls through to the lossless fallback.
 */
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/core/ModelSegment.hpp"

#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace CortexAICompression;

namespace {

ModelSegment makeMLPSegment(const std::string& name, size_t layer_index,
                             uint32_t R, uint32_t C, uint32_t seed) {
    ModelSegment seg;
    seg.type = SegmentType::FEED_FORWARD_WEIGHTS;
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

bool tryCompress(const HierarchicalSDRStrategy& strat, const ModelSegment& seg) {
    try {
        (void)strat.compress(seg);
        return true;  // succeeded
    } catch (const CompressionError& e) {
        std::cout << "    THREW: " << e.what() << "\n";
        return false;
    }
}

}  // namespace

int main() {
    std::cout << "HSDR protection-policy test\n";

    // Pretend we have a 22-layer decoder model. The C.2 finding: protect MLPs
    // at the first/last few decoder layers. Pick n_boundary=3.
    const size_t total_layers = 22;
    const size_t n_boundary = 3;

    // Use a small config so fits are quick.
    HierarchicalSDRConfig cfg = HierarchicalSDRConfig::forRow1D(
        /*row_width=*/256, /*n_atoms=*/64, /*k_per_stage=*/6);
    cfg.ksvd_iters = 4;

    HierarchicalSDRStrategy strat(cfg, cfg);
    strat.setProtectionPredicate(ProtectionPolicies::boundaryMLPs(n_boundary, total_layers));

    // 8 segments: MLPs at depths {0, 2, 5, 10, 18, 19, 21} + one named "lm_head"
    // (which boundaryMLPs won't catch unless we also use byName).
    std::vector<ModelSegment> segs;
    segs.push_back(makeMLPSegment("model.layers.0.mlp.down_proj.weight",  0, 128, 256, 0xA0));
    segs.push_back(makeMLPSegment("model.layers.2.mlp.down_proj.weight",  2, 128, 256, 0xA2));
    segs.push_back(makeMLPSegment("model.layers.5.mlp.down_proj.weight",  5, 128, 256, 0xA5));
    segs.push_back(makeMLPSegment("model.layers.10.mlp.down_proj.weight", 10, 128, 256, 0xB0));
    segs.push_back(makeMLPSegment("model.layers.18.mlp.down_proj.weight", 18, 128, 256, 0xB8));
    segs.push_back(makeMLPSegment("model.layers.19.mlp.down_proj.weight", 19, 128, 256, 0xB9));
    segs.push_back(makeMLPSegment("model.layers.21.mlp.down_proj.weight", 21, 128, 256, 0xBA));
    // Pretend lm_head is named outside the layer-index pattern; we'll combine
    // policies via byName below.
    {
        auto s = makeMLPSegment("lm_head.weight", 0, 128, 256, 0xCC);
        s.type = SegmentType::WEIGHTS_FP32;  // not MLP, so boundaryMLPs ignores it
        segs.push_back(s);
    }

    std::cout << "  Test 1: boundaryMLPs(n_boundary=" << n_boundary
              << ", total_layers=" << total_layers << ")\n";
    std::cout << "    Expected: protect MLPs at depths 0-2 and 19-21; "
              << "skip lm_head and middle MLPs.\n";

    struct ExpectedResult { std::string name; bool should_succeed; };
    const std::vector<ExpectedResult> expectations = {
        {"model.layers.0.mlp.down_proj.weight",  false},  // protected (depth 0 < 3)
        {"model.layers.2.mlp.down_proj.weight",  false},  // protected (depth 2 < 3)
        {"model.layers.5.mlp.down_proj.weight",  true},   // middle MLP — V4b
        {"model.layers.10.mlp.down_proj.weight", true},   // middle MLP — V4b
        {"model.layers.18.mlp.down_proj.weight", true},   // depth 18, n=3 → late_threshold=19 → not protected
        {"model.layers.19.mlp.down_proj.weight", false},  // protected (depth 19 >= 22-3 = 19)
        {"model.layers.21.mlp.down_proj.weight", false},  // protected (depth 21 >= 19)
        {"lm_head.weight",                        true},   // not an MLP — V4b would handle it
    };

    int passes = 0, fails = 0;
    for (size_t i = 0; i < segs.size(); ++i) {
        std::cout << "  '" << segs[i].name << "':\n";
        bool succeeded = tryCompress(strat, segs[i]);
        const bool expected = expectations[i].should_succeed;
        if (succeeded == expected) {
            std::cout << "    OK (" << (expected ? "compressed" : "rejected as protected") << ")\n";
            ++passes;
        } else {
            std::cout << "    FAIL: expected " << (expected ? "success" : "rejection")
                      << ", got " << (succeeded ? "success" : "rejection") << "\n";
            ++fails;
        }
    }

    std::cout << "\n  Test 2: byName({embed, lm_head}) combined with boundary policy.\n";
    // Stacking policies: byName for static lists, boundaryMLPs for the rule.
    auto bn = ProtectionPolicies::byName({"lm_head.weight"});
    auto bm = ProtectionPolicies::boundaryMLPs(n_boundary, total_layers);
    strat.setProtectionPredicate([bn, bm](const ModelSegment& seg) {
        return bn(seg) || bm(seg);
    });

    // Now lm_head should ALSO be rejected.
    const auto& head = segs.back();
    std::cout << "  '" << head.name << "' under combined policy:\n";
    bool head_succeeded = tryCompress(strat, head);
    if (!head_succeeded) {
        std::cout << "    OK (rejected as protected via byName)\n";
        ++passes;
    } else {
        std::cout << "    FAIL: should have been rejected via byName\n";
        ++fails;
    }

    std::cout << "\nResult: " << passes << " pass, " << fails << " fail\n";
    if (fails == 0) {
        std::cout << "PASS\n";
        return 0;
    }
    std::cout << "FAIL\n";
    return 1;
}
