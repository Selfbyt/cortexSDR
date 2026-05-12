/**
 * @file test_hsda_reader.cpp
 * @brief Validate the name-indexed HSDAReader: build an archive, write it to
 * disk, open via HSDAReader, look up segments by name and verify
 * decompression + fused matmul.
 *
 * Closes the loop for the .hsda flow the upcoming inference-engine migration
 * will use.
 */
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/core/ModelSegment.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
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

}  // namespace

int main() {
    std::cout << "HSDAReader end-to-end test\n";

    // Build 5 attention + 3 mlp segments.
    std::vector<ModelSegment> segs;
    for (int i = 0; i < 5; ++i) {
        segs.push_back(makeFP32Segment(SegmentType::ATTENTION_WEIGHTS,
                                        "model.layers." + std::to_string(i) + ".attn.q_proj.weight",
                                        96, 256, 0xAA00 + i, /*layer_index=*/i));
    }
    for (int i = 0; i < 3; ++i) {
        segs.push_back(makeFP32Segment(SegmentType::FEED_FORWARD_WEIGHTS,
                                        "model.layers." + std::to_string(i) + ".mlp.down_proj.weight",
                                        96, 256, 0xBB00 + i, /*layer_index=*/i));
    }

    HierarchicalSDRConfig attn_cfg = HierarchicalSDRConfig::forRow1D(256, 64, 4);
    attn_cfg.ksvd_iters = 4;
    HierarchicalSDRConfig mlp_cfg = HierarchicalSDRConfig::forRow1D(256, 80, 6);
    mlp_cfg.ksvd_iters = 4;
    HierarchicalSDRStrategy strat(attn_cfg, mlp_cfg);

    // 1. Pipeline + disk write.
    std::vector<std::string> skipped;
    auto archive = strat.compressGroupedSegments(segs, &skipped);
    const std::string path = (std::filesystem::temp_directory_path()
                              / "hsda_reader_test.hsda").string();
    archive.writeToFile(path);
    std::cout << "  pipeline produced " << archive.dictionaries.size() << " dicts, "
              << archive.segments.size() << " segments; wrote " << path << "\n";

    // 2. Open via HSDAReader.
    HSDAReader reader = HSDAReader::fromFile(path);
    std::cout << "  HSDAReader loaded: " << reader.numDictionaries() << " dicts, "
              << reader.numSegments() << " segments\n";
    if (reader.numSegments() != archive.segments.size()
        || reader.numDictionaries() != archive.dictionaries.size()) {
        std::cerr << "FAIL: counts mismatch after HSDAReader load\n";
        return 1;
    }

    // 3. Look up segments by name and verify.
    const size_t batch = 2;
    std::vector<float> x(256 * batch);
    std::mt19937 xrng(0xDEAD);
    std::normal_distribution<float> xd(0.0f, 1.0f);
    for (auto& v : x) v = xd(xrng);

    int rt_fail = 0;
    double worst_matmul_diff = 0.0;
    for (const auto& orig : segs) {
        if (!reader.hasSegment(orig.name)) {
            std::cerr << "FAIL: reader missing segment '" << orig.name << "'\n";
            return 1;
        }
        // 3a. decompress + matmul via reader.
        auto rec_bytes = reader.decompress(orig.name);
        auto Y_fused = reader.matmul(orig.name, x.data(), batch);

        // 3b. Dense reference from the decompressed weight.
        if (rec_bytes.size() != orig.data.size()) {
            std::cerr << "FAIL: size mismatch on '" << orig.name << "'\n";
            return 1;
        }
        const float* rec_w = reinterpret_cast<const float*>(rec_bytes.data());
        const auto& dims = orig.tensor_metadata.value().dimensions;
        const uint32_t R = static_cast<uint32_t>(dims[0]);
        const uint32_t C = static_cast<uint32_t>(dims[1]);
        std::vector<float> Y_dense(static_cast<size_t>(R) * batch, 0.0f);
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t c = 0; c < C; ++c) {
                const float w = rec_w[r * C + c];
                for (size_t b = 0; b < batch; ++b) {
                    Y_dense[r * batch + b] += w * x[c * batch + b];
                }
            }
        }
        for (size_t k = 0; k < Y_dense.size(); ++k) {
            double d = std::abs(static_cast<double>(Y_dense[k]) - Y_fused[k]);
            if (d > worst_matmul_diff) worst_matmul_diff = d;
        }
    }
    std::cout << "  verified " << segs.size() << " segments via reader; "
              << "worst dense-vs-fused diff = " << worst_matmul_diff << "\n";

    // 4. Missing-name path returns a useful error.
    try {
        reader.decompress("does.not.exist");
        std::cerr << "FAIL: missing-name decompress should have thrown\n";
        return 1;
    } catch (const std::exception& e) {
        std::cout << "  missing-name throws as expected: " << e.what() << "\n";
    }

    // Cleanup.
    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (rt_fail == 0 && worst_matmul_diff < 1e-3) {
        std::cout << "PASS\n";
        return 0;
    }
    std::cout << "FAIL\n";
    return 1;
}
