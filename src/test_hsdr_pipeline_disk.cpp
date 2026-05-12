/**
 * @file test_hsdr_pipeline_disk.cpp
 * @brief Round-trip a SharedDictArchive through disk: pipeline -> writeToFile
 * -> readFromFile -> identity check on dictionaries/segments + decompress
 * + fused matmul.
 */
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/core/ModelSegment.hpp"

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
    std::normal_distribution<float> d(0.0f, 0.05f);
    float* w = reinterpret_cast<float*>(seg.data.data());
    for (size_t i = 0; i < R * C; ++i) w[i] = d(rng);
    return seg;
}

}  // namespace

int main() {
    std::cout << "HSDR pipeline disk-roundtrip test\n";

    // Build 4 attention + 3 mlp segments (same as the pipeline test, just smaller).
    std::vector<ModelSegment> segs;
    for (int i = 0; i < 4; ++i) {
        segs.push_back(makeFP32Segment(SegmentType::ATTENTION_WEIGHTS,
                                        "model.layers." + std::to_string(i) + ".attn.q_proj.weight",
                                        96, 256, 0xA00 + i, /*layer_index=*/i));
    }
    for (int i = 0; i < 3; ++i) {
        segs.push_back(makeFP32Segment(SegmentType::FEED_FORWARD_WEIGHTS,
                                        "model.layers." + std::to_string(i) + ".mlp.down_proj.weight",
                                        96, 256, 0xB00 + i, /*layer_index=*/i));
    }

    HierarchicalSDRConfig attn_cfg = HierarchicalSDRConfig::forRow1D(256, 64, 4);
    attn_cfg.ksvd_iters = 4;
    HierarchicalSDRConfig mlp_cfg = HierarchicalSDRConfig::forRow1D(256, 80, 6);
    mlp_cfg.ksvd_iters = 4;
    HierarchicalSDRStrategy strat(attn_cfg, mlp_cfg);

    // ----------------------------------------------------------------------
    // 1. Build an in-memory archive via the pipeline.
    // ----------------------------------------------------------------------
    std::vector<std::string> skipped;
    auto archive_in = strat.compressGroupedSegments(segs, &skipped);
    std::cout << "  pipeline produced " << archive_in.dictionaries.size() << " dictionaries, "
              << archive_in.segments.size() << " segments\n";
    if (archive_in.dictionaries.empty() || archive_in.segments.empty()) {
        std::cerr << "FAIL: empty archive\n";
        return 1;
    }

    // ----------------------------------------------------------------------
    // 2. Round-trip through disk.
    // ----------------------------------------------------------------------
    const std::string path = (std::filesystem::temp_directory_path()
                              / "hsdr_pipeline_roundtrip.hsda").string();
    archive_in.writeToFile(path);
    const auto bytes_on_disk = std::filesystem::file_size(path);
    std::cout << "  wrote " << path << "  (" << bytes_on_disk << " bytes)\n";

    auto archive_out = HierarchicalSDRStrategy::SharedDictArchive::readFromFile(path);
    std::cout << "  read back: " << archive_out.dictionaries.size() << " dictionaries, "
              << archive_out.segments.size() << " segments\n";

    // ----------------------------------------------------------------------
    // 3. Structural equality.
    // ----------------------------------------------------------------------
    if (archive_out.dictionaries.size() != archive_in.dictionaries.size()
        || archive_out.segments.size() != archive_in.segments.size()) {
        std::cerr << "FAIL: archive size mismatch after roundtrip\n";
        return 1;
    }
    for (size_t i = 0; i < archive_in.dictionaries.size(); ++i) {
        const auto& a = archive_in.dictionaries[i];
        const auto& b = archive_out.dictionaries[i];
        if (std::memcmp(&a.config, &b.config, sizeof(a.config)) != 0
            || a.stage_scales != b.stage_scales
            || a.atoms != b.atoms) {
            std::cerr << "FAIL: dictionary " << i << " differs after roundtrip\n";
            return 1;
        }
    }
    for (size_t i = 0; i < archive_in.segments.size(); ++i) {
        const auto& a = archive_in.segments[i];
        const auto& b = archive_out.segments[i];
        if (a.name != b.name || a.dict_index != b.dict_index
            || a.original_size != b.original_size
            || a.original_type != b.original_type
            || a.codes_bytes != b.codes_bytes) {
            std::cerr << "FAIL: segment " << i << " ('" << a.name
                      << "') differs after roundtrip\n";
            return 1;
        }
    }
    std::cout << "  structural equality: OK\n";

    // ----------------------------------------------------------------------
    // 4. Decompress + fused matmul through the read-back archive.
    // ----------------------------------------------------------------------
    const size_t batch = 2;
    std::vector<float> x(256 * batch);
    std::mt19937 xrng(0xFEED);
    std::normal_distribution<float> xd(0.0f, 1.0f);
    for (auto& v : x) v = xd(xrng);

    double worst_diff = 0.0;
    for (const auto& entry : archive_out.segments) {
        const auto& dict = archive_out.dictionaries[entry.dict_index];
        // Decompress using the read-back dictionary.
        auto rec = strat.decompressWithExternalDictionary(
            entry.codes_bytes, dict, entry.original_size);
        const float* rec_w = reinterpret_cast<const float*>(rec.data());

        // Find original to determine R.
        const ModelSegment* orig = nullptr;
        for (const auto& s : segs) if (s.name == entry.name) { orig = &s; break; }
        if (!orig) { std::cerr << "FAIL: original missing\n"; return 1; }
        const auto& dims = orig->tensor_metadata.value().dimensions;
        const uint32_t R = static_cast<uint32_t>(dims[0]);
        const uint32_t C = static_cast<uint32_t>(dims[1]);

        // Dense reference uses the reconstructed weights (since fused
        // operates on the same compressed form).
        std::vector<float> Y_dense(R * batch, 0.0f);
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
        for (size_t k = 0; k < Y_dense.size(); ++k) {
            double d = std::abs(static_cast<double>(Y_dense[k]) - Y_fused[k]);
            if (d > worst_diff) worst_diff = d;
        }
    }
    std::cout << "  worst dense-vs-fused diff (from read-back archive): " << worst_diff << "\n";

    // Cleanup.
    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (worst_diff < 1e-3) {
        std::cout << "PASS\n";
        return 0;
    }
    std::cout << "FAIL: numeric drift > 1e-3\n";
    return 1;
}
