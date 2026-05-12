/**
 * @file test_hsdr_dtypes.cpp
 * @brief Validate that HSDR accepts FP16 / BF16 / INT8 inputs by dequantising
 * to FP32 internally, then compressing as usual. The decompressed output is
 * FP32 regardless of input dtype.
 */
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/core/ModelSegment.hpp"
#include "ai_compression/utils/fp16_convert.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace CortexAICompression;

namespace {

/// Helper: build a synthetic FP32 weight, then mirror it as FP16 / BF16 / INT8.
std::vector<float> makeReferenceWeight(uint32_t R, uint32_t C, uint32_t seed) {
    std::vector<float> w(static_cast<size_t>(R) * C);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (auto& v : w) v = dist(rng);
    return w;
}

ModelSegment segmentFP32(const std::vector<float>& weight, uint32_t R, uint32_t C,
                          const std::string& name) {
    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_FP32;
    seg.name = name;
    seg.original_size = weight.size() * sizeof(float);
    seg.data.resize(seg.original_size);
    std::memcpy(seg.data.data(), weight.data(), seg.data.size());
    seg.data_format = "f32";
    TensorMetadata meta;
    meta.dimensions = {R, C};
    seg.tensor_metadata = meta;
    return seg;
}

ModelSegment segmentFP16(const std::vector<float>& weight, uint32_t R, uint32_t C,
                          const std::string& name) {
    std::vector<uint16_t> fp16(weight.size());
    Utils::fp32_to_fp16_array(weight.data(), fp16.data(), weight.size());

    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_FP16;
    seg.name = name;
    seg.data.resize(fp16.size() * sizeof(uint16_t));
    std::memcpy(seg.data.data(), fp16.data(), seg.data.size());
    seg.original_size = seg.data.size();
    seg.data_format = "f16";
    TensorMetadata meta;
    meta.dimensions = {R, C};
    seg.tensor_metadata = meta;
    return seg;
}

ModelSegment segmentBF16(const std::vector<float>& weight, uint32_t R, uint32_t C,
                          const std::string& name) {
    std::vector<uint16_t> bf16(weight.size());
    for (size_t i = 0; i < weight.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &weight[i], sizeof(float));
        bf16[i] = static_cast<uint16_t>(bits >> 16);
    }

    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_FP16;  // closest existing enum; data_format tags as bf16
    seg.name = name;
    seg.data.resize(bf16.size() * sizeof(uint16_t));
    std::memcpy(seg.data.data(), bf16.data(), seg.data.size());
    seg.original_size = seg.data.size();
    seg.data_format = "bf16";
    TensorMetadata meta;
    meta.dimensions = {R, C};
    seg.tensor_metadata = meta;
    return seg;
}

/// Synthesise a Q4_0 block-quantised segment from FP32 reference. Layout:
///   18 bytes per 32 elements = fp16 scale (2B) + 16B packed 4-bit values.
/// Q4_0 uses offset-8 signed 4-bit (centre at 0, range [-8, 7]).
ModelSegment segmentQ4_0(const std::vector<float>& weight, uint32_t R, uint32_t C,
                          const std::string& name) {
    constexpr size_t QK = 32;
    constexpr size_t BLOCK_BYTES = 2 + QK / 2;  // 18
    if (weight.size() % QK != 0) {
        throw std::runtime_error("segmentQ4_0: element count must be a multiple of 32");
    }
    const size_t n_blocks = weight.size() / QK;
    std::vector<uint8_t> blob(n_blocks * BLOCK_BYTES);
    uint8_t* out = blob.data();
    for (size_t b = 0; b < n_blocks; ++b) {
        const float* src = weight.data() + b * QK;
        // Per-block scale so 4-bit covers the full magnitude.
        float max_abs = 0.0f;
        for (size_t j = 0; j < QK; ++j) max_abs = std::max(max_abs, std::abs(src[j]));
        const float d = max_abs / 7.0f;  // signed 4-bit max is 7 (centre 0)
        const uint16_t d_bits = Utils::fp32_to_fp16(d);
        std::memcpy(out, &d_bits, 2);
        uint8_t* qs = out + 2;
        for (size_t j = 0; j < QK / 2; ++j) {
            int q_low  = static_cast<int>(std::round(src[2 * j    ] / std::max(d, 1e-12f))) + 8;
            int q_high = static_cast<int>(std::round(src[2 * j + 1] / std::max(d, 1e-12f))) + 8;
            q_low  = std::max(0, std::min(15, q_low));
            q_high = std::max(0, std::min(15, q_high));
            qs[j] = static_cast<uint8_t>(q_low | (q_high << 4));
        }
        out += BLOCK_BYTES;
    }
    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_INT4;
    seg.name = name;
    seg.data.resize(blob.size());
    std::memcpy(seg.data.data(), blob.data(), blob.size());
    seg.original_size = seg.data.size();
    seg.data_format = "q4_0";
    TensorMetadata meta;
    meta.dimensions = {R, C};
    seg.tensor_metadata = meta;
    return seg;
}

/// Synthesise a Q8_0 block-quantised segment. 34 bytes per 32 elements:
/// fp16 scale + 32 × int8 values.
ModelSegment segmentQ8_0(const std::vector<float>& weight, uint32_t R, uint32_t C,
                          const std::string& name) {
    constexpr size_t QK = 32;
    constexpr size_t BLOCK_BYTES = 2 + QK;
    if (weight.size() % QK != 0) {
        throw std::runtime_error("segmentQ8_0: element count must be a multiple of 32");
    }
    const size_t n_blocks = weight.size() / QK;
    std::vector<uint8_t> blob(n_blocks * BLOCK_BYTES);
    uint8_t* out = blob.data();
    for (size_t b = 0; b < n_blocks; ++b) {
        const float* src = weight.data() + b * QK;
        float max_abs = 0.0f;
        for (size_t j = 0; j < QK; ++j) max_abs = std::max(max_abs, std::abs(src[j]));
        const float d = max_abs / 127.0f;
        const uint16_t d_bits = Utils::fp32_to_fp16(d);
        std::memcpy(out, &d_bits, 2);
        int8_t* qs = reinterpret_cast<int8_t*>(out + 2);
        for (size_t j = 0; j < QK; ++j) {
            int v = static_cast<int>(std::round(src[j] / std::max(d, 1e-12f)));
            v = std::max(-128, std::min(127, v));
            qs[j] = static_cast<int8_t>(v);
        }
        out += BLOCK_BYTES;
    }
    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_INT8;
    seg.name = name;
    seg.data.resize(blob.size());
    std::memcpy(seg.data.data(), blob.data(), blob.size());
    seg.original_size = seg.data.size();
    seg.data_format = "q8_0";
    TensorMetadata meta;
    meta.dimensions = {R, C};
    seg.tensor_metadata = meta;
    return seg;
}

ModelSegment segmentINT8(const std::vector<float>& weight, uint32_t R, uint32_t C,
                          const std::string& name) {
    // Compute symmetric per-tensor scale so int8 ranges nicely.
    float max_abs = 0.0f;
    for (float v : weight) max_abs = std::max(max_abs, std::abs(v));
    const float scale = max_abs / 127.0f;
    std::vector<int8_t> q(weight.size());
    for (size_t i = 0; i < weight.size(); ++i) {
        int v = static_cast<int>(std::round(weight[i] / std::max(scale, 1e-12f)));
        if (v >  127) v =  127;
        if (v < -128) v = -128;
        q[i] = static_cast<int8_t>(v);
    }

    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_INT8;
    seg.name = name;
    seg.data.resize(q.size() * sizeof(int8_t));
    std::memcpy(seg.data.data(), q.data(), seg.data.size());
    seg.original_size = seg.data.size();
    seg.data_format = "i8";
    TensorMetadata meta;
    meta.dimensions = {R, C};
    meta.scale = scale;
    meta.zero_point = 0.0f;
    seg.tensor_metadata = meta;
    return seg;
}

double normalizedMSE(const float* a, const float* b, size_t n) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += a[i];
    mean /= n;
    double mse = 0.0, var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i]; mse += d * d;
        double dv = a[i] - mean; var += dv * dv;
    }
    return (mse / n) / std::max(var / n, 1e-30);
}

bool runDtypeRoundTrip(const std::string& label, const ModelSegment& seg,
                       const std::vector<float>& reference, double max_nmse) {
    std::cout << "\n=== " << label
              << "  shape=(" << seg.tensor_metadata.value().dimensions[0]
              << ", " << seg.tensor_metadata.value().dimensions[1] << ")"
              << "  data_format='" << seg.data_format << "' ===\n";

    HierarchicalSDRConfig cfg = HierarchicalSDRConfig::forRow1D(
        /*row_width=*/static_cast<uint16_t>(seg.tensor_metadata.value().dimensions[1]),
        /*n_atoms=*/64, /*k_per_stage=*/6);
    cfg.ksvd_iters = 6;
    HierarchicalSDRStrategy strat(cfg, cfg);

    std::vector<std::byte> bytes;
    try {
        bytes = strat.compress(seg);
    } catch (const std::exception& e) {
        std::cerr << "  FAIL: compress threw: " << e.what() << "\n";
        return false;
    }
    std::cout << "  compressed: " << bytes.size() << " bytes\n";

    // Decompress always returns FP32 (R*C*sizeof(float)).
    const size_t expected_fp32_bytes = reference.size() * sizeof(float);
    std::vector<std::byte> out;
    try {
        out = strat.decompress(bytes, seg.type, expected_fp32_bytes);
    } catch (const std::exception& e) {
        std::cerr << "  FAIL: decompress threw: " << e.what() << "\n";
        return false;
    }
    if (out.size() != expected_fp32_bytes) {
        std::cerr << "  FAIL: expected " << expected_fp32_bytes
                  << " bytes back, got " << out.size() << "\n";
        return false;
    }
    const float* rec = reinterpret_cast<const float*>(out.data());
    double nmse = normalizedMSE(reference.data(), rec, reference.size());
    std::cout << "  NMSE (vs original FP32): " << nmse << "\n";
    if (nmse > max_nmse) {
        std::cerr << "  FAIL: NMSE " << nmse << " > limit " << max_nmse << "\n";
        return false;
    }
    std::cout << "  PASS\n";
    return true;
}

}  // namespace

int main() {
    std::cout << "HSDR multi-dtype input test\n";

    const uint32_t R = 64;
    const uint32_t C = 256;
    const auto reference = makeReferenceWeight(R, C, 0xCAFEFADE);

    bool ok = true;
    // FP32 — baseline, should reconstruct best.
    ok &= runDtypeRoundTrip("FP32 input",
                             segmentFP32(reference, R, C, "tensor.fp32"),
                             reference, /*max_nmse=*/0.8);
    // FP16 — tiny precision loss from dtype conversion, NMSE budget basically
    // the same as FP32 because the V4b reconstruction error dominates.
    ok &= runDtypeRoundTrip("FP16 input",
                             segmentFP16(reference, R, C, "tensor.fp16"),
                             reference, /*max_nmse=*/0.8);
    // BF16 — coarser than FP16, looser tolerance.
    ok &= runDtypeRoundTrip("BF16 input",
                             segmentBF16(reference, R, C, "tensor.bf16"),
                             reference, /*max_nmse=*/0.85);
    // INT8 — uniform quant adds noticeable noise; the budget reflects
    // dtype + reconstruction error stacking.
    ok &= runDtypeRoundTrip("INT8 input (symmetric scale)",
                             segmentINT8(reference, R, C, "tensor.int8"),
                             reference, /*max_nmse=*/0.95);
    // Q4_0 — 4-bit per-block quant. Coarser than INT8 but still recoverable.
    ok &= runDtypeRoundTrip("Q4_0 input (GGUF-style 32-elem blocks)",
                             segmentQ4_0(reference, R, C, "tensor.q4_0"),
                             reference, /*max_nmse=*/0.95);
    // Q8_0 — 8-bit per-block. Should perform similarly to plain INT8.
    ok &= runDtypeRoundTrip("Q8_0 input (GGUF-style 32-elem blocks)",
                             segmentQ8_0(reference, R, C, "tensor.q8_0"),
                             reference, /*max_nmse=*/0.95);

    // Verify the unsupported-dtype path still throws cleanly. Q4_K and Q6_K
    // are super-block layouts that require careful spec-based dequant code
    // (deferred to a future round). The CLI surface them as "unsupported".
    std::cout << "\n=== Unsupported dtype rejection (Q4_K super-block) ===\n";
    {
        ModelSegment q4k = segmentINT8(reference, R, C, "tensor.q4_k");
        q4k.data_format = "Q4_K";  // upper-case matches GGUF parser convention
        q4k.tensor_metadata.value().scale.reset();
        q4k.tensor_metadata.value().zero_point.reset();
        q4k.type = SegmentType::WEIGHTS_INT4;
        HierarchicalSDRStrategy strat;
        try {
            (void)strat.compress(q4k);
            std::cerr << "  FAIL: expected throw on Q4_K\n";
            ok = false;
        } catch (const std::exception& e) {
            std::cout << "  OK — threw as expected: " << e.what() << "\n";
        }
    }

    if (ok) {
        std::cout << "\nAll dtype paths OK.\n";
        return 0;
    }
    std::cout << "\nFAIL\n";
    return 1;
}
