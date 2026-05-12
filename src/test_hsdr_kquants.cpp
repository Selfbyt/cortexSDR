/**
 * @file test_hsdr_kquants.cpp
 * @brief Hand-computed unit tests for Q4_K / Q5_K / Q6_K dequantisation.
 *
 * The K-quant block layouts are bit-fiddly and easy to get subtly wrong
 * (one shift off → silent corruption that still "looks reasonable" on real
 * data). To validate the implementation independently of llama.cpp, this
 * test constructs byte streams with known fields (d=1, dmin=0, all-zero
 * mins, specific scales/values) and checks that dequantisation produces the
 * exact expected floats.
 *
 * If these checks pass, the bit layouts match the public spec. If they fail,
 * a specific shift/offset is wrong — easy to localise.
 */
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/core/ModelSegment.hpp"
#include "ai_compression/utils/fp16_convert.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace CortexAICompression;

namespace {

void write_fp16(uint8_t* dst, float v) {
    uint16_t h = Utils::fp32_to_fp16(v);
    std::memcpy(dst, &h, sizeof(uint16_t));
}

// Decode helper: dequantise a one-block byte stream as the named dtype.
// Uses the public dequantizeToFP32 entry point — no V4b fitting involved,
// so test expectations are exact down to FP32 precision.
std::vector<float> decode(const std::vector<uint8_t>& bytes, const std::string& fmt,
                           uint32_t R, uint32_t C) {
    ModelSegment seg;
    seg.type = SegmentType::WEIGHTS_INT8;  // generic tag; data_format string decides path
    seg.name = "kquant.test";
    seg.data.resize(bytes.size());
    std::memcpy(seg.data.data(), bytes.data(), bytes.size());
    seg.original_size = seg.data.size();
    seg.data_format = fmt;
    TensorMetadata meta;
    meta.dimensions = {R, C};
    seg.tensor_metadata = meta;
    return HierarchicalSDRStrategy::dequantizeToFP32(seg);
}

bool nearly(float a, float b, float eps = 1e-6f) {
    return std::abs(a - b) < eps;
}

int run() {
    int failures = 0;

    // ------------------------------------------------------------------
    // Q4_K: hand-computed expectation.
    //
    //   d = 1.0, dmin = 0.0  → element = q * scale, min ignored
    //   scales12 set so sub-block 0 has scale=1, min=0
    //   qs[0] = 0x21 → element[0] = 1, element[1] = 2 (× ds=1 = 1, 2)
    //   qs[1] = 0x43 → element[2] = 3, element[3] = 4
    //   rest of qs = 0 → remaining elements = 0
    //
    // Sub-blocks 1..7 have scale=0 and min=0 → all zero.
    // ------------------------------------------------------------------
    {
        constexpr size_t QK_K = 256;
        std::vector<uint8_t> bytes(144, 0);
        write_fp16(&bytes[0], 1.0f);    // d
        write_fp16(&bytes[2], 0.0f);    // dmin (all-zero mins ⇒ contribution is 0)
        // scales12 layout (per spec, unpack_k_scales_mins):
        //   For is < 4: sc[is] = scales12[is] & 63,  m[is] = scales12[is+4] & 63
        // Sub-block 0: scale=1, min=0
        bytes[4] = 0x01;  // scales12[0] → sc[0]=1
        bytes[8] = 0x00;  // scales12[4] → m[0]=0
        // qs[0] = 0x21 → low nibble=1, high nibble=2
        bytes[16] = 0x21;
        // qs[1] = 0x43 → low nibble=3, high nibble=4
        bytes[17] = 0x43;

        auto out = decode(bytes, "q4_k", 1, QK_K);
        bool ok = true;
        ok &= nearly(out[0], 1.0f);
        ok &= nearly(out[1], 2.0f);
        ok &= nearly(out[2], 3.0f);
        ok &= nearly(out[3], 4.0f);
        for (size_t i = 4; i < 32; ++i) ok &= nearly(out[i], 0.0f);
        // Sub-blocks 1..7 should be 0 (scale=0 there).
        for (size_t i = 32; i < QK_K; ++i) ok &= nearly(out[i], 0.0f);
        std::cout << "  Q4_K basic: " << (ok ? "PASS" : "FAIL") << "\n";
        if (!ok) ++failures;
    }

    // ------------------------------------------------------------------
    // Q4_K with min offset.
    //
    //   d = 2.0, dmin = 1.0
    //   sub-block 0: scale = 3, min = 4
    //   qs[0] = 0x05 → q[0]=5, q[1]=0
    //   element[0] = 5 * (2*3) - 1*4 = 30 - 4 = 26
    //   element[1] = 0 * 6     - 4   = -4
    // ------------------------------------------------------------------
    {
        constexpr size_t QK_K = 256;
        std::vector<uint8_t> bytes(144, 0);
        write_fp16(&bytes[0], 2.0f);   // d
        write_fp16(&bytes[2], 1.0f);   // dmin
        bytes[4] = 3;   // scales12[0] → sc[0]=3
        bytes[8] = 4;   // scales12[4] → m[0]=4
        bytes[16] = 0x05;  // qs[0]

        auto out = decode(bytes, "q4_k", 1, QK_K);
        bool ok = true;
        ok &= nearly(out[0], 26.0f);
        ok &= nearly(out[1], -4.0f);
        std::cout << "  Q4_K with min offset: " << (ok ? "PASS" : "FAIL")
                  << "  (got " << out[0] << ", " << out[1] << ")\n";
        if (!ok) ++failures;
    }

    // ------------------------------------------------------------------
    // Q5_K: 5-bit values, one extra high bit from qh[].
    //
    //   d = 1.0, dmin = 0.0, sub-block 0 scale=1, min=0
    //   qh[0] = 0x01 → element[0] has high bit set
    //   qs[0] = 0x02 → element[0] low = 2, element[1] low = 0
    //   element[0] full = (1 << 4) | 2 = 18 → value = 18
    //   element[1] high bit = (qh[0] >> 1) & 1 = 0; low = 0 → value = 0
    // ------------------------------------------------------------------
    {
        constexpr size_t QK_K = 256;
        std::vector<uint8_t> bytes(176, 0);
        write_fp16(&bytes[0], 1.0f);   // d
        write_fp16(&bytes[2], 0.0f);   // dmin
        bytes[4] = 1;   // scales12[0] → sc[0]=1
        bytes[8] = 0;   // scales12[4] → m[0]=0
        bytes[16] = 0x01;  // qh[0] → element 0 has high bit, element 1 does not
        bytes[16 + 32] = 0x02;  // qs[0] (qs starts after qh[QK_K/8]=32 bytes)

        auto out = decode(bytes, "q5_k", 1, QK_K);
        bool ok = true;
        ok &= nearly(out[0], 18.0f);
        ok &= nearly(out[1], 0.0f);
        std::cout << "  Q5_K basic: " << (ok ? "PASS" : "FAIL")
                  << "  (got " << out[0] << ", " << out[1] << ")\n";
        if (!ok) ++failures;
    }

    // ------------------------------------------------------------------
    // Q6_K: 6-bit signed values centred at -32.
    //
    //   d = 1.0
    //   Sub-block 0 (within half 0) has scale = 2
    //   ql[0] = 0x21 → element[0] low = 1, element[1] low = 2
    //   qh[0] = 0x00 → element[0] high = 0, element[1] high = 0
    //   element[0] unsigned = 1; signed = 1 - 32 = -31; → value = -31 * 1 * 2 = -62
    //   element[1] unsigned = 2; signed = -30;        → value = -30 * 1 * 2 = -60
    // ------------------------------------------------------------------
    {
        constexpr size_t QK_K = 256;
        std::vector<uint8_t> bytes(210, 0);
        // ql at offset 0, qh at QK_K/2 = 128, scales at 128+64=192, d at 192+16=208
        bytes[0] = 0x21;  // ql[0]
        bytes[128] = 0x00;  // qh[0]
        bytes[192] = static_cast<uint8_t>(static_cast<int8_t>(2));  // scales[0] = 2
        write_fp16(&bytes[208], 1.0f);  // d

        auto out = decode(bytes, "q6_k", 1, QK_K);
        bool ok = true;
        ok &= nearly(out[0], -62.0f);
        ok &= nearly(out[1], -60.0f);
        std::cout << "  Q6_K basic: " << (ok ? "PASS" : "FAIL")
                  << "  (got " << out[0] << ", " << out[1] << ")\n";
        if (!ok) ++failures;
    }

    // ------------------------------------------------------------------
    // Q6_K: exercise the high-bit (qh) part.
    //
    //   Same as above but qh[0] = 0x03 → element[0] high = 3 (bits [1:0])
    //   element[0] unsigned = (3 << 4) | 1 = 49; signed = 49 - 32 = 17
    //   → value = 17 * 1 * 2 = 34
    //   element[1] high = (qh[0] >> 2) & 3 = 0; unsigned = 2; signed = -30 → -60
    // ------------------------------------------------------------------
    {
        constexpr size_t QK_K = 256;
        std::vector<uint8_t> bytes(210, 0);
        bytes[0] = 0x21;
        bytes[128] = 0x03;
        bytes[192] = static_cast<uint8_t>(static_cast<int8_t>(2));
        write_fp16(&bytes[208], 1.0f);

        auto out = decode(bytes, "q6_k", 1, QK_K);
        bool ok = true;
        ok &= nearly(out[0], 34.0f);
        ok &= nearly(out[1], -60.0f);
        std::cout << "  Q6_K with qh bits: " << (ok ? "PASS" : "FAIL")
                  << "  (got " << out[0] << ", " << out[1] << ")\n";
        if (!ok) ++failures;
    }

    return failures;
}

}  // namespace

int main() {
    std::cout << "K-quant hand-computed unit tests\n";
    int failures = run();
    if (failures == 0) {
        std::cout << "All K-quant layouts match the spec.\n";
        return 0;
    }
    std::cout << failures << " test(s) FAILED — bit layout disagrees with spec.\n";
    return 1;
}
