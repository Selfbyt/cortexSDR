/**
 * @file DequantToFP32Strategy.cpp
 * @brief Implementation of the dequant→FP32→gzip fallback strategy.
 */
#include "DequantToFP32Strategy.hpp"
#include "HierarchicalSDRStrategy.hpp"

#include <zlib.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace CortexAICompression {

namespace {
constexpr size_t kZChunk = 16384;

std::vector<std::byte> gzipDeflate(const float* fp32, size_t nFloats, int level) {
    z_stream strm{};
    if (deflateInit2(&strm, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw CompressionError("DequantToFP32Strategy: zlib deflateInit failed");
    }
    std::vector<std::byte> out;
    std::vector<std::byte> buf(kZChunk);
    strm.avail_in = static_cast<uInt>(nFloats * sizeof(float));
    strm.next_in = reinterpret_cast<Bytef*>(const_cast<float*>(fp32));
    int ret;
    do {
        strm.avail_out = static_cast<uInt>(kZChunk);
        strm.next_out = reinterpret_cast<Bytef*>(buf.data());
        ret = deflate(&strm, Z_FINISH);
        if (ret == Z_STREAM_ERROR) {
            deflateEnd(&strm);
            throw CompressionError("DequantToFP32Strategy: zlib deflate Z_STREAM_ERROR");
        }
        const size_t have = kZChunk - strm.avail_out;
        out.insert(out.end(), buf.begin(), buf.begin() + have);
    } while (ret != Z_STREAM_END);
    deflateEnd(&strm);
    return out;
}

std::vector<std::byte> gzipInflate(const std::vector<std::byte>& src, size_t expected_size) {
    z_stream strm{};
    if (inflateInit2(&strm, 15 + 16) != Z_OK) {
        throw CompressionError("DequantToFP32Strategy: zlib inflateInit failed");
    }
    std::vector<std::byte> out;
    out.reserve(expected_size);
    std::vector<std::byte> buf(kZChunk);
    strm.avail_in = static_cast<uInt>(src.size());
    strm.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(src.data()));
    int ret;
    do {
        strm.avail_out = static_cast<uInt>(kZChunk);
        strm.next_out = reinterpret_cast<Bytef*>(buf.data());
        ret = inflate(&strm, Z_NO_FLUSH);
        if (ret == Z_NEED_DICT || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
            inflateEnd(&strm);
            throw CompressionError("DequantToFP32Strategy: zlib inflate error " + std::to_string(ret));
        }
        const size_t have = kZChunk - strm.avail_out;
        out.insert(out.end(), buf.begin(), buf.begin() + have);
    } while (ret != Z_STREAM_END);
    inflateEnd(&strm);
    return out;
}
}  // namespace

DequantToFP32Strategy::DequantToFP32Strategy(int compressionLevel)
    : compression_level_(compressionLevel) {}

std::vector<std::byte> DequantToFP32Strategy::compress(const ModelSegment& segment) const {
    // HSDR's public dequant covers FP32 pass-through, FP16, BF16, INT8,
    // Q4_0, Q8_0, Q4_K, Q5_K, Q6_K. Anything outside that set throws and
    // bubbles up to the strategy chain's next fallback (plain Gzip).
    std::vector<float> fp32 = HierarchicalSDRStrategy::dequantizeToFP32(segment);
    return gzipDeflate(fp32.data(), fp32.size(), compression_level_);
}

std::vector<std::byte> DequantToFP32Strategy::decompress(
    const std::vector<std::byte>& compressed,
    SegmentType /*originalType*/, size_t originalSize) const
{
    auto out = gzipInflate(compressed, originalSize);
    if (out.size() != originalSize) {
        throw CompressionError("DequantToFP32Strategy: inflate size mismatch (got "
            + std::to_string(out.size()) + " expected " + std::to_string(originalSize) + ")");
    }
    return out;
}

}  // namespace CortexAICompression
