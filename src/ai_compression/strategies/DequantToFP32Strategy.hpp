/**
 * @file DequantToFP32Strategy.hpp
 * @brief Fallback strategy: dequantise quantised weight bytes to FP32 then
 * compress the FP32 stream with zlib.
 *
 * The default Gzip fallback stores a segment's bytes verbatim — fine for FP32
 * input, but wrong for Q4_K / Q5_K / Q6_K / FP16 inputs because the inference
 * engine's per-segment readers (especially the embedding lookup) treat the
 * decompressed bytes as a flat FP32 array. Storing Q4_K bytes raw under a
 * "WEIGHTS_FP32"-shaped reader gives garbage.
 *
 * This strategy fixes that: dequantise first (via HSDR's shared dequant
 * helper, which already handles all the GGUF + ONNX dtypes we care about),
 * then gzip the FP32 bytes. On decompress, gunzip back to FP32. The
 * inference engine never has to know anything about the source dtype.
 *
 * Used as a higher-priority fallback than plain Gzip for segments that HSDR
 * can't handle (e.g. the 150k-vocab embedding that OOMs the K-SVD fit) so
 * those still produce a usable FP32 stream at archive read time.
 */
#ifndef DEQUANT_TO_FP32_STRATEGY_HPP
#define DEQUANT_TO_FP32_STRATEGY_HPP

#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"
#include <vector>

namespace CortexAICompression {

class DequantToFP32Strategy : public ICompressionStrategy {
public:
    explicit DequantToFP32Strategy(int compressionLevel = 6);

    /// Compress: dequant segment → FP32 bytes → zlib deflate.
    std::vector<std::byte> compress(const ModelSegment& segment) const override;

    /// Decompress: zlib inflate → FP32 bytes (size = originalSize).
    /// originalSize is the FP32 byte count expected at the reader end, NOT
    /// the source dtype's byte count.
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressed,
                                       SegmentType originalType,
                                       size_t originalSize) const override;

private:
    int compression_level_;
};

}  // namespace CortexAICompression

#endif
