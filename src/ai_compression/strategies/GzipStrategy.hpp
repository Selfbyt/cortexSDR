/**
 * @file GzipStrategy.hpp
 * @brief zlib-deflate based compression strategy.
 */
#ifndef GZIP_STRATEGY_HPP
#define GZIP_STRATEGY_HPP

#include <zlib.h>
#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"

namespace CortexAICompression {

/**
 * @brief Compression strategy using zlib's deflate/inflate.
 */
class GzipStrategy : public ICompressionStrategy {
public:
    /** Configure compression level (default uses zlib's default). */
    explicit GzipStrategy(int compressionLevel = Z_DEFAULT_COMPRESSION) 
        : compression_level_(compressionLevel) {}

    /** Compress data using zlib's deflate. */
    std::vector<std::byte> compress(const ModelSegment& segment) const override;

    /** Decompress data using zlib's inflate. */
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData, SegmentType originalType, size_t originalSize) const override;

private:
    int compression_level_;
    static constexpr size_t CHUNK_SIZE = 16384; // Buffer size for streaming inflate/deflate if needed
};

} // namespace CortexAICompression

#endif // GZIP_STRATEGY_HPP
