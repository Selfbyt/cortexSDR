#ifndef GZIP_STRATEGY_HPP
#define GZIP_STRATEGY_HPP

#include <zlib.h>
#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"

namespace CortexAICompression {

// Compression strategy using zlib's gzip format (specifically, deflate/inflate)
class GzipStrategy : public ICompressionStrategy {
public:
    // Constructor to set compression level
    explicit GzipStrategy(int compressionLevel = Z_DEFAULT_COMPRESSION) 
        : compression_level_(compressionLevel) {}

    // Compresses data using zlib's deflate.
    std::vector<std::byte> compress(const ModelSegment& segment) const override;

    // Decompresses data using zlib's inflate.
    // originalType is ignored for Gzip but required by the interface.
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData, SegmentType originalType, size_t originalSize) const override;

private:
    int compression_level_;
    static constexpr size_t CHUNK_SIZE = 16384; // Buffer size for streaming inflate/deflate if needed
};

} // namespace CortexAICompression

#endif // GZIP_STRATEGY_HPP
