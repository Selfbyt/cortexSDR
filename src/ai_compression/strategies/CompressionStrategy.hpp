#ifndef COMPRESSION_STRATEGY_HPP
#define COMPRESSION_STRATEGY_HPP

#include "../core/ModelSegment.hpp"
#include <vector>
#include <stdexcept>
#include <memory> // For std::unique_ptr

namespace CortexAICompression {

// Base class for compression/decompression errors
class CompressionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Interface for a compression strategy
class ICompressionStrategy {
public:
    virtual ~ICompressionStrategy() = default;

    // Compresses the data within a ModelSegment.
    // Modifies the segment's data in place or returns new data.
    // For simplicity, let's return new data for now.
    // Throws CompressionError on failure.
    virtual std::vector<std::byte> compress(const ModelSegment& segment) const = 0;

    // Decompresses data back into its original form (as raw bytes).
    // Requires the original segment type to correctly interpret the data.
    // Throws CompressionError on failure.
    virtual std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData, SegmentType originalType, size_t originalSize) const = 0;

    // Optional: Returns a name or identifier for the strategy (e.g., "RLE", "GZIP")
    // virtual std::string getName() const = 0;
};

// We will define NumericalRLEStrategy and SDRIndexStrategy later.

} // namespace CortexAICompression

#endif // COMPRESSION_STRATEGY_HPP
