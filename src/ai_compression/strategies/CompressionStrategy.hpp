/**
 * @file CompressionStrategy.hpp
 * @brief Interface definitions for neural network model compression strategies
 * 
 * This header defines the abstract interfaces and error handling for various
 * compression strategies used in neural network model compression. Provides
 * a plugin-style architecture for different compression algorithms.
 * 
 * Key Features:
 * - Strategy pattern for pluggable compression algorithms
 * - Type-safe error handling with specialized exceptions
 * - Support for both compression and decompression operations
 * - ModelSegment-aware processing for context-sensitive compression
 * - Extensible architecture for custom compression strategies
 */

#ifndef COMPRESSION_STRATEGY_HPP
#define COMPRESSION_STRATEGY_HPP

#include "../core/ModelSegment.hpp"
#include <vector>
#include <stdexcept>
#include <memory>

namespace CortexAICompression {

/**
 * @brief Base exception class for compression and decompression errors
 * 
 * Provides specialized error handling for compression-related failures
 * with meaningful error messages and context preservation.
 */
class CompressionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Abstract interface for neural network compression strategies
 * 
 * Defines the contract that all compression strategies must implement.
 * Supports both compression and decompression operations with proper
 * error handling and ModelSegment context awareness.
 */
class ICompressionStrategy {
public:
    virtual ~ICompressionStrategy() = default;

    /**
     * @brief Compress data within a ModelSegment using strategy-specific algorithm
     * @param segment ModelSegment containing data to compress
     * @return Compressed data as byte vector
     * @throws CompressionError if compression fails
     * 
     * Implementations should analyze the segment type and data characteristics
     * to apply optimal compression parameters and algorithms.
     */
    virtual std::vector<std::byte> compress(const ModelSegment& segment) const = 0;

    /**
     * @brief Decompress data back to original form
     * @param compressedData Previously compressed data
     * @param originalType Original segment type for proper interpretation
     * @param originalSize Expected size after decompression for validation
     * @return Decompressed data as byte vector
     * @throws CompressionError if decompression fails or validation errors occur
     * 
     * Implementations must ensure bit-perfect reconstruction of original data
     * and validate output size and format consistency.
     */
    virtual std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData, SegmentType originalType, size_t originalSize) const = 0;

    // Optional: Returns a name or identifier for the strategy (e.g., "RLE", "GZIP")
    // virtual std::string getName() const = 0;
};

// We will define NumericalRLEStrategy and SDRIndexStrategy later.

} // namespace CortexAICompression

#endif // COMPRESSION_STRATEGY_HPP
