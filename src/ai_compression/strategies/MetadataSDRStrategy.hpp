#ifndef METADATA_SDR_STRATEGY_HPP
#define METADATA_SDR_STRATEGY_HPP

#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace CortexAICompression {

/**
 * MetadataSDRStrategy implements a specialized compression strategy for metadata
 * using Sparse Distributed Representations (SDRs) with reversible encoding.
 * 
 * This strategy handles metadata as strings and ensures that the original information
 * can be fully recovered during decompression using a direct SDR encoding approach.
 */
class MetadataSDRStrategy : public ICompressionStrategy {
public:
    // Constructor with configurable parameters
    MetadataSDRStrategy(float sparsity = 0.02f, size_t sdrWidth = 2048);
    
    // Compression/Decompression Interface
    std::vector<std::byte> compress(const ModelSegment& segment) const override;
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData,
                                    SegmentType originalType,
                                    size_t originalSize) const override;

    // Sparsity Configuration
    void setSparsity(float sparsity) { sparsity_ = sparsity; }
    float getSparsity() const { return sparsity_; }
    
    // SDR Width Configuration
    void setSDRWidth(size_t width) { sdrWidth_ = width; }
    size_t getSDRWidth() const { return sdrWidth_; }

private:
    // Core encoding/decoding methods
    std::vector<size_t> encodeString(const std::string& str) const;
    std::string decodeString(const std::vector<size_t>& indices) const;
    
    // Binary data encoding/decoding methods for exact protobuf preservation
    std::vector<size_t> encodeBinaryData(const std::vector<std::byte>& data) const;
    std::vector<std::byte> decodeBinaryData(const std::vector<size_t>& indices, size_t originalSize) const;
    
    // Compression parameters
    float sparsity_;
    size_t sdrWidth_;
};

} // namespace CortexAICompression

#endif // METADATA_SDR_STRATEGY_HPP
