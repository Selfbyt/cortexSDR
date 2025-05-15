#ifndef SDR_INDEX_STORAGE_HPP
#define SDR_INDEX_STORAGE_HPP

#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm> // For std::sort

namespace CortexAICompression {

class SDRIndexStorageStrategy : public ICompressionStrategy {
public:
    SDRIndexStorageStrategy() : sparsity_(0.02f) {}
    explicit SDRIndexStorageStrategy(float sparsity) : sparsity_(sparsity) {}

    // Compression/Decompression Interface
    std::vector<std::byte> compress(const ModelSegment& segment) const override;
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData,
                                      SegmentType originalType,
                                      size_t originalSize) const override;

    // Sparsity Configuration
    void setSparsity(float sparsity) noexcept { sparsity_ = sparsity; }
    float getSparsity() const noexcept { return sparsity_; }

private:
    float sparsity_;

    // Core Helpers
    std::vector<std::byte> compressIndicesWithDelta(const std::vector<size_t>& indices) const;
    std::vector<size_t> extractSignificantIndices(const ModelSegment& segment) const;

    // Decompression Paths
    std::vector<std::byte> decompressToSparseIndices(const std::vector<std::byte>& compressedData,
                                                   size_t originalSize) const;
    std::vector<std::byte> decompressToTensor(const std::vector<std::byte>& compressedData,
                                            size_t originalSize) const;

    // Varint Helpers (Static)
    static void encodeVarint(std::vector<std::byte>& buffer, size_t value);
    static size_t decodeVarint(const std::vector<std::byte>& buffer, size_t& offset);
};

} // namespace CortexAICompression

#endif // SDR_INDEX_STORAGE_HPP