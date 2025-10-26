/**
 * @file SDRIndexStorage.hpp
 * @brief SDR index-based compression for sparse tensors.
 */
#ifndef SDR_INDEX_STORAGE_HPP
#define SDR_INDEX_STORAGE_HPP

#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm> // For std::sort
#include <functional>

namespace CortexAICompression {

/**
 * @brief Store sparse tensors as index lists with delta + varint coding.
 */
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

    // Specialized tensor decoders for various formats (public for static table access)
    static std::vector<std::byte> decodeFormat88Tensor(const std::vector<std::byte>& compressedData, size_t originalSize);
    static std::vector<std::byte> decodeFormat0xD0Tensor(const std::vector<std::byte>& compressedData, size_t originalSize);
    static std::vector<std::byte> decodeFormat0x90Tensor(const std::vector<std::byte>& compressedData, size_t originalSize);
    static std::vector<std::byte> decodeFormatBiasTensor(const std::vector<std::byte>& compressedData, size_t originalSize, uint8_t formatFlag);
    static std::vector<std::byte> createTensorWithPattern(size_t originalSize, const std::vector<std::byte>& sourceData);

    // Streaming decode helpers for zero-decompression inference
    // Visits each sparse index in the compressed buffer (handles optional leading format flag)
    void forEachIndex(const std::vector<std::byte>& compressedData,
                      const std::function<void(size_t)>& visitor) const;

    // Visits each (index, value) pair in the compressed buffer (0x95/0x96 formats)
    // Values are de-quantized to float using the encoded scale
    void forEachIndexValue(const std::vector<std::byte>& compressedData,
                           size_t originalSize,
                           const std::function<void(size_t, float)>& visitor) const;

    // Weight preservation decompression (public for access from decompress method)
    std::vector<std::byte> decompressIndicesWithValues(const std::vector<std::byte>& compressedData,
                                                     size_t originalSize) const;


private:
    float sparsity_;

    // Core Helpers
    std::vector<std::byte> compressIndicesWithDelta(const std::vector<size_t>& indices) const;
    std::vector<std::byte> compressIndicesWithValues(const std::vector<std::pair<size_t, float>>& indexValuePairs) const;
    std::vector<size_t> extractSignificantIndices(const ModelSegment& segment) const;
    std::vector<std::pair<size_t, float>> extractSignificantIndicesWithValues(const ModelSegment& segment) const;

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