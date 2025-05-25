#ifndef ADAPTIVE_SDR_STRATEGY_HPP
#define ADAPTIVE_SDR_STRATEGY_HPP

#include "CompressionStrategy.hpp"
#include "MetadataSDRStrategy.hpp"
#include "SDRIndexStorage.hpp"
#include "../core/ModelSegment.hpp"
#include <vector>
#include <string>
#include <memory>

namespace CortexAICompression {

/**
 * AdaptiveSDRStrategy implements a size-adaptive compression strategy
 * that automatically switches between direct storage and SDR-based encoding
 * based on data size.
 * 
 * This approach maintains the SDR philosophy but prevents size expansion
 * for small models.
 */
class AdaptiveSDRStrategy : public ICompressionStrategy {
public:
    // Constructor with configurable parameters
    AdaptiveSDRStrategy(float sparsity = 0.02f, size_t sdrWidth = 2048, size_t smallDataThreshold = 1024);
    
    // Compression/Decompression Interface
    std::vector<std::byte> compress(const ModelSegment& segment) const override;
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData,
                                    SegmentType originalType,
                                    size_t originalSize) const override;

    // Sparsity Configuration
    void setSparsity(float sparsity);
    float getSparsity() const { return sparsity_; }
    
    // SDR Width Configuration
    void setSDRWidth(size_t width);
    size_t getSDRWidth() const { return sdrWidth_; }
    
    // Small Data Threshold Configuration
    void setSmallDataThreshold(size_t threshold) { smallDataThreshold_ = threshold; }
    size_t getSmallDataThreshold() const { return smallDataThreshold_; }

private:
    // Delegate strategies - created on demand to save memory
    mutable std::shared_ptr<MetadataSDRStrategy> metadataStrategy_;
    mutable std::shared_ptr<SDRIndexStorageStrategy> sdrStrategy_;
    
    // Get the metadata strategy, lazy-loaded
    std::shared_ptr<MetadataSDRStrategy> getMetadataStrategy() const {
        if (!metadataStrategy_) {
            metadataStrategy_ = std::make_shared<MetadataSDRStrategy>(sparsity_, sdrWidth_);
        }
        return metadataStrategy_;
    }
    
    // Get the SDR strategy, lazy-loaded
    std::shared_ptr<SDRIndexStorageStrategy> getSDRStrategy() const {
        if (!sdrStrategy_) {
            sdrStrategy_ = std::make_shared<SDRIndexStorageStrategy>(sparsity_);
        }
        return sdrStrategy_;
    }
    
    // Configuration
    float sparsity_;
    size_t sdrWidth_;
    size_t smallDataThreshold_; // Threshold below which direct storage is used
    
    // Encoding flags
    static constexpr uint8_t DIRECT_STORAGE_FLAG = 0;
    static constexpr uint8_t SDR_ENCODING_FLAG = 1;
    
    // Helper methods for direct storage approach
    std::vector<std::byte> compressWithDirectStorage(const ModelSegment& segment) const;
    std::vector<std::byte> decompressDirectStorage(const std::vector<std::byte>& compressedData) const;
};

} // namespace CortexAICompression

#endif // ADAPTIVE_SDR_STRATEGY_HPP
