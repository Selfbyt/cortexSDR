#include "AdaptiveSDRStrategy.hpp"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cassert>

namespace CortexAICompression {

AdaptiveSDRStrategy::AdaptiveSDRStrategy(float sparsity, size_t sdrWidth, size_t smallDataThreshold)
    : sparsity_(sparsity), sdrWidth_(sdrWidth), smallDataThreshold_(smallDataThreshold) {
    
    // Validate parameters
    if (sparsity <= 0.0f || sparsity >= 1.0f) {
        throw std::invalid_argument("Sparsity must be between 0 and 1");
    }
    
    if (sdrWidth_ == 0) {
        throw std::invalid_argument("SDR width must be greater than 0");
    }
    
    // No delegate strategies created yet - they'll be lazy-loaded when needed
}

void AdaptiveSDRStrategy::setSparsity(float sparsity) {
    if (sparsity <= 0.0f || sparsity >= 1.0f) {
        throw std::invalid_argument("Sparsity must be between 0 and 1");
    }
    
    sparsity_ = sparsity;
    
    // Update delegate strategies if they've been created
    if (metadataStrategy_) {
        metadataStrategy_->setSparsity(sparsity);
    }
    
    if (sdrStrategy_) {
        sdrStrategy_->setSparsity(sparsity);
    }
}

void AdaptiveSDRStrategy::setSDRWidth(size_t width) {
    if (width == 0) {
        throw std::invalid_argument("SDR width must be greater than 0");
    }
    
    sdrWidth_ = width;
    
    // Update metadata strategy if it's been created
    if (metadataStrategy_) {
        metadataStrategy_->setSDRWidth(width);
    }
}

std::vector<std::byte> AdaptiveSDRStrategy::compress(const ModelSegment& segment) const {
    // For very small segments, use direct storage approach
    if (segment.data.size() <= smallDataThreshold_) {
        std::cerr << "Using direct storage for small segment '" << segment.name 
                  << "' (" << segment.data.size() << " bytes)" << std::endl;
        return compressWithDirectStorage(segment);
    }
    
    // For larger segments, use the appropriate SDR-based strategy
    if (segment.type == SegmentType::METADATA_JSON || segment.type == SegmentType::GRAPH_STRUCTURE_PROTO) {
        return getMetadataStrategy()->compress(segment);
    } else {
        return getSDRStrategy()->compress(segment);
    }
}

std::vector<std::byte> AdaptiveSDRStrategy::decompress(
    const std::vector<std::byte>& compressedData,
    SegmentType originalType,
    size_t originalSize) const {
    
    // Early return for empty data
    if (compressedData.empty()) {
        return std::vector<std::byte>();
    }
    
    // Check the encoding flag (first byte)
    uint8_t encodingFlag = static_cast<uint8_t>(compressedData[0]);
    
    if (encodingFlag == DIRECT_STORAGE_FLAG) {
        // Direct storage approach
        return decompressDirectStorage(compressedData);
    } else if (encodingFlag == SDR_ENCODING_FLAG) {
        // SDR-based encoding - use the appropriate strategy
        // Skip the flag byte
        std::vector<std::byte> sdrData(compressedData.begin() + 1, compressedData.end());
        
        if (originalType == SegmentType::METADATA_JSON || originalType == SegmentType::GRAPH_STRUCTURE_PROTO) {
            return getMetadataStrategy()->decompress(sdrData, originalType, originalSize);
        } else {
            return getSDRStrategy()->decompress(sdrData, originalType, originalSize);
        }
    } else {
        // Invalid encoding flag - try to recover
        std::cerr << "Warning: Invalid encoding flag 0x" << std::hex << static_cast<int>(encodingFlag) 
                  << std::dec << " for data of type " << static_cast<int>(originalType) << std::endl;
        
        // For tensor weights, try passing to the appropriate strategy based on type
        ModelSegment dummySegment;
        dummySegment.type = originalType;
        
        if (dummySegment.isWeightTensor()) {
            // For weight tensors, use the SDRStrategy
            std::cerr << "Attempting to decompress weight tensor with SDRStrategy" << std::endl;
            return getSDRStrategy()->decompress(compressedData, originalType, originalSize);
        } else {
            // For metadata and graph structure, use the MetadataStrategy
            std::cerr << "Attempting to decompress with MetadataStrategy" << std::endl;
            return getMetadataStrategy()->decompress(compressedData, originalType, originalSize);
        }
    }
}

std::vector<std::byte> AdaptiveSDRStrategy::compressWithDirectStorage(const ModelSegment& segment) const {
    // Simple compression for small data:
    // 1. Flag byte (DIRECT_STORAGE_FLAG)
    // 2. Original data
    
    std::vector<std::byte> compressedData;
    compressedData.reserve(segment.data.size() + 1);
    
    // Add flag byte
    compressedData.push_back(static_cast<std::byte>(DIRECT_STORAGE_FLAG));
    
    // Add original data
    compressedData.insert(compressedData.end(), segment.data.begin(), segment.data.end());
    
    return compressedData;
}

std::vector<std::byte> AdaptiveSDRStrategy::decompressDirectStorage(const std::vector<std::byte>& compressedData) const {
    // Verify the flag byte
    if (compressedData.empty() || static_cast<uint8_t>(compressedData[0]) != DIRECT_STORAGE_FLAG) {
        throw CompressionError("Invalid direct storage data");
    }
    
    // Skip the flag byte and return the original data
    return std::vector<std::byte>(compressedData.begin() + 1, compressedData.end());
}

} // namespace CortexAICompression
