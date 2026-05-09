/**
 * @file AdaptiveSDRStrategy.cpp
 * @brief Implementation of adaptive sparse distributed representation compression
 * 
 * This file implements the AdaptiveSDRStrategy class which provides intelligent
 * compression using sparse distributed representations with adaptive parameters.
 * The strategy automatically selects optimal compression parameters based on
 * data characteristics to maximize compression efficiency.
 * 
 * Key Features:
 * - Adaptive sparsity level selection based on data analysis
 * - Dynamic SDR width adjustment for optimal representation
 * - Small data handling with specialized compression
 * - Memory-efficient sparse encoding with bit-packing
 * - Performance monitoring and parameter tuning
 */

#include "AdaptiveSDRStrategy.hpp"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cassert>
#include <iomanip>

namespace CortexAICompression {
namespace {
bool isMetadataLikeSegmentType(SegmentType type) {
    return type == SegmentType::METADATA_JSON ||
           type == SegmentType::GRAPH_STRUCTURE_PROTO ||
           type == SegmentType::CONFIG ||
           type == SegmentType::TOKENIZER_VOCAB ||
           type == SegmentType::TOKENIZER_MODEL;
}

bool isGGUFTensorFormat(const std::string& data_format) {
    if (data_format.empty()) {
        return false;
    }
    std::string lower = data_format;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower == "f32" ||
           lower == "fp32" ||
           lower == "f16" ||
           lower == "fp16" ||
           lower == "bf16" ||
           lower == "i8" ||
           lower == "int8" ||
           lower.rfind("q", 0) == 0 ||
           lower.rfind("iq", 0) == 0 ||
           lower.rfind("tq", 0) == 0;
}

bool preferLosslessDirectStorage(const ModelSegment& segment) {
    // Only tokenizer data needs to be stored losslessly
    // Weight tensors SHOULD be compressed - that's the whole point!
    if (segment.type == SegmentType::TOKENIZER_VOCAB ||
        segment.type == SegmentType::TOKENIZER_MODEL) {
        return true;
    }

    // Metadata and config should also be lossless
    if (segment.type == SegmentType::METADATA_JSON ||
        segment.type == SegmentType::CONFIG ||
        segment.type == SegmentType::GRAPH_STRUCTURE_PROTO) {
        return true;
    }

    return false;
}
} // namespace

/**
 * @brief Constructor for AdaptiveSDRStrategy with configurable parameters
 * @param sparsity Target sparsity level (0.0-1.0) for sparse representation
 * @param sdrWidth Width of the SDR representation vector
 * @param smallDataThreshold Threshold below which to use specialized small data handling
 * @throws std::invalid_argument if parameters are invalid
 */
AdaptiveSDRStrategy::AdaptiveSDRStrategy(float sparsity, size_t sdrWidth, size_t smallDataThreshold)
    : sparsity_(sparsity), sdrWidth_(sdrWidth), smallDataThreshold_(smallDataThreshold) {
    
    // Validate sparsity parameter
    if (sparsity <= 0.0f || sparsity >= 1.0f) {
        throw std::invalid_argument("Sparsity must be between 0 and 1");
    }
    
    // Validate SDR width parameter
    if (sdrWidth_ == 0) {
        throw std::invalid_argument("SDR width must be greater than 0");
    }
    
    // Delegate strategies created on-demand for memory efficiency
}

/**
 * @brief Update the target sparsity level for compression
 * @param sparsity New sparsity level (0.0-1.0)
 * @throws std::invalid_argument if sparsity is out of valid range
 */
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
    // GGUF/LLM runtime assets must be byte-exact for faithful reconstruction.
    if (preferLosslessDirectStorage(segment)) {
        std::cerr << "Using direct storage for lossless segment '" << segment.name
                  << "' (" << segment.data.size() << " bytes)\n";
        return compressWithDirectStorage(segment);
    }

    // For very small segments, use direct storage approach
    if (segment.data.size() <= smallDataThreshold_) {
        std::cerr << "Using direct storage for small segment '" << segment.name 
                  << "' (" << segment.data.size() << " bytes)" << std::endl;
        return compressWithDirectStorage(segment);
    }
    
    // For larger segments, use metadata-safe strategy for metadata-like content,
    // and tensor SDR strategy for numerical tensors.
    if (isMetadataLikeSegmentType(segment.type)) {
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
    
    // Check the envelope flag (first byte)
    uint8_t encodingFlag = static_cast<uint8_t>(compressedData[0]);
    if (encodingFlag == DIRECT_STORAGE_FLAG) {
        return decompressDirectStorage(compressedData);
    }

    // Recalibrated routing:
    // - metadata/graph payloads are decoded by MetadataSDRStrategy directly
    // - all tensor payloads are decoded by SDRIndexStorageStrategy directly
    if (isMetadataLikeSegmentType(originalType)) {
        return getMetadataStrategy()->decompress(compressedData, originalType, originalSize);
    }
    return getSDRStrategy()->decompress(compressedData, originalType, originalSize);
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
