#include "SDRIndexStorage.hpp"
#include "../core/ModelSegment.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstring>

namespace CortexAICompression {

// --- Varint Helpers (Static Member Functions) ---
void SDRIndexStorageStrategy::encodeVarint(std::vector<std::byte>& buffer, size_t value) {
    while (value >= 0x80) {
        buffer.push_back(static_cast<std::byte>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    buffer.push_back(static_cast<std::byte>(value));
}

size_t SDRIndexStorageStrategy::decodeVarint(const std::vector<std::byte>& buffer, size_t& offset) {
    size_t value = 0;
    int shift = 0;
    size_t startOffset = offset;
    while (offset < buffer.size()) {
        std::byte currentByte = buffer[offset++];
        value |= (static_cast<size_t>(currentByte) & 0x7F) << shift;
        if ((static_cast<uint8_t>(currentByte) & 0x80) == 0) {
            return value;
        }
        shift += 7;
        if (shift >= 64) {
            throw CompressionError("Varint overflow at offset " + std::to_string(startOffset));
        }
    }
    throw CompressionError("Incomplete varint at offset " + std::to_string(startOffset));
}

// --- compressIndicesWithDelta Implementation ---
std::vector<std::byte> SDRIndexStorageStrategy::compressIndicesWithDelta(const std::vector<size_t>& indices) const {
    std::vector<std::byte> compressedOutput;
    if (indices.empty()) return compressedOutput;

    // Store the number of indices
    encodeVarint(compressedOutput, indices.size());
    
    // Check if indices are sorted for delta encoding
    bool isSorted = std::is_sorted(indices.begin(), indices.end());
    
    // Add a flag to indicate if we're using delta encoding (1) or direct encoding (0)
    compressedOutput.push_back(static_cast<std::byte>(isSorted ? 1 : 0));
    
    if (isSorted) {
        // Delta encoding - store differences between consecutive indices
        // This is more efficient when indices are clustered or evenly spaced
        size_t lastIndex = 0;
        for (size_t index : indices) {
            size_t delta = index - lastIndex;
            encodeVarint(compressedOutput, delta);
            lastIndex = index;
        }
    } else {
        // Sort indices anyway - often more compressible
        std::vector<size_t> sortedIndices = indices;
        std::sort(sortedIndices.begin(), sortedIndices.end());
        
        // Check if sorted compression would be better
        std::vector<std::byte> deltaCompressed;
        encodeVarint(deltaCompressed, sortedIndices.size());
        deltaCompressed.push_back(static_cast<std::byte>(1)); // Mark as delta encoded
        
        size_t lastIndex = 0;
        for (size_t index : sortedIndices) {
            encodeVarint(deltaCompressed, index - lastIndex);
            lastIndex = index;
        }
        
        // Direct encoding of original indices
        std::vector<std::byte> directCompressed;
        for (size_t index : indices) {
            encodeVarint(directCompressed, index);
        }
        
        // Use whichever method produces smaller output
        if (deltaCompressed.size() < directCompressed.size()) {
            std::cerr << "  Using delta encoding (saved " 
                      << directCompressed.size() - deltaCompressed.size() 
                      << " bytes)\n";
            return deltaCompressed;
        } else {
            std::cerr << "  Using direct encoding\n";
            for (size_t index : indices) {
                encodeVarint(compressedOutput, index);
            }
        }
    }
    
    return compressedOutput;
}

// --- decompressToTensor Implementation ---
std::vector<std::byte> SDRIndexStorageStrategy::decompressToTensor(const std::vector<std::byte>& compressedData,
                                                                 size_t originalSize) const {
    if (compressedData.empty()) {
        throw CompressionError("Empty compressed data");
    }
    
    // Create buffer for reconstructed data
    std::vector<std::byte> reconstructedData(originalSize, std::byte{0});
    std::vector<size_t> indices;
    size_t offset = 0;
    
    // Read number of indices
    size_t numIndices = decodeVarint(compressedData, offset);
    
    // Check if we have enough data for the encoding flag
    if (offset >= compressedData.size()) {
        throw CompressionError("Compressed data truncated after index count");
    }
    
    // Read encoding flag (1 = delta encoding, 0 = direct encoding)
    bool isDeltaEncoded = (static_cast<uint8_t>(compressedData[offset++]) != 0);
    
    // Reserve space for indices
    indices.reserve(numIndices);
    
    try {
        if (isDeltaEncoded) {
            // Delta decoding
            size_t lastIndex = 0;
            for (size_t i = 0; i < numIndices; ++i) {
                size_t delta = decodeVarint(compressedData, offset);
                lastIndex += delta;
                indices.push_back(lastIndex);
            }
        } else {
            // Direct decoding
            for (size_t i = 0; i < numIndices; ++i) {
                indices.push_back(decodeVarint(compressedData, offset));
            }
        }
    } catch (const CompressionError& e) {
        throw CompressionError(std::string("Failed to decode indices: ") + e.what());
    }
    
    // Determine if we need to handle header data
    size_t dataOffset = 0;
    size_t dataSize = originalSize;
    
    // Check if this might be an ONNX tensor with header
    if (originalSize >= sizeof(size_t)) {
        // Try to read dimension count from the original size
        size_t dimCount = 0;
        std::memcpy(&dimCount, reconstructedData.data(), sizeof(size_t));
        
        // If dimension count seems valid, adjust data offset
        if (dimCount > 0 && dimCount < 10) { // Reasonable limit for dimensions
            size_t headerSize = sizeof(size_t) * (dimCount + 1);
            if (headerSize < originalSize) {
                dataOffset = headerSize;
                dataSize = originalSize - headerSize;
                std::cerr << "  Detected header in tensor: " << headerSize << " bytes\n";
            }
        }
    }
    
    // Determine element size (assume float32 by default)
    size_t elementSize = 4;
    size_t numElements = dataSize / elementSize;
    
    // Set active bits in the reconstructed tensor
    float* floatData = reinterpret_cast<float*>(reconstructedData.data() + dataOffset);
    for (size_t index : indices) {
        if (index < numElements) {
            floatData[index] = 1.0f;
        }
    }
    
    return reconstructedData;
}

// --- Main Interface Implementations ---

std::vector<std::byte> SDRIndexStorageStrategy::compress(const ModelSegment& segment) const {
    std::vector<std::byte> compressedOutput;
    std::vector<size_t> indices;
    
    std::cerr << "\nCompressing segment '" << segment.name 
              << "' of type " << static_cast<int>(segment.type) 
              << " with size " << segment.data.size() << " bytes\n";
    
    // Print tensor metadata if available
    if (segment.tensor_metadata) {
        std::cerr << "  Tensor dimensions: [";
        for (size_t i = 0; i < segment.tensor_metadata->dimensions.size(); ++i) {
            std::cerr << segment.tensor_metadata->dimensions[i];
            if (i < segment.tensor_metadata->dimensions.size() - 1) {
                std::cerr << ", ";
            }
        }
        std::cerr << "]\n";
        std::cerr << "  Sparsity ratio from metadata: " << segment.tensor_metadata->sparsity_ratio << "\n";
    }
    
    // Handle segment types
    if (segment.type == SegmentType::SPARSE_INDICES) {
        std::cerr << "  Processing as SPARSE_INDICES\n";
        if (segment.data.size() % sizeof(size_t) != 0) {
            throw CompressionError("Input data size is not a multiple of sizeof(size_t).");
        }
        size_t numIndices = segment.data.size() / sizeof(size_t);
        const size_t* indicesPtr = reinterpret_cast<const size_t*>(segment.data.data());
        indices.assign(indicesPtr, indicesPtr + numIndices);
        std::cerr << "  Extracted " << indices.size() << " indices from sparse data\n";
    } else if (segment.type == SegmentType::MODEL_INPUT || 
               segment.type == SegmentType::MODEL_OUTPUT || 
               segment.type == SegmentType::WEIGHTS_FP32 || 
               segment.type == SegmentType::WEIGHTS_FP16 || 
               segment.isWeightTensor()) {
        
        std::cerr << "  Processing as tensor data\n";
        try {
            indices = extractSignificantIndices(segment);
            if (indices.empty()) {
                std::cerr << "  No indices extracted, compression will fail\n";
                throw CompressionError("No indices extracted. Check sparsity or data validity.");
            }
            std::cerr << "  Successfully extracted " << indices.size() << " indices\n";
        } catch (const std::exception& e) {
            std::cerr << "  Exception during index extraction: " << e.what() << "\n";
            throw CompressionError(std::string("Failed to extract indices: ") + e.what());
        }
    } else {
        std::cerr << "  Unsupported segment type: " << static_cast<int>(segment.type) << "\n";
        throw CompressionError("Unsupported segment type: " + std::to_string(static_cast<int>(segment.type)));
    }
    
    // Compress the indices
    try {
        // Compress the indices using delta or direct encoding
        compressedOutput = compressIndicesWithDelta(indices);
        std::cerr << "  Compressed " << indices.size() << " indices to " 
                  << compressedOutput.size() << " bytes\n";
                  
        // For small tensors, we might need to accept less compression
        // to avoid falling back to uncompressed storage
        float compressionThreshold = 1.0f; // Default: must be smaller than original
        
        // For very small segments, allow slightly larger compressed size
        // This prevents small tensors from being stored uncompressed
        if (segment.data.size() < 1024) {
            compressionThreshold = 1.2f; // Allow up to 20% larger for small segments
            std::cerr << "  Small segment, allowing up to 20% size increase\n";
        }
        
        // Check if compression was effective enough
        if (compressedOutput.size() > segment.data.size() * compressionThreshold) {
            std::cerr << "  Compression ineffective (" << compressedOutput.size() 
                      << " > " << segment.data.size() * compressionThreshold 
                      << "), will store uncompressed\n";
            throw CompressionError("Compression ineffective, output larger than threshold");
        }
        
        return compressedOutput;
    } catch (const std::exception& e) {
        std::cerr << "  Exception during compression: " << e.what() << "\n";
        throw CompressionError(std::string("Failed to compress indices: ") + e.what());
    }
}

std::vector<std::byte> SDRIndexStorageStrategy::decompress(const std::vector<std::byte>& compressedData, 
                                                          SegmentType originalType, 
                                                          size_t originalSize) const {
    if (compressedData.empty()) return {};
    
    if (originalType == SegmentType::SPARSE_INDICES) {
        return decompressToSparseIndices(compressedData, originalSize);
    } else if (originalType == SegmentType::MODEL_INPUT || 
               originalType == SegmentType::MODEL_OUTPUT || 
               originalType == SegmentType::WEIGHTS_FP32 || 
               originalType == SegmentType::WEIGHTS_FP16 || 
               originalType == SegmentType::WEIGHTS_INT8 || 
               originalType == SegmentType::WEIGHTS_INT4 || 
               originalType == SegmentType::ATTENTION_WEIGHTS || 
               originalType == SegmentType::FEED_FORWARD_WEIGHTS || 
               originalType == SegmentType::EMBEDDING_WEIGHTS || 
               originalType == SegmentType::LAYER_NORM_WEIGHTS) {
        return decompressToTensor(compressedData, originalSize);
    } else {
        throw CompressionError("Unsupported type for decompression: " + std::to_string(static_cast<int>(originalType)));
    }
}

std::vector<size_t> SDRIndexStorageStrategy::extractSignificantIndices(const ModelSegment& segment) const {
    std::vector<size_t> activeIndices;
    size_t elementSize = 4; // Default to float32
    size_t headerSize = 0; // Default to no header
    size_t expectedElements = 0;
    
    // Determine element size based on segment type
    if (segment.type == SegmentType::WEIGHTS_FP16) {
        elementSize = 2;
    } else if (segment.type == SegmentType::WEIGHTS_INT8) {
        elementSize = 1;
    }
    
    // For ONNX model input/output, detect header size
    if (segment.type == SegmentType::MODEL_INPUT || segment.type == SegmentType::MODEL_OUTPUT) {
        // ONNX models typically include dimension info at the start
        // The header format is: [dim_count (size_t), dim1, dim2, ...]
        if (segment.data.size() >= sizeof(size_t)) {
            // Extract dimension count
            size_t dimCount = 0;
            std::memcpy(&dimCount, segment.data.data(), sizeof(size_t));
            
            // Validate dimension count (sanity check)
            if (dimCount > 0 && dimCount < 10) { // Reasonable limit for dimensions
                headerSize = sizeof(size_t) * (dimCount + 1); // +1 for dim_count itself
                
                // Calculate expected elements from dimensions
                if (headerSize <= segment.data.size()) {
                    expectedElements = 1;
                    for (size_t i = 0; i < dimCount; i++) {
                        size_t dim = 0;
                        std::memcpy(&dim, segment.data.data() + sizeof(size_t) * (i + 1), sizeof(size_t));
                        expectedElements *= dim;
                    }
                    
                    std::cerr << "Detected header in ONNX data: " << headerSize << " bytes, ";
                    std::cerr << dimCount << " dimensions, " << expectedElements << " expected elements\n";
                    
                    // Verify that remaining data size matches expected elements
                    if (segment.data.size() - headerSize != expectedElements * elementSize) {
                        std::cerr << "Warning: Data size after header (" << segment.data.size() - headerSize
                                  << ") doesn't match expected size (" << expectedElements * elementSize << ")\n";
                        // If mismatch, use data size to determine element count
                        expectedElements = (segment.data.size() - headerSize) / elementSize;
                    }
                }
            }
        }
    }
    
    // Use tensor metadata if available and no header was detected
    if (headerSize == 0 && segment.tensor_metadata && !segment.tensor_metadata->dimensions.empty()) {
        expectedElements = 1;
        for (const auto& dim : segment.tensor_metadata->dimensions) {
            expectedElements *= dim;
        }
        
        std::cerr << "Using tensor metadata: " << expectedElements << " expected elements\n";
    }
    
    // Calculate total elements based on data size minus header
    size_t totalElements;
    if (headerSize > 0 && expectedElements > 0) {
        totalElements = expectedElements;
    } else {
        totalElements = (segment.data.size() - headerSize) / elementSize;
    }
    
    // Calculate active bits count based on sparsity parameter
    size_t activeBitsCount = static_cast<size_t>(std::ceil(totalElements * sparsity_));
    if (activeBitsCount == 0) activeBitsCount = 1;
    
    std::cerr << "Processing segment '" << segment.name << "' with " 
              << totalElements << " elements (after " << headerSize << " byte header), targeting " 
              << activeBitsCount << " active bits (" 
              << (sparsity_ * 100.0f) << "% sparsity)\n";
    
    // Handle float32 data
    if (elementSize == 4) {
        // Point to the start of actual tensor data (after header)
        const float* floatData = reinterpret_cast<const float*>(segment.data.data() + headerSize);
        
        // Verify we have valid data
        bool hasValidData = false;
        for (size_t i = 0; i < std::min(totalElements, size_t(100)); ++i) {
            if (std::isfinite(floatData[i]) && floatData[i] != 0.0f) {
                hasValidData = true;
                break;
            }
        }
        
        if (!hasValidData) {
            std::cerr << "Warning: No valid non-zero float data found in first 100 elements\n";
        }
        
        // Create value-index pairs for sorting
        std::vector<std::pair<float, size_t>> valuePairs;
        valuePairs.reserve(totalElements);
        
        for (size_t i = 0; i < totalElements; ++i) {
            if (std::isfinite(floatData[i])) { // Skip NaN and Inf values
                valuePairs.emplace_back(std::abs(floatData[i]), i);
            }
        }
        
        // Sort by absolute value (descending)
        std::sort(valuePairs.begin(), valuePairs.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        
        // Extract top indices based on sparsity
        activeIndices.reserve(activeBitsCount);
        
        // First pass: get non-zero values
        for (size_t i = 0; i < valuePairs.size() && activeIndices.size() < activeBitsCount; ++i) {
            if (valuePairs[i].first > 0.0f) {
                activeIndices.push_back(valuePairs[i].second);
            }
        }
        
        // If we don't have enough non-zero values, add zeros until we reach the target count
        if (activeIndices.size() < activeBitsCount) {
            for (size_t i = 0; i < valuePairs.size() && activeIndices.size() < activeBitsCount; ++i) {
                if (valuePairs[i].first == 0.0f && 
                    std::find(activeIndices.begin(), activeIndices.end(), valuePairs[i].second) == activeIndices.end()) {
                    activeIndices.push_back(valuePairs[i].second);
                }
            }
        }
        
        // If we still don't have enough indices, add some uniformly distributed ones
        if (activeIndices.size() < activeBitsCount && totalElements > 0) {
            size_t step = std::max(size_t(1), totalElements / (activeBitsCount - activeIndices.size() + 1));
            for (size_t i = 0; i < totalElements && activeIndices.size() < activeBitsCount; i += step) {
                if (std::find(activeIndices.begin(), activeIndices.end(), i) == activeIndices.end()) {
                    activeIndices.push_back(i);
                }
            }
        }
        
        // Sort indices for better delta compression
        std::sort(activeIndices.begin(), activeIndices.end());
        
        std::cerr << "Extracted " << activeIndices.size() << " indices out of " 
                  << totalElements << " elements (effective sparsity: " 
                  << (static_cast<float>(activeIndices.size()) / totalElements) * 100.0f 
                  << "%)\n";
    } else {
        // For other element types, use uniform sampling
        if (totalElements > 0) {
            size_t step = std::max(size_t(1), totalElements / activeBitsCount);
            for (size_t i = 0; i < totalElements && activeIndices.size() < activeBitsCount; i += step) {
                activeIndices.push_back(i);
            }
            
            // Sort for better delta compression
            std::sort(activeIndices.begin(), activeIndices.end());
        }
    }
    
    return activeIndices;
}

std::vector<std::byte> SDRIndexStorageStrategy::decompressToSparseIndices(const std::vector<std::byte>& compressedData, 
                                                                        size_t originalSize) const {
    std::vector<size_t> indices;
    size_t offset = 0;
    
    if (compressedData.empty()) {
        throw CompressionError("Empty compressed data");
    }
    
    // Read number of indices
    size_t numIndices = decodeVarint(compressedData, offset);
    
    // Check if we have enough data for the encoding flag
    if (offset >= compressedData.size()) {
        throw CompressionError("Compressed data truncated after index count");
    }
    
    // Read encoding flag (1 = delta encoding, 0 = direct encoding)
    bool isDeltaEncoded = (static_cast<uint8_t>(compressedData[offset++]) != 0);
    
    // Reserve space for indices
    indices.reserve(numIndices);
    
    try {
        if (isDeltaEncoded) {
            // Delta decoding
            size_t lastIndex = 0;
            for (size_t i = 0; i < numIndices; ++i) {
                size_t delta = decodeVarint(compressedData, offset);
                lastIndex += delta;
                indices.push_back(lastIndex);
            }
        } else {
            // Direct decoding
            for (size_t i = 0; i < numIndices; ++i) {
                indices.push_back(decodeVarint(compressedData, offset));
            }
        }
    } catch (const CompressionError& e) {
        throw CompressionError(std::string("Failed to decode indices: ") + e.what());
    }
    
    // Reconstruct original data
    std::vector<std::byte> reconstructedData(indices.size() * sizeof(size_t));
    std::memcpy(reconstructedData.data(), indices.data(), reconstructedData.size());
    return reconstructedData;
}

} // namespace CortexAICompression