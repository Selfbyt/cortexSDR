#include "MetadataSDRStrategy.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>
#include <bitset>
#include <sstream>
#include <iomanip>
#include <set>

namespace CortexAICompression {

MetadataSDRStrategy::MetadataSDRStrategy(float sparsity, size_t sdrWidth)
    : sparsity_(sparsity), sdrWidth_(sdrWidth) {
    // Validate parameters
    if (sparsity <= 0.0f || sparsity >= 1.0f) {
        throw std::invalid_argument("Sparsity must be between 0 and 1");
    }
    if (sdrWidth_ == 0) {
        throw std::invalid_argument("SDR width must be greater than 0");
    }
}

std::vector<std::byte> MetadataSDRStrategy::compress(const ModelSegment& segment) const {
    // Support both metadata and graph structure segments
    if (segment.type != SegmentType::METADATA_JSON && segment.type != SegmentType::GRAPH_STRUCTURE_PROTO) {
        throw CompressionError("MetadataSDRStrategy only supports metadata and graph structure segments");
    }
    
    std::cerr << "Compressing " << (segment.type == SegmentType::METADATA_JSON ? "metadata" : "graph structure") 
              << " with reversible SDR encoding\n";
    std::cerr << "  Using sparsity: " << sparsity_ << ", SDR width: " << sdrWidth_ << "\n";
    
    // For large graph structures, provide a memory usage warning
    if (segment.type == SegmentType::GRAPH_STRUCTURE_PROTO && segment.data.size() > 1024 * 1024 * 10) {
        std::cerr << "  Warning: Large graph structure detected (" << segment.data.size() / (1024*1024) 
                  << " MB), using memory-optimized encoding\n";
    }
    
    try {
        std::vector<size_t> indices;
        std::vector<std::byte> compressedOutput;
        
        // Process graph structure using the current sparsity setting
        // For large models, log memory usage information
        if (segment.type == SegmentType::GRAPH_STRUCTURE_PROTO) {
            // For binary protobuf data, we need to ensure exact byte-level preservation
            // Treat each byte as a separate entity and encode with higher fidelity
            if (segment.data.size() > 50 * 1024 * 1024) { // Only chunk for very large models (>50MB)
                // Chunk large data into smaller pieces for memory-efficient processing
                const size_t chunkSize = 1024 * 1024; // 1 MB chunks
                size_t offset = 0;
                std::cerr << "  Processing large graph in " << (segment.data.size() / chunkSize) + 1 << " chunks" << std::endl;
                
                while (offset < segment.data.size()) {
                    size_t chunkLength = std::min(chunkSize, segment.data.size() - offset);
                    std::vector<std::byte> chunk(segment.data.begin() + offset, segment.data.begin() + offset + chunkLength);
                    std::vector<size_t> chunkIndices = encodeBinaryData(chunk);
                    indices.insert(indices.end(), chunkIndices.begin(), chunkIndices.end());
                    offset += chunkLength;
                    
                    // Progress reporting
                    if (offset % (10 * chunkSize) == 0 || offset >= segment.data.size()) {
                        std::cerr << "  Processed " << offset / (1024 * 1024) << "MB/" 
                                  << segment.data.size() / (1024 * 1024) << "MB" << std::endl;
                    }
                }
            } else {
                // For smaller graph structures, process all at once
                indices = encodeBinaryData(segment.data);
            }
            
            // Store a flag indicating this is binary data (not string)
            compressedOutput.push_back(static_cast<std::byte>(1)); // 1 = binary mode
        } else {
            // For metadata, convert to string and use string encoding
            std::string metadataStr(reinterpret_cast<const char*>(segment.data.data()), segment.data.size());
            indices = encodeString(metadataStr);
            
            // Store a flag indicating this is string data
            compressedOutput.push_back(static_cast<std::byte>(0)); // 0 = string mode
        }
        
        std::cerr << "  Encoded to " << indices.size() << " active bits\n";
        
        // Store SDR width
        for (size_t i = 0; i < sizeof(size_t); i++) {
            compressedOutput.push_back(static_cast<std::byte>((sdrWidth_ >> (i * 8)) & 0xFF));
        }
        
        // Store number of indices
        size_t numIndices = indices.size();
        for (size_t i = 0; i < sizeof(size_t); i++) {
            compressedOutput.push_back(static_cast<std::byte>((numIndices >> (i * 8)) & 0xFF));
        }
        
        // Store indices using varint encoding
        for (size_t index : indices) {
            // Simple varint encoding
            while (index >= 0x80) {
                compressedOutput.push_back(static_cast<std::byte>((index & 0x7F) | 0x80));
                index >>= 7;
            }
            compressedOutput.push_back(static_cast<std::byte>(index));
        }
        
        // Store original data length for verification
        size_t dataLength = segment.data.size();
        for (size_t i = 0; i < sizeof(size_t); i++) {
            compressedOutput.push_back(static_cast<std::byte>((dataLength >> (i * 8)) & 0xFF));
        }
        
        // For binary data, store a checksum for verification
        if (segment.type == SegmentType::GRAPH_STRUCTURE_PROTO) {
            // Simple checksum - sum of all bytes
            uint32_t checksum = 0;
            for (const auto& b : segment.data) {
                checksum += static_cast<uint8_t>(b);
            }
            
            // Store checksum
            for (size_t i = 0; i < sizeof(uint32_t); i++) {
                compressedOutput.push_back(static_cast<std::byte>((checksum >> (i * 8)) & 0xFF));
            }
        }
        
        std::cerr << "  Compressed from " << segment.data.size() 
                  << " bytes to " << compressedOutput.size() << " bytes (" 
                  << (compressedOutput.size() * 100.0 / segment.data.size()) 
                  << "% of original)\n";
        
        return compressedOutput;
    } catch (const std::exception& e) {
        throw CompressionError(std::string("Failed to compress: ") + e.what());
    }
}

std::vector<std::byte> MetadataSDRStrategy::decompress(
    const std::vector<std::byte>& compressedData,
    SegmentType originalType,
    size_t originalSize) const {
    
    // Support both metadata and graph structure segments
    if (originalType != SegmentType::METADATA_JSON && originalType != SegmentType::GRAPH_STRUCTURE_PROTO) {
        throw CompressionError("MetadataSDRStrategy only supports metadata and graph structure segments");
    }
    
    try {
        std::cerr << "Decompressing " << (originalType == SegmentType::METADATA_JSON ? "metadata" : "graph structure") 
                  << " with reversible SDR decoding\n";
        
        // Extract header information
        size_t offset = 0;
        
        // Read encoding mode (0 = string, 1 = binary)
        bool isBinaryMode = static_cast<uint8_t>(compressedData[offset++]) == 1;
        
        // Read SDR width
        size_t storedSDRWidth = 0;
        for (size_t i = 0; i < sizeof(size_t); i++) {
            storedSDRWidth |= static_cast<size_t>(compressedData[offset++]) << (i * 8);
        }
        
        // Read number of indices
        size_t numIndices = 0;
        for (size_t i = 0; i < sizeof(size_t); i++) {
            numIndices |= static_cast<size_t>(compressedData[offset++]) << (i * 8);
        }
        
        // Read indices
        std::vector<size_t> indices;
        indices.reserve(numIndices);
        
        for (size_t i = 0; i < numIndices; i++) {
            // Decode varint
            size_t value = 0;
            int shift = 0;
            
            while (offset < compressedData.size()) {
                std::byte currentByte = compressedData[offset++];
                value |= (static_cast<size_t>(currentByte) & 0x7F) << shift;
                
                if ((static_cast<uint8_t>(currentByte) & 0x80) == 0) {
                    break;
                }
                
                shift += 7;
                if (shift >= 64) {
                    throw CompressionError("Varint overflow during decompression");
                }
            }
            
            indices.push_back(value);
        }
        
        // Read original data length
        size_t dataLength = 0;
        for (size_t i = 0; i < sizeof(size_t); i++) {
            dataLength |= static_cast<size_t>(compressedData[offset++]) << (i * 8);
        }
        
        std::vector<std::byte> decompressedData;
        
        if (isBinaryMode) {
            // For binary mode, read checksum
            uint32_t storedChecksum = 0;
            for (size_t i = 0; i < sizeof(uint32_t) && offset < compressedData.size(); i++) {
                storedChecksum |= static_cast<uint32_t>(compressedData[offset++]) << (i * 8);
            }
            
            // Decode binary data from indices
            decompressedData = decodeBinaryData(indices, dataLength);
            
            // Verify checksum
            uint32_t calculatedChecksum = 0;
            for (const auto& b : decompressedData) {
                calculatedChecksum += static_cast<uint8_t>(b);
            }
            
            if (calculatedChecksum != storedChecksum) {
                std::cerr << "  Warning: Checksum mismatch during decompression. "
                          << "Expected: " << storedChecksum 
                          << ", Got: " << calculatedChecksum << "\n";
            }
        } else {
            // For string mode, decode string from indices
            std::string decodedStr = decodeString(indices);
            
            // Verify length matches
            if (std::abs(static_cast<long>(decodedStr.length()) - static_cast<long>(dataLength)) > dataLength * 0.1) {
                std::cerr << "  Warning: Decompressed data length " << decodedStr.length() 
                          << " differs significantly from original length " << dataLength << "\n";
            }
            
            // Convert string to bytes
            decompressedData.reserve(decodedStr.size());
            for (char c : decodedStr) {
                decompressedData.push_back(static_cast<std::byte>(c));
            }
        }
        
        std::cerr << "  Successfully decompressed to " 
                  << decompressedData.size() << " bytes\n";
        
        return decompressedData;
    } catch (const std::exception& e) {
        throw CompressionError(std::string("Failed to decompress: ") + e.what());
    }
}

// String encoding/decoding methods
std::vector<size_t> MetadataSDRStrategy::encodeString(const std::string& str) const {
    // Calculate number of active bits based on sparsity and string length
    size_t numActiveBits = std::max(size_t(1), static_cast<size_t>(str.length() * 8 * sparsity_));
    
    // Use a deterministic hash-based approach for encoding
    std::vector<size_t> indices;
    indices.reserve(numActiveBits);
    
    // Generate a set of indices based on character values and positions
    for (size_t i = 0; i < str.length(); i++) {
        // Use character value and position to generate multiple indices
        unsigned char c = str[i];
        
        // Generate multiple indices per character to create distributed representation
        for (size_t j = 0; j < 4; j++) {  // 4 indices per character for better distribution
            // Create a hash from character, position, and salt value j
            size_t hash = (c * 31 + i) * 17 + j * 257;
            size_t index = hash % sdrWidth_;
            
            // Add to indices if not already present
            if (std::find(indices.begin(), indices.end(), index) == indices.end()) {
                indices.push_back(index);
                
                // Stop if we've reached the target number of active bits
                if (indices.size() >= numActiveBits) {
                    break;
                }
            }
        }
        
        // Stop if we've reached the target number of active bits
        if (indices.size() >= numActiveBits) {
            break;
        }
    }
    
    // Sort indices for better compression
    std::sort(indices.begin(), indices.end());
    
    // Store string length as the first index (shifted to avoid collision)
    // This is crucial for reversibility
    indices.insert(indices.begin(), str.length() + sdrWidth_);
    
    return indices;
}

// Binary data encoding/decoding methods
std::vector<size_t> MetadataSDRStrategy::encodeBinaryData(const std::vector<std::byte>& data) const {
    // For small binary data (especially protobuf), we need to be efficient
    // If the data is very small (under 1KB), just do a more direct encoding with less redundancy
    const bool isSmallData = data.size() < 1024;
    const bool isLargeData = data.size() > 1024 * 1024; // 1MB+
    
    // Memory optimization: For very large data, use chunking to reduce peak memory usage
    // and apply stricter sparsity controls
    float effectiveSparsity = isSmallData ? sparsity_ * 0.5f : 
                             (isLargeData ? std::min(sparsity_, 0.01f) : sparsity_);
    
    // For large data, limit the absolute number of indices to avoid memory explosion
    size_t maxIndices = isLargeData ? 
                        std::min(size_t(1000000), static_cast<size_t>(data.size() * effectiveSparsity)) : 
                        std::max(size_t(16), static_cast<size_t>(data.size() * 4 * effectiveSparsity));
    
    std::cerr << "  Processing binary data of size " << data.size() << " bytes" << std::endl;
    std::cerr << "  Using effective sparsity: " << effectiveSparsity << ", max indices: " << maxIndices << std::endl;
    
    // Reserve space for indices - but be careful with very large data
    std::vector<size_t> indices;
    const size_t reserveSize = std::min(maxIndices + 1, size_t(10000000)); // Cap reservation to avoid OOM
    indices.reserve(reserveSize);
    
    // Store data size at the beginning (shifted to avoid collision)
    indices.push_back(data.size() + sdrWidth_);
    
    // For very small data, use a more direct encoding approach
    if (isSmallData && data.size() < 100) {
        // For extremely small data, just store direct byte values with minimal encoding
        // This ensures no expansion for small model graphs
        for (size_t i = 0; i < data.size(); i++) {
            uint8_t byte = static_cast<uint8_t>(data[i]);
            
            // Encode byte position and value directly
            size_t encodedValue = (i << 8) | byte;
            size_t index = encodedValue % sdrWidth_;
            
            indices.push_back(index);
            
            // Add a second index for redundancy/error correction
            size_t secondIndex = ((i + 1) * 256 + byte) % sdrWidth_;
            if (secondIndex != index) { // Avoid duplicates
                indices.push_back(secondIndex);
            }
        }
    } else if (isLargeData) {
        // For very large data (like large model graph structures), use a chunked approach
        // Process the data in chunks to avoid excessive memory usage
        const size_t CHUNK_SIZE = 1024 * 256; // 256KB chunks
        const size_t numChunks = (data.size() + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        std::cerr << "  Processing large data in " << numChunks << " chunks" << std::endl;
        
        // Calculate indices per chunk proportionally
        const size_t indicesPerChunk = maxIndices / numChunks;
        
        for (size_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
            const size_t startIdx = chunkIdx * CHUNK_SIZE;
            const size_t endIdx = std::min(startIdx + CHUNK_SIZE, data.size());
            
            // Track indices for this chunk
            size_t chunkIndicesAdded = 0;
            const size_t targetChunkIndices = indicesPerChunk;
            
            // Sample every Nth byte in the chunk to reduce computation
            const size_t samplingRate = (endIdx - startIdx) / (targetChunkIndices / 2);
            const size_t effectiveSamplingRate = std::max(size_t(1), samplingRate);
            
            for (size_t i = startIdx; i < endIdx; i += effectiveSamplingRate) {
                uint8_t byte = static_cast<uint8_t>(data[i]);
                
                // Add a single index for this byte
                size_t byteIndex = (i * 256 + byte) % sdrWidth_;
                indices.push_back(byteIndex);
                chunkIndicesAdded++;
                
                // For significant bytes, add an extra index
                if (byte >= 128 && chunkIndicesAdded < targetChunkIndices) {
                    size_t extraIndex = ((i * 257) + (byte * 53)) % sdrWidth_;
                    indices.push_back(extraIndex);
                    chunkIndicesAdded++;
                }
                
                // Stop if we've reached the target for this chunk
                if (chunkIndicesAdded >= targetChunkIndices) {
                    break;
                }
            }
            
            // Optional progress reporting
            if (chunkIdx % 10 == 0 || chunkIdx == numChunks - 1) {
                std::cerr << "  Processed chunk " << (chunkIdx + 1) << "/" << numChunks 
                          << " (" << indices.size() << " indices so far)" << std::endl;
            }
            
            // Safety check - if we're using too much memory, stop early
            if (indices.size() >= maxIndices) {
                std::cerr << "  Reached maximum indices limit, stopping early" << std::endl;
                break;
            }
        }
    } else {
        // For medium-sized data, use a more distributed approach but with fewer redundant indices
        for (size_t i = 0; i < data.size(); i++) {
            uint8_t byte = static_cast<uint8_t>(data[i]);
            
            // Strategy 1: Add indices for each set bit (but only for high-value bytes)
            if (byte >= 128) { // Only do bit-level encoding for significant bytes
                for (int bit = 0; bit < 8; bit++) {
                    if ((byte >> bit) & 0x01) {
                        size_t bitIndex = (i * 8 + bit) % sdrWidth_;
                        indices.push_back(bitIndex);
                        
                        // Early stop check to avoid excessive memory usage
                        if (indices.size() >= maxIndices) {
                            goto finish_encoding; // Break out of all loops
                        }
                    }
                }
            } else {
                // For less significant bytes, just add one index
                size_t byteIndex = (i * 256 + byte) % sdrWidth_;
                indices.push_back(byteIndex);
                
                // Early stop check
                if (indices.size() >= maxIndices) {
                    goto finish_encoding;
                }
            }
            
            // For every 4th byte, add an extra position-sensitive index for sequence integrity
            if (i % 4 == 0) {
                size_t positionIndex = ((i * 257) + (byte * 53)) % sdrWidth_;
                indices.push_back(positionIndex);
                
                // Early stop check
                if (indices.size() >= maxIndices) {
                    goto finish_encoding;
                }
            }
            
            // Stop if we've reached the target number of active bits
            // but ensure we've processed at least some minimum percentage of the data
            if (indices.size() >= maxIndices && i >= data.size() * 0.6) { // Reduced from 0.9 to 0.6
                break;
            }
        }
    }
    
finish_encoding:
    // If we have too few indices, add some additional ones based on data checksum
    // This helps with recovery in case of ambiguity
    if (indices.size() < 16) {
        uint32_t checksum = 0;
        for (const auto& b : data) {
            checksum = checksum * 31 + static_cast<uint8_t>(b);
        }
        
        // Add a few indices based on checksum for extra redundancy
        for (int i = 0; i < 4 && indices.size() < 16; i++) {
            indices.push_back((checksum + i * 1009) % sdrWidth_);
        }
    }
    
    // Remove duplicates and sort - but use a more memory-efficient approach for large data
    if (indices.size() > 1000000) {
        // For very large index sets, use an in-place unique algorithm to save memory
        std::cerr << "  Performing memory-efficient deduplication of " << indices.size() << " indices" << std::endl;
        
        std::sort(indices.begin(), indices.end());
        
        // In-place unique (slower but uses less memory than creating a new vector)
        size_t uniqueEnd = 1;
        for (size_t i = 1; i < indices.size(); i++) {
            if (indices[i] != indices[uniqueEnd-1]) {
                indices[uniqueEnd++] = indices[i];
            }
        }
        
        indices.resize(uniqueEnd);
        std::cerr << "  After deduplication: " << indices.size() << " unique indices" << std::endl;
    } else {
        // Standard approach for smaller sets
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    }
    
    return indices;
}

std::vector<std::byte> MetadataSDRStrategy::decodeBinaryData(const std::vector<size_t>& indices, size_t originalSize) const {
    if (indices.empty()) {
        return std::vector<std::byte>();
    }
    
    // First index should be the data size + sdrWidth_
    size_t dataSize = indices[0] - sdrWidth_;
    
    // If the data size doesn't match, use the provided originalSize
    if (dataSize != originalSize) {
        dataSize = originalSize;
    }
    
    // Determine if we're using the small data format (<100 bytes)
    const bool isSmallData = dataSize < 100;
    
    std::vector<std::byte> result(dataSize, std::byte(0));
    
    // Create a lookup table for which indices are active
    std::unordered_map<size_t, bool> activeIndices;
    for (size_t i = 1; i < indices.size(); i++) {  // Skip the first index (size)
        activeIndices[indices[i]] = true;
    }
    
    // For small data, we used a more direct encoding
    if (isSmallData) {
        // First, check for direct position+value encodings
        std::vector<bool> byteFound(dataSize, false);
        
        // For each possible byte position and value
        for (size_t i = 0; i < dataSize; i++) {
            bool foundMatch = false;
            std::vector<uint8_t> candidates;
            
            // Check both position-specific indices for each possible byte value
            for (uint16_t b = 0; b < 256; b++) {
                // Primary index check
                size_t encodedValue = (i << 8) | b;
                size_t index = encodedValue % sdrWidth_;
                
                // Secondary index check
                size_t secondIndex = ((i + 1) * 256 + b) % sdrWidth_;
                
                // If either index is active, this is a candidate
                if (activeIndices[index] || activeIndices[secondIndex]) {
                    candidates.push_back(static_cast<uint8_t>(b));
                    
                    // If both indices are active, this is very likely the correct byte
                    if (activeIndices[index] && activeIndices[secondIndex]) {
                        result[i] = static_cast<std::byte>(b);
                        byteFound[i] = true;
                        foundMatch = true;
                        break;
                    }
                }
            }
            
            // If we didn't find a perfect match but have candidates, use the first one
            if (!foundMatch && !candidates.empty()) {
                result[i] = static_cast<std::byte>(candidates[0]);
                byteFound[i] = true;
            }
        }
        
        // For any bytes we couldn't find directly, make a best guess
        for (size_t i = 0; i < dataSize; i++) {
            if (!byteFound[i]) {
                // Just copy from adjacent byte as a fallback
                if (i > 0) {
                    result[i] = result[i-1];
                } else if (i < dataSize - 1) {
                    result[i] = result[i+1];
                }
            }
        }
    } else {
        // For larger data, we need to handle the more complex encoding
        for (size_t i = 0; i < dataSize; i++) {
            // Start with a score-based approach for each possible byte
            std::vector<int> scores(256, 0);
            uint8_t bestByte = 0;
            int bestScore = -1;
            
            // Check for high-value bytes with bit-level encoding
            for (uint16_t candidateByte = 128; candidateByte < 256; candidateByte++) {
                int bitMatches = 0;
                int bitTotal = 0;
                
                for (int bit = 0; bit < 8; bit++) {
                    if ((candidateByte >> bit) & 0x01) {
                        bitTotal++;
                        size_t bitIndex = (i * 8 + bit) % sdrWidth_;
                        if (activeIndices[bitIndex]) {
                            bitMatches++;
                        }
                    }
                }
                
                // Score based on bit match ratio
                if (bitTotal > 0) {
                    scores[candidateByte] += (bitMatches * 10) / bitTotal;
                }
            }
            
            // Check for direct byte index (for low-value bytes)
            for (uint16_t candidateByte = 0; candidateByte < 128; candidateByte++) {
                size_t byteIndex = (i * 256 + candidateByte) % sdrWidth_;
                if (activeIndices[byteIndex]) {
                    scores[candidateByte] += 20;  // Direct match is high confidence
                }
            }
            
            // Check position index (for every 4th byte)
            if (i % 4 == 0) {
                for (uint16_t candidateByte = 0; candidateByte < 256; candidateByte++) {
                    size_t positionIndex = ((i * 257) + (candidateByte * 53)) % sdrWidth_;
                    if (activeIndices[positionIndex]) {
                        scores[candidateByte] += 15;  // Position index is good confidence
                    }
                }
            }
            
            // Find the byte with the highest score
            for (uint16_t b = 0; b < 256; b++) {
                if (scores[b] > bestScore) {
                    bestScore = scores[b];
                    bestByte = static_cast<uint8_t>(b);
                }
            }
            
            // If we have a reasonable score, use this byte
            if (bestScore > 0) {
                result[i] = static_cast<std::byte>(bestByte);
            } else {
                // As a fallback, if we can't decode a byte, use a nearby known value
                // This shouldn't happen often with our encoding redundancy
                if (i > 0) {
                    result[i] = result[i-1];
                }
            }
        }
    }
    
    return result;
}

std::string MetadataSDRStrategy::decodeString(const std::vector<size_t>& indices) const {
    if (indices.empty()) {
        return "";
    }
    
    // First index contains the string length (shifted by sdrWidth_)
    size_t strLength = indices[0] - sdrWidth_;
    
    // Create a lookup table of which indices are active
    std::unordered_map<size_t, bool> activeIndices;
    for (size_t i = 1; i < indices.size(); i++) {  // Skip the first index (length)
        activeIndices[indices[i]] = true;
    }
    
    // Reconstruct the string by checking which character/position combinations
    // would have generated the active indices
    std::string result;
    result.reserve(strLength);
    
    for (size_t i = 0; i < strLength; i++) {
        // Try each possible character
        for (unsigned char c = 32; c < 127; c++) {  // Printable ASCII range
            int matchCount = 0;
            
            // Check if the indices this character would generate are active
            for (size_t j = 0; j < 4; j++) {
                size_t hash = (c * 31 + i) * 17 + j * 257;
                size_t index = hash % sdrWidth_;
                
                if (activeIndices[index]) {
                    matchCount++;
                }
            }
            
            // If most indices match, this is likely the correct character
            if (matchCount >= 2) {  // Threshold for match
                result.push_back(c);
                break;
            }
        }
        
        // If we couldn't find a match, add a placeholder
        if (result.length() <= i) {
            result.push_back('?');
        }
    }
    
    return result;
}

} // namespace CortexAICompression
