#include "StreamingCompressor.hpp"
#include "../core/ArchiveConstants.hpp" // Include for ARCHIVE_MAGIC, ARCHIVE_VERSION
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring> // For std::memcpy

namespace CortexAICompression {

// Helper function to write basic types to the stream (copied for locality)
template<typename T>
void writeBasicTypeLocal(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Helper function to write string (length prefixed) (copied for locality)
void writeStringLocal(std::ostream& stream, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.length());
    if (str.length() > UINT32_MAX) {
        // This check might be better placed before calling this helper
        throw std::runtime_error("Segment name or layer name too long for archive format.");
    }
    writeBasicTypeLocal(stream, len);
    stream.write(str.data(), len);
}


StreamingCompressor::StreamingCompressor(const std::string& outputPath)
    : outputFile_(outputPath, std::ios::binary | std::ios::trunc) // Ensure file is truncated if it exists
{
    if (!outputFile_) {
        throw std::runtime_error("Failed to open output file for writing: " + outputPath);
    }
    // Header is no longer written immediately
}

StreamingCompressor::~StreamingCompressor() {
    // Ensure the archive is finalized and written when the object is destroyed
    if (outputFile_.is_open()) {
        try {
            finalizeArchive();
        } catch (const std::exception& e) {
            std::cerr << "Error during StreamingCompressor destruction (finalizeArchive): " << e.what() << std::endl;
            // Avoid throwing from destructor, but log the error
        }
        outputFile_.close();
    }
}

void StreamingCompressor::handleCompressedSegment(const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData) {
    std::lock_guard<std::mutex> lock(writeMutex_);

    // Store the segment header and data instead of writing immediately
    segments_.emplace_back(header, compressedData);

    // Update statistics
    totalCompressedSize_ += compressedData.size();
    totalOriginalSize_ += header.original_size;

    // Optional: Print progress (can be verbose)
    // std::cout << "Buffered segment '" << header.name << "': "
    //           << header.original_size << " -> " << compressedData.size() << " bytes\n";
}

void StreamingCompressor::finalizeArchive() {
    if (!outputFile_.is_open() || segments_.empty()) {
        if (segments_.empty()) {
             std::cerr << "Warning: Finalizing archive with zero segments." << std::endl;
        }
        if (outputFile_.is_open()) outputFile_.close();
        return;
    }

    // --- Write Archive Header ---
    outputFile_.write(ARCHIVE_MAGIC, sizeof(ARCHIVE_MAGIC));
    writeBasicTypeLocal(outputFile_, ARCHIVE_VERSION);

    // --- Write Number of Segments ---
    uint32_t numSegments = static_cast<uint32_t>(segments_.size());
    writeBasicTypeLocal(outputFile_, numSegments);

    // --- Calculate Index Table Size and Data Offsets ---
    uint64_t currentDataOffset = sizeof(ARCHIVE_MAGIC) + sizeof(ARCHIVE_VERSION) + sizeof(numSegments);
    uint64_t indexTableSize = 0;
    std::vector<uint64_t> dataOffsets;
    dataOffsets.reserve(numSegments);

    for (const auto& segmentPair : segments_) {
        const auto& header = segmentPair.first;
        // Calculate size of this header entry in the index
        uint64_t headerEntrySize = 0;
        headerEntrySize += sizeof(uint16_t) + header.name.length(); // Name length + name
        headerEntrySize += sizeof(uint8_t); // Type
        headerEntrySize += sizeof(uint8_t); // Strategy ID
        headerEntrySize += sizeof(uint64_t); // Original Size (now 8 bytes)
        headerEntrySize += sizeof(uint64_t); // Compressed Size (now 8 bytes)
        headerEntrySize += sizeof(uint64_t); // Offset
        
        // Add size for tensor metadata if present
        if (header.tensor_metadata.has_value()) {
            headerEntrySize += sizeof(uint8_t); // Number of dimensions
            headerEntrySize += header.tensor_metadata->dimensions.size() * sizeof(uint32_t); // Each dimension
        }
        
        indexTableSize += headerEntrySize;
    }
    currentDataOffset += indexTableSize;

    // --- Write Index Table ---
    for (size_t i = 0; i < segments_.size(); ++i) {
        const auto& header = segments_[i].first;
        const auto& compressedData = segments_[i].second;

        uint64_t segmentDataOffset = currentDataOffset;
        dataOffsets.push_back(segmentDataOffset);
        currentDataOffset += compressedData.size();

        // Write segment header
        writeStringLocal(outputFile_, header.name);
        writeBasicTypeLocal(outputFile_, static_cast<uint8_t>(header.original_type));
        writeBasicTypeLocal(outputFile_, header.compression_strategy_id);
        writeBasicTypeLocal(outputFile_, static_cast<uint64_t>(header.original_size));
        writeBasicTypeLocal(outputFile_, static_cast<uint64_t>(compressedData.size()));
        writeBasicTypeLocal(outputFile_, segmentDataOffset);

        // Write tensor metadata presence flag
        uint8_t has_metadata = header.tensor_metadata.has_value() ? 1 : 0;
        writeBasicTypeLocal(outputFile_, has_metadata);
        if (has_metadata) {
            const auto& meta = header.tensor_metadata.value();
            uint8_t num_dims = static_cast<uint8_t>(meta.dimensions.size());
            writeBasicTypeLocal(outputFile_, num_dims);
            for (size_t dim : meta.dimensions) {
                writeBasicTypeLocal(outputFile_, static_cast<uint32_t>(dim));
            }
        }
    }

    // --- Write Compressed Data Blocks ---
    for (const auto& segmentPair : segments_) {
        const auto& compressedData = segmentPair.second;
        outputFile_.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());
    }

    // --- Finalize ---
    outputFile_.flush();
    if (!outputFile_) {
         outputFile_.close();
         throw std::runtime_error("Failed to write data during archive finalization.");
    }
    outputFile_.close();

    std::cout << "Archive finalized successfully. Total Segments: " << numSegments << std::endl;
}


// getCompressionRatio is removed as stats are only final after finalizeArchive

} // namespace CortexAICompression
