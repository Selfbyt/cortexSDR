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
    uint16_t len = static_cast<uint16_t>(str.length());
    if (str.length() > UINT16_MAX) {
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
        // Nothing to write or file not open
        if (segments_.empty()) {
             std::cerr << "Warning: Finalizing archive with zero segments." << std::endl;
        }
        // Ensure file is closed if it was opened but no segments were added
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
        headerEntrySize += sizeof(uint8_t); // Original Type
        headerEntrySize += sizeof(uint8_t); // Strategy ID
        headerEntrySize += sizeof(uint64_t); // Original Size
        headerEntrySize += sizeof(uint64_t); // Compressed Size
        headerEntrySize += sizeof(uint64_t); // Offset field itself
        
        // Add sizes for layer info
        headerEntrySize += sizeof(uint16_t) + header.layer_name.length(); // Layer name length + name
        headerEntrySize += sizeof(uint32_t); // Layer index (using uint32_t for layer_index)

        // Add size for tensor_metadata
        headerEntrySize += sizeof(bool); // hasTensorMetadata flag
        if (header.tensor_metadata.has_value()) {
            headerEntrySize += sizeof(uint8_t); // Number of dimensions
            headerEntrySize += header.tensor_metadata->dimensions.size() * sizeof(size_t); // Each dimension
            // Add other TensorMetadata fields if serialized (e.g., scale, zero_point)
            headerEntrySize += sizeof(float); // sparsity_ratio
            headerEntrySize += sizeof(bool);  // is_sorted
            headerEntrySize += sizeof(bool);  // scale.has_value()
            if (header.tensor_metadata->scale.has_value()) {
                headerEntrySize += sizeof(float); // scale value
            }
            headerEntrySize += sizeof(bool);  // zero_point.has_value()
            if (header.tensor_metadata->zero_point.has_value()) {
                headerEntrySize += sizeof(float); // zero_point value
            }
        }
        indexTableSize += headerEntrySize;
    }
    currentDataOffset += indexTableSize; // Start of first data block is after the full index

    // --- Write Index Table ---
    for (size_t i = 0; i < segments_.size(); ++i) {
        const auto& header = segments_[i].first;
        const auto& compressedData = segments_[i].second; // Needed for compressed_size

        uint64_t segmentDataOffset = currentDataOffset; // Offset for *this* segment's data
        dataOffsets.push_back(segmentDataOffset);
        currentDataOffset += compressedData.size(); // Update offset for the *next* segment's data

        // Write header fields to the index table
        writeStringLocal(outputFile_, header.name);
        uint8_t type_val = static_cast<uint8_t>(header.original_type);
        writeBasicTypeLocal(outputFile_, type_val);
        writeBasicTypeLocal(outputFile_, header.compression_strategy_id);
        writeBasicTypeLocal(outputFile_, header.original_size);
        writeBasicTypeLocal(outputFile_, static_cast<uint64_t>(compressedData.size())); // Use actual data size
        writeBasicTypeLocal(outputFile_, segmentDataOffset); // Write calculated offset

        // Write layer info
        writeStringLocal(outputFile_, header.layer_name);
        writeBasicTypeLocal(outputFile_, static_cast<uint32_t>(header.layer_index));

        // Write tensor_metadata
        bool has_metadata = header.tensor_metadata.has_value();
        writeBasicTypeLocal(outputFile_, has_metadata);
        if (has_metadata) {
            const auto& meta = header.tensor_metadata.value();
            uint8_t num_dims = static_cast<uint8_t>(meta.dimensions.size());
            writeBasicTypeLocal(outputFile_, num_dims);
            for (size_t dim : meta.dimensions) {
                writeBasicTypeLocal(outputFile_, dim);
            }
            writeBasicTypeLocal(outputFile_, meta.sparsity_ratio);
            writeBasicTypeLocal(outputFile_, meta.is_sorted);
            
            bool has_scale = meta.scale.has_value();
            writeBasicTypeLocal(outputFile_, has_scale);
            if (has_scale) {
                writeBasicTypeLocal(outputFile_, meta.scale.value());
            }
            
            bool has_zero_point = meta.zero_point.has_value();
            writeBasicTypeLocal(outputFile_, has_zero_point);
            if (has_zero_point) {
                writeBasicTypeLocal(outputFile_, meta.zero_point.value());
            }
        }
    }

    // --- Write Compressed Data Blocks ---
    for (const auto& segmentPair : segments_) {
        const auto& compressedData = segmentPair.second;
        outputFile_.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());
    }

    // --- Finalize ---
    outputFile_.flush(); // Ensure all data is written
    if (!outputFile_) {
         // Close the file before throwing to release the handle
         outputFile_.close();
         throw std::runtime_error("Failed to write data during archive finalization.");
    }
    outputFile_.close(); // Close the file explicitly after successful writing

    std::cout << "Archive finalized successfully. Total Segments: " << numSegments << std::endl;
}


// getCompressionRatio is removed as stats are only final after finalizeArchive

} // namespace CortexAICompression
