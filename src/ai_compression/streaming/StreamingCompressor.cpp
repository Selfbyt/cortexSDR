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
        throw std::runtime_error("Segment name or layer name too long for archive format.");
    }
    writeBasicTypeLocal(stream, len);
    stream.write(str.data(), len);
}


StreamingCompressor::StreamingCompressor(const std::string& outputPath)
    : outputFile_(outputPath, std::ios::binary | std::ios::trunc)
{
    if (!outputFile_) {
        throw std::runtime_error("Failed to open output file for writing: " + outputPath);
    }
    // Write archive header placeholders
    outputFile_.write(ARCHIVE_MAGIC, sizeof(ARCHIVE_MAGIC));
    writeBasicTypeLocal(outputFile_, ARCHIVE_VERSION);

    // Placeholders for segment count and index offset
    uint64_t placeholder_count = 0;
    uint64_t placeholder_offset = 0;
    writeBasicTypeLocal(outputFile_, placeholder_count);
    writeBasicTypeLocal(outputFile_, placeholder_offset);

    // The rest of the file is segment data
    currentDataOffset_ = outputFile_.tellp();
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

    // Get the offset for the current segment
    uint64_t segmentOffset = currentDataOffset_;

    // Write compressed data immediately
    outputFile_.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());
    if (!outputFile_) {
        throw std::runtime_error("Failed to write segment data to file.");
    }

    // Update the offset for the next segment
    currentDataOffset_ += compressedData.size();

    // Store the segment header and its offset (not the data)
    segment_headers_.emplace_back(header, segmentOffset);

    // Update statistics
    totalCompressedSize_ += compressedData.size();
    totalOriginalSize_ += header.original_size;
}

void StreamingCompressor::finalizeArchive() {
    if (!outputFile_.is_open()) {
        return; // File already closed or never opened
    }

    if (segment_headers_.empty()) {
        std::cerr << "Warning: Finalizing archive with zero segments." << std::endl;
        outputFile_.close();
        return;
    }

    // --- Get Index Table Offset ---
    uint64_t index_table_offset = outputFile_.tellp();

    // --- Write Index Table ---
    for (const auto& pair : segment_headers_) {
        const auto& header = pair.first;
        uint64_t segmentDataOffset = pair.second;

        // Write segment header to the index
        writeStringLocal(outputFile_, header.name);
        writeStringLocal(outputFile_, header.layer_type);
        writeBasicTypeLocal(outputFile_, static_cast<uint8_t>(header.original_type));
        writeBasicTypeLocal(outputFile_, header.compression_strategy_id);
        writeBasicTypeLocal(outputFile_, static_cast<uint64_t>(header.original_size));
        writeBasicTypeLocal(outputFile_, static_cast<uint64_t>(header.compressed_size));
        writeBasicTypeLocal(outputFile_, segmentDataOffset);

        // Write layer_name and layer_index (fix for archive format)
        writeStringLocal(outputFile_, header.layer_name);
        writeBasicTypeLocal(outputFile_, static_cast<uint32_t>(header.layer_index));

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
            // Write additional tensor metadata fields
            writeBasicTypeLocal(outputFile_, meta.sparsity_ratio);
            writeBasicTypeLocal(outputFile_, meta.is_sorted);
            uint8_t has_scale = meta.scale.has_value() ? 1 : 0;
            writeBasicTypeLocal(outputFile_, has_scale);
            if (has_scale) writeBasicTypeLocal(outputFile_, meta.scale.value());
            uint8_t has_zero_point = meta.zero_point.has_value() ? 1 : 0;
            writeBasicTypeLocal(outputFile_, has_zero_point);
            if (has_zero_point) writeBasicTypeLocal(outputFile_, meta.zero_point.value());
        }
        // Write input_shape
        uint8_t has_input_shape = header.input_shape.empty() ? 0 : 1;
        writeBasicTypeLocal(outputFile_, has_input_shape);
        if (has_input_shape) {
            uint8_t num_in = static_cast<uint8_t>(header.input_shape.size());
            writeBasicTypeLocal(outputFile_, num_in);
            for (size_t dim : header.input_shape) {
                writeBasicTypeLocal(outputFile_, static_cast<uint32_t>(dim));
            }
        }
        // Write output_shape
        uint8_t has_output_shape = header.output_shape.empty() ? 0 : 1;
        writeBasicTypeLocal(outputFile_, has_output_shape);
        if (has_output_shape) {
            uint8_t num_out = static_cast<uint8_t>(header.output_shape.size());
            writeBasicTypeLocal(outputFile_, num_out);
            for (size_t dim : header.output_shape) {
                writeBasicTypeLocal(outputFile_, static_cast<uint32_t>(dim));
            }
        }
    }

    // --- Update Header with Final Values ---
    uint64_t num_segments = segment_headers_.size();
    
    // Seek back to the placeholder locations and write the correct values
    outputFile_.seekp(sizeof(ARCHIVE_MAGIC) + sizeof(ARCHIVE_VERSION), std::ios::beg);
    writeBasicTypeLocal(outputFile_, num_segments);
    writeBasicTypeLocal(outputFile_, index_table_offset);

    // --- Finalize ---
    outputFile_.flush();
    if (!outputFile_) {
         outputFile_.close();
         throw std::runtime_error("Failed to write data during archive finalization.");
    }
    outputFile_.close();

    std::cout << "Archive finalized successfully. Total Segments: " << num_segments << std::endl;
}


// getCompressionRatio is removed as stats are only final after finalizeArchive

} // namespace CortexAICompression
