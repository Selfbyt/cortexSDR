/**
 * @file AIDecompressor.cpp
 * @brief Implementation of AI model decompression with on-demand loading
 * 
 * This file implements the AIDecompressor class which provides efficient
 * neural network model decompression supporting various compression strategies.
 * Designed for on-demand loading to minimize memory usage during inference.
 * 
 * Key Features:
 * - Multi-strategy decompression (SDR, RLE, Gzip, Quantization)
 * - On-demand segment loading for memory efficiency
 * - Stream-based decompression for large models  
 * - Header-only reading for metadata access
 * - Parallel decompression support for performance
 */

#include "AIDecompressor.hpp"
#include "ArchiveConstants.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <fstream>

namespace CortexAICompression {

/**
 * @brief Read binary data of basic types from input stream
 * @tparam T Type of data to read (must be trivially copyable)
 * @param stream Input stream to read from
 * @param value Reference to store the read value
 * @return True if read successful, false otherwise
 */
template<typename T>
bool readBasicType(std::istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return stream.good();
}

/**
 * @brief Read length-prefixed string from input stream
 * @param stream Input stream to read from
 * @param str Reference to string to populate
 * @return True if read successful, false otherwise
 */
bool readString(std::istream& stream, std::string& str) {
    uint32_t len;
    if (!readBasicType(stream, len)) {
        return false;
    }
    if (len > 0) { // Avoid reading 0 bytes if length is 0
        str.resize(len);
        stream.read(&str[0], len);
        if (!stream.good()) {
            return false;
        }
    } else {
        str.clear(); // Ensure string is empty if length is 0
    }
    return stream.good();
}


AIDecompressor::AIDecompressor() {
    // Constructor - strategies are registered via registerStrategy
}

void AIDecompressor::registerStrategy(uint8_t strategy_id, std::shared_ptr<ICompressionStrategy> strategy) {
    if (!strategy) {
        throw std::invalid_argument("Strategy cannot be null.");
    }
    if (strategy_id == 0) {
        throw std::invalid_argument("Strategy ID 0 is reserved for 'uncompressed'.");
    }
    strategyMap_[strategy_id] = std::move(strategy);
}

std::vector<CompressedSegmentHeader> AIDecompressor::readArchiveIndex(std::istream& stream) {
    // 1. Read and Verify Magic Number & Version
    char magic[8];
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, ARCHIVE_MAGIC, sizeof(ARCHIVE_MAGIC)) != 0) {
        throw CompressionError("Invalid or missing archive magic number.");
    }

    uint32_t version;
    if (!readBasicType(stream, version) || version != ARCHIVE_VERSION) {
         throw CompressionError("Unsupported archive version.");
    }

    // 2. Read Number of Segments
    uint64_t numSegments;
    if (!readBasicType(stream, numSegments)) {
        throw CompressionError("Failed to read segment count.");
    }
    uint64_t index_offset;
    if (!readBasicType(stream, index_offset)) {
        throw CompressionError("Failed to read index offset.");
    }
    stream.seekg(index_offset, std::ios::beg);

    // 3. Read Index Table Entries
    std::vector<CompressedSegmentHeader> headers;
    headers.reserve(numSegments);
    for (uint64_t i = 0; i < numSegments; ++i) {
        CompressedSegmentHeader header;
        uint8_t type_val;
        uint64_t offset; // Read offset, store it temporarily


        if (!readString(stream, header.name)) {
             throw CompressionError("Failed to read segment header information from index (name).");
        }

        if (!readString(stream, header.layer_type)) {
            throw CompressionError("Failed to read segment header information from index (layer_type).");
        }

        if (!readBasicType(stream, type_val)) {
             throw CompressionError("Failed to read segment header information from index (type).");
        }
        header.original_type = static_cast<SegmentType>(type_val);

        if (!readBasicType(stream, header.compression_strategy_id)) {
             throw CompressionError("Failed to read segment header information from index (strategy ID).");
        }

        if (!readBasicType(stream, header.original_size)) {
             throw CompressionError("Failed to read segment header information from index (original size).");
        }

        if (!readBasicType(stream, header.compressed_size)) {
             throw CompressionError("Failed to read segment header information from index (compressed size).");
        }

        if (!readBasicType(stream, offset)) // Read offset
        {
             throw CompressionError("Failed to read segment header information from index (offset).");
        }
        header.data_offset = offset;

        // Read layer_name
        if (!readString(stream, header.layer_name)) {
            throw CompressionError("Failed to read segment header information from index (layer_name).");
        }

        // Read layer_index (as uint32_t)
        uint32_t layer_idx_u32;
        if (!readBasicType(stream, layer_idx_u32)) {
            throw CompressionError("Failed to read segment header information from index (layer_index).");
        }
        header.layer_index = static_cast<size_t>(layer_idx_u32);

        // Read tensor_metadata
        bool has_metadata;
        if (!readBasicType(stream, has_metadata)) {
            throw CompressionError("Failed to read segment header information from index (has_metadata flag).");
        }

        if (has_metadata) {
            header.tensor_metadata.emplace(); // Create the TensorMetadata object
            TensorMetadata& meta = header.tensor_metadata.value();

            uint8_t num_dims;
            if (!readBasicType(stream, num_dims)) {
                throw CompressionError("Failed to read tensor metadata (num_dims).");
            }
            meta.dimensions.resize(num_dims);
            for (uint8_t d = 0; d < num_dims; ++d) {
                uint32_t dim_u32;
                if (!readBasicType(stream, dim_u32)) {
                    throw CompressionError("Failed to read tensor metadata (dimension value).");
                }
                meta.dimensions[d] = static_cast<size_t>(dim_u32);
            }

            if (!readBasicType(stream, meta.sparsity_ratio)) {
                throw CompressionError("Failed to read tensor metadata (sparsity_ratio).");
            }

            if (!readBasicType(stream, meta.is_sorted)) {
                throw CompressionError("Failed to read tensor metadata (is_sorted).");
            }

            bool has_scale;
            if (!readBasicType(stream, has_scale)) {
                throw CompressionError("Failed to read tensor metadata (has_scale flag).");
            }
            if (has_scale) {
                float scale_val;
                if (!readBasicType(stream, scale_val)) {
                    throw CompressionError("Failed to read tensor metadata (scale value).");
                }
                meta.scale = scale_val;
            }

            bool has_zero_point;
            if (!readBasicType(stream, has_zero_point)) {
                throw CompressionError("Failed to read tensor metadata (has_zero_point flag).");
            }
            if (has_zero_point) {
                float zp_val;
                if (!readBasicType(stream, zp_val)) {
                    throw CompressionError("Failed to read tensor metadata (zero_point value).");
                }
                meta.zero_point = zp_val;
            }
        }
        // Read input_shape
        uint8_t has_input_shape = 0;
        if (!readBasicType(stream, has_input_shape)) {
            throw CompressionError("Failed to read input_shape presence flag.");
        }
        if (has_input_shape) {
            uint8_t num_in = 0;
            if (!readBasicType(stream, num_in)) {
                throw CompressionError("Failed to read input_shape num dims.");
            }
            header.input_shape.resize(num_in);
            for (uint8_t d = 0; d < num_in; ++d) {
                uint32_t dim = 0;
                if (!readBasicType(stream, dim)) {
                    throw CompressionError("Failed to read input_shape dimension value.");
                }
                header.input_shape[d] = dim;
            }
        }
        // Read output_shape
        uint8_t has_output_shape = 0;
        if (!readBasicType(stream, has_output_shape)) {
            throw CompressionError("Failed to read output_shape presence flag.");
        }
        if (has_output_shape) {
            uint8_t num_out = 0;
            if (!readBasicType(stream, num_out)) {
                throw CompressionError("Failed to read output_shape num dims.");
            }
            header.output_shape.resize(num_out);
            for (uint8_t d = 0; d < num_out; ++d) {
                uint32_t dim = 0;
                if (!readBasicType(stream, dim)) {
                    throw CompressionError("Failed to read output_shape dimension value.");
                }
                header.output_shape[d] = dim;
            }
        }
        headers.push_back(header);
    }
    return headers;
}


ModelSegment AIDecompressor::readAndDecompressSegment(std::istream& stream, const CompressedSegmentHeader& header) {
    // 1. Read Compressed Data
    std::vector<std::byte> compressedData(header.compressed_size);
    stream.read(reinterpret_cast<char*>(compressedData.data()), header.compressed_size);
    if (!stream) {
        throw CompressionError("Failed to read compressed data for segment: " + header.name);
    }

    // 2. Select Decompression Strategy
    std::vector<std::byte> decompressedData;
    if (header.compression_strategy_id == 0) {
        // Strategy ID 0 means uncompressed
        std::cerr << "  Using uncompressed strategy for segment '" << header.name << "'" << std::endl;
        decompressedData = std::move(compressedData);
    } else {
        auto it = strategyMap_.find(header.compression_strategy_id);
        if (it == strategyMap_.end() || !it->second) {
            std::cerr << "  Error: Unknown strategy ID " << static_cast<int>(header.compression_strategy_id) 
                      << " for segment '" << header.name << "'" << std::endl;
            throw CompressionError("Unknown or unregistered compression strategy ID encountered: " + std::to_string(header.compression_strategy_id));
        }
        std::shared_ptr<ICompressionStrategy> strategy = it->second;
        std::cerr << "  Using strategy ID " << static_cast<int>(header.compression_strategy_id) 
                  << " for segment '" << header.name << "' of type " << static_cast<int>(header.original_type) << std::endl;

        // 3. Decompress
        try {
            // Print first few bytes of compressed data for debugging
            
            decompressedData = strategy->decompress(compressedData, header.original_type, header.original_size);
            
        } catch (const CompressionError& e) {
            std::cerr << "  Decompression error: " << e.what() << std::endl;
            throw CompressionError("Decompression failed for segment '" + header.name + "': " + e.what());
        }
    }

    // 4. Verify Size (Optional but recommended)
    if (decompressedData.size() != header.original_size) {
         std::cerr << "Warning: Decompressed size (" << decompressedData.size()
                   << ") does not match expected original size (" << header.original_size
                   << ") for segment '" << header.name << "'." << std::endl;
         // Depending on strictness, could throw here:
         // throw CompressionError("Decompressed size mismatch for segment: " + header.name);
    }

    // 5. Create ModelSegment
    ModelSegment segment;
    segment.name = header.name;
    segment.type = header.original_type;
    segment.original_size = header.original_size; // Store original size
    segment.data = std::move(decompressedData);
    segment.tensor_metadata = header.tensor_metadata; // Copy tensor metadata
    segment.layer_name = header.layer_name;           // Copy layer name
    segment.layer_index = header.layer_index;         // Copy layer index
    segment.layer_type = header.layer_type;           // Copy layer type
    segment.input_shape = header.input_shape;         // Copy input shape
    segment.output_shape = header.output_shape;       // Copy output shape

    return segment;
}


void AIDecompressor::decompressModelStream(std::istream& inputArchiveStream, ISegmentHandler& handler) {
    if (!inputArchiveStream) {
        throw std::runtime_error("AIDecompressor: Input stream is invalid.");
    }

    // 1. Read Index
    std::vector<CompressedSegmentHeader> headers = readArchiveIndex(inputArchiveStream);

    // 2. Process Segments Sequentially
    for (const auto& header : headers) {
        inputArchiveStream.clear();
        inputArchiveStream.seekg(static_cast<std::streamoff>(header.data_offset), std::ios::beg);
        if (!inputArchiveStream) {
            throw CompressionError("Failed to seek to segment payload for: " + header.name);
        }

        ModelSegment segment = readAndDecompressSegment(inputArchiveStream, header);

        // 3. Pass to Handler
        handler.handleSegment(std::move(segment)); // Pass ownership

         if (!inputArchiveStream) {
             // Check stream state after reading/handling each segment
             throw CompressionError("Stream error occurred during segment processing.");
         }
    }

    // Optional: Check if we consumed the expected amount of data or if there's trailing data.
}

ModelSegment AIDecompressor::decompressSegment(const std::string& archivePath, const CompressedSegmentHeader& segmentInfo, uint64_t offset) {
    std::ifstream stream(archivePath, std::ios::binary);
    if (!stream) {
        throw CompressionError("Failed to open archive for on-demand segment loading: " + archivePath);
    }

    // Seek to the beginning of the compressed data for this segment
    stream.seekg(offset);
    if (!stream) {
        throw CompressionError("Failed to seek to offset for segment: " + segmentInfo.name);
    }

    // 1. Read Compressed Data
    std::vector<std::byte> compressedData(segmentInfo.compressed_size);
    stream.read(reinterpret_cast<char*>(compressedData.data()), segmentInfo.compressed_size);
    if (!stream) {
        throw CompressionError("Failed to read compressed data for segment: " + segmentInfo.name);
    }

    // 2. Select Decompression Strategy
    std::vector<std::byte> decompressedData;
    if (segmentInfo.compression_strategy_id == 0) {
        // Strategy ID 0 means uncompressed
        decompressedData = std::move(compressedData);
    } else {
        auto it = strategyMap_.find(segmentInfo.compression_strategy_id);
        if (it == strategyMap_.end() || !it->second) {
            throw CompressionError("Unknown or unregistered compression strategy ID: " + std::to_string(segmentInfo.compression_strategy_id));
        }
        std::shared_ptr<ICompressionStrategy> strategy = it->second;

        // 3. Decompress
        decompressedData = strategy->decompress(compressedData, segmentInfo.original_type, segmentInfo.original_size);
    }

    // 4. Verify Size
    if (decompressedData.size() != segmentInfo.original_size) {
        std::cerr << "Warning: Decompressed size (" << decompressedData.size()
                  << ") does not match expected original size (" << segmentInfo.original_size
                  << ") for on-demand segment '" << segmentInfo.name << "'." << std::endl;
    }

    // 5. Create ModelSegment
    ModelSegment segment;
    segment.name = segmentInfo.name;
    segment.type = segmentInfo.original_type;
    segment.original_size = segmentInfo.original_size;
    segment.data = std::move(decompressedData);
    segment.tensor_metadata = segmentInfo.tensor_metadata;
    segment.layer_name = segmentInfo.layer_name;
    segment.layer_index = segmentInfo.layer_index;
    segment.layer_type = segmentInfo.layer_type;
    segment.input_shape = segmentInfo.input_shape;
    segment.output_shape = segmentInfo.output_shape;

    return segment;
}

std::vector<std::byte> AIDecompressor::readCompressedBytes(const std::string& archivePath, const CompressedSegmentHeader& segmentInfo, uint64_t offset) {
    std::ifstream stream(archivePath, std::ios::binary);
    if (!stream) {
        throw CompressionError("Failed to open archive for raw segment read: " + archivePath);
    }

    stream.seekg(offset);
    if (!stream) {
        throw CompressionError("Failed to seek to offset for segment: " + segmentInfo.name);
    }

    std::vector<std::byte> compressedData(segmentInfo.compressed_size);
    stream.read(reinterpret_cast<char*>(compressedData.data()), segmentInfo.compressed_size);
    if (!stream) {
        throw CompressionError("Failed to read compressed data for segment: " + segmentInfo.name);
    }
    return compressedData;
}

std::vector<CompressedSegmentHeader> AIDecompressor::readArchiveHeaders(std::istream& inputArchiveStream) {
    return readArchiveIndex(inputArchiveStream);
}

} // namespace CortexAICompression
