#include "AIDecompressor.hpp"
#include "ArchiveConstants.hpp"
#include <vector>
#include <stdexcept>
#include <iostream> // For potential debug/error messages
#include <cstring> // For std::memcpy, std::memcmp

namespace CortexAICompression {

// Helper function to read basic types from the stream
template<typename T>
bool readBasicType(std::istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return stream.good();
}

// Helper function to read string (length prefixed)
bool readString(std::istream& stream, std::string& str) {
    uint32_t len;
    std::cerr << "  DEBUG readString: Attempting to read length (uint32_t)..." << std::endl; // DEBUG
    if (!readBasicType(stream, len)) {
        std::cerr << "  DEBUG readString: Failed to read length. Stream state good: " << stream.good() << ", eof: " << stream.eof() << ", fail: " << stream.fail() << ", bad: " << stream.bad() << std::endl; // DEBUG
        return false;
    }
    std::cerr << "  DEBUG readString: Read length = " << len << ". Attempting to read " << len << " bytes for string..." << std::endl; // DEBUG
    if (len > 0) { // Avoid reading 0 bytes if length is 0
        str.resize(len);
        stream.read(&str[0], len);
        if (!stream.good()) {
             std::cerr << "  DEBUG readString: Failed to read string data. Stream state good: " << stream.good() << ", eof: " << stream.eof() << ", fail: " << stream.fail() << ", bad: " << stream.bad() << std::endl; // DEBUG
             return false;
        }
    } else {
        str.clear(); // Ensure string is empty if length is 0
    }
    std::cerr << "  DEBUG readString: Successfully read string." << std::endl; // DEBUG
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
    uint32_t numSegments;
    if (!readBasicType(stream, numSegments)) {
        throw CompressionError("Failed to read segment count.");
    }

    // 3. Read Index Table Entries
    std::vector<CompressedSegmentHeader> headers;
    headers.reserve(numSegments);
    std::cerr << "DEBUG: Expecting " << numSegments << " segment headers." << std::endl; // DEBUG
    for (uint32_t i = 0; i < numSegments; ++i) {
        CompressedSegmentHeader header;
        uint8_t type_val;
        uint64_t offset; // Read offset, store it temporarily

        std::cerr << "DEBUG: Reading header for segment " << i << "..." << std::endl; // DEBUG

        if (!readString(stream, header.name)) {
             std::cerr << "DEBUG: Failed reading name for segment " << i << std::endl; // DEBUG
             throw CompressionError("Failed to read segment header information from index (name).");
        }
         std::cerr << "  DEBUG Name: " << header.name << " (Length: " << header.name.length() << ")" << std::endl; // DEBUG

        if (!readBasicType(stream, type_val)) {
             std::cerr << "DEBUG: Failed reading type for segment " << i << std::endl; // DEBUG
             throw CompressionError("Failed to read segment header information from index (type).");
        }
        header.original_type = static_cast<SegmentType>(type_val);
         std::cerr << "  DEBUG Type: " << static_cast<int>(header.original_type) << std::endl; // DEBUG

        if (!readBasicType(stream, header.compression_strategy_id)) {
             std::cerr << "DEBUG: Failed reading strategy ID for segment " << i << std::endl; // DEBUG
             throw CompressionError("Failed to read segment header information from index (strategy ID).");
        }
          std::cerr << "  DEBUG Strategy ID: " << static_cast<int>(header.compression_strategy_id) << std::endl; // DEBUG

        if (!readBasicType(stream, header.original_size)) {
             std::cerr << "DEBUG: Failed reading original size for segment " << i << std::endl; // DEBUG
             throw CompressionError("Failed to read segment header information from index (original size).");
        }
         std::cerr << "  DEBUG Original Size: " << header.original_size << std::endl; // DEBUG

        if (!readBasicType(stream, header.compressed_size)) {
             std::cerr << "DEBUG: Failed reading compressed size for segment " << i << std::endl; // DEBUG
             throw CompressionError("Failed to read segment header information from index (compressed size).");
        }
         std::cerr << "  DEBUG Compressed Size: " << header.compressed_size << std::endl; // DEBUG

        if (!readBasicType(stream, offset)) // Read offset
        {
             std::cerr << "DEBUG: Failed reading offset for segment " << i << std::endl; // DEBUG
             throw CompressionError("Failed to read segment header information from index (offset).");
        }
        std::cerr << "  DEBUG Offset: " << offset << std::endl; // DEBUG
        // Note: The 'offset' read here is part of the header in the file,
        // but CompressedSegmentHeader struct itself doesn't store it as it's implicit in sequential read.

        // Read layer_name
        if (!readString(stream, header.layer_name)) {
            throw CompressionError("Failed to read segment header information from index (layer_name).");
        }
        std::cerr << "  DEBUG Layer Name: " << header.layer_name << std::endl;

        // Read layer_index (as uint32_t)
        uint32_t layer_idx_u32;
        if (!readBasicType(stream, layer_idx_u32)) {
            throw CompressionError("Failed to read segment header information from index (layer_index).");
        }
        header.layer_index = static_cast<size_t>(layer_idx_u32);
        std::cerr << "  DEBUG Layer Index: " << header.layer_index << std::endl;

        // Read tensor_metadata
        bool has_metadata;
        if (!readBasicType(stream, has_metadata)) {
            throw CompressionError("Failed to read segment header information from index (has_metadata flag).");
        }
        std::cerr << "  DEBUG Has Metadata: " << (has_metadata ? "true" : "false") << std::endl;

        if (has_metadata) {
            header.tensor_metadata.emplace(); // Create the TensorMetadata object
            TensorMetadata& meta = header.tensor_metadata.value();

            uint8_t num_dims;
            if (!readBasicType(stream, num_dims)) {
                throw CompressionError("Failed to read tensor metadata (num_dims).");
            }
            std::cerr << "    DEBUG Num Dims: " << static_cast<int>(num_dims) << std::endl;
            meta.dimensions.resize(num_dims);
            for (uint8_t d = 0; d < num_dims; ++d) {
                if (!readBasicType(stream, meta.dimensions[d])) {
                    throw CompressionError("Failed to read tensor metadata (dimension value).");
                }
                std::cerr << "      DEBUG Dim " << static_cast<int>(d) << ": " << meta.dimensions[d] << std::endl;
            }

            if (!readBasicType(stream, meta.sparsity_ratio)) {
                throw CompressionError("Failed to read tensor metadata (sparsity_ratio).");
            }
            std::cerr << "    DEBUG Sparsity Ratio: " << meta.sparsity_ratio << std::endl;

            if (!readBasicType(stream, meta.is_sorted)) {
                throw CompressionError("Failed to read tensor metadata (is_sorted).");
            }
            std::cerr << "    DEBUG Is Sorted: " << (meta.is_sorted ? "true" : "false") << std::endl;

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
                std::cerr << "    DEBUG Scale: " << scale_val << std::endl;
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
                std::cerr << "    DEBUG Zero Point: " << zp_val << std::endl;
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
            std::cerr << "  First 10 bytes of compressed data: ";
            for (size_t i = 0; i < std::min(size_t(10), compressedData.size()); i++) {
                std::cerr << std::hex << static_cast<int>(compressedData[i]) << " ";
            }
            std::cerr << std::dec << std::endl;
            
            decompressedData = strategy->decompress(compressedData, header.original_type, header.original_size);
            
            // Print first few bytes of decompressed data for debugging
            std::cerr << "  First 10 bytes of decompressed data: ";
            for (size_t i = 0; i < std::min(size_t(10), decompressedData.size()); i++) {
                std::cerr << std::hex << static_cast<int>(decompressedData[i]) << " ";
            }
            std::cerr << std::dec << std::endl;
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

    return segment;
}


void AIDecompressor::decompressModelStream(std::istream& inputArchiveStream, ISegmentHandler& handler) {
    if (!inputArchiveStream) {
        throw std::runtime_error("AIDecompressor: Input stream is invalid.");
    }

    // Store the starting position to calculate data offsets relative to it
    std::streampos indexStartPos = inputArchiveStream.tellg();

    // 1. Read Index
    std::vector<CompressedSegmentHeader> headers = readArchiveIndex(inputArchiveStream);

    // The index reading finished here. The stream position is now at the start of the first data block.
    std::streampos dataStartPos = inputArchiveStream.tellg();

    // 2. Process Segments Sequentially
    for (const auto& header : headers) {
        // Note: The archive format defined in AICompressor writes data sequentially *after* the index.
        // So, we just read the next block of compressed_size bytes.
        // If the format stored absolute offsets, we would use inputArchiveStream.seekg() here.

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

} // namespace CortexAICompression
