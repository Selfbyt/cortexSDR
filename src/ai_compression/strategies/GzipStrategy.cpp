#include "GzipStrategy.hpp"
#include <zlib.h>
#include <vector>
#include <stdexcept> // For CompressionError
#include <iostream> // For potential debug output

namespace CortexAICompression {

// --- Compression ---
std::vector<std::byte> GzipStrategy::compress(const ModelSegment& segment) const {
    if (segment.data.empty()) {
        return {}; // Nothing to compress
    }

    if (segment.type == SegmentType::GRAPH_STRUCTURE_PROTO) {
        std::cerr << "GzipStrategy::compress - GRAPH_STRUCTURE_PROTO input data (first 20 bytes of " << segment.data.size() << "): ";
        for (size_t i = 0; i < std::min(size_t(20), segment.data.size()); ++i) {
            std::cerr << std::hex << static_cast<int>(segment.data[i]) << " ";
        }
        std::cerr << std::dec << std::endl;
    }

    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;

    // Initialize deflate with gzip encoding (windowBits = 15 + 16)
    // Use the member variable compression_level_
    int ret = deflateInit2(&strm, compression_level_, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
    if (ret != Z_OK) {
        throw CompressionError("zlib deflateInit failed: " + std::to_string(ret));
    }

    std::vector<std::byte> compressedData;
    std::vector<std::byte> outBuffer(CHUNK_SIZE);

    strm.avail_in = static_cast<uInt>(segment.data.size());
    strm.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(segment.data.data()));

    // Compress loop
    do {
        strm.avail_out = static_cast<uInt>(CHUNK_SIZE);
        strm.next_out = reinterpret_cast<Bytef*>(outBuffer.data());
        ret = deflate(&strm, Z_FINISH); // Z_FINISH indicates this is the last chunk

        if (ret == Z_STREAM_ERROR) {
            deflateEnd(&strm);
            throw CompressionError("zlib deflate failed: Z_STREAM_ERROR");
        }

        size_t have = CHUNK_SIZE - strm.avail_out;
        compressedData.insert(compressedData.end(), outBuffer.begin(), outBuffer.begin() + have);

    } while (strm.avail_out == 0); // Continue if the output buffer was filled

    // Check for complete compression
    if (ret != Z_STREAM_END) {
        deflateEnd(&strm);
        throw CompressionError("zlib deflate failed to produce Z_STREAM_END: " + std::to_string(ret));
    }

    // Clean up
    deflateEnd(&strm);

    return compressedData;
}

// --- Decompression ---
std::vector<std::byte> GzipStrategy::decompress(const std::vector<std::byte>& compressedData, SegmentType originalType, size_t originalSize) const {
    // originalType is ignored for Gzip, but originalSize can be a hint (though zlib handles dynamic output)
    if (compressedData.empty()) {
        std::cerr << "GzipStrategy::decompress - Empty compressed data" << std::endl;
        return {};
    }
    
    // Special handling for GRAPH_STRUCTURE_PROTO type
    if (originalType == SegmentType::GRAPH_STRUCTURE_PROTO) {
        std::cerr << "GzipStrategy::decompress - Special handling for GRAPH_STRUCTURE_PROTO" << std::endl;
        
        // Check if the data has the gzip magic number
        if (compressedData.size() >= 2 && 
            static_cast<uint8_t>(compressedData[0]) == 0x1F && 
            static_cast<uint8_t>(compressedData[1]) == 0x8B) {
            std::cerr << "  Valid gzip header detected" << std::endl;
        } else {
            std::cerr << "  Warning: No valid gzip header detected" << std::endl;
        }
    }

    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;

    // Initialize inflate to handle gzip format (windowBits = 15 + 16)
    int ret = inflateInit2(&strm, 15 + 16);
    if (ret != Z_OK) {
        std::cerr << "GzipStrategy::decompress - inflateInit2 failed: " << ret << std::endl;
        throw CompressionError("zlib inflateInit failed: " + std::to_string(ret));
    }

    std::vector<std::byte> decompressedData;
    // Reserve space if originalSize is known and seems reasonable, otherwise let it grow
    if (originalSize > 0 && originalSize < 1024*1024*1024) { // Avoid huge pre-allocations
         decompressedData.reserve(originalSize);
    }
    std::vector<std::byte> outBuffer(CHUNK_SIZE);

    strm.avail_in = static_cast<uInt>(compressedData.size());
    strm.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(compressedData.data()));

    // Decompress loop
    do {
        strm.avail_out = static_cast<uInt>(CHUNK_SIZE);
        strm.next_out = reinterpret_cast<Bytef*>(outBuffer.data());
        ret = inflate(&strm, Z_NO_FLUSH);
        
        std::cerr << "  inflate returned: " << ret << " (" << 
                     (ret == Z_OK ? "Z_OK" : 
                     ret == Z_STREAM_END ? "Z_STREAM_END" : 
                     ret == Z_STREAM_ERROR ? "Z_STREAM_ERROR" : 
                     ret == Z_DATA_ERROR ? "Z_DATA_ERROR" : 
                     ret == Z_MEM_ERROR ? "Z_MEM_ERROR" : 
                     ret == Z_BUF_ERROR ? "Z_BUF_ERROR" : "unknown") << ")" << std::endl;

        switch (ret) {
            case Z_STREAM_ERROR:
                inflateEnd(&strm);
                std::cerr << "GzipStrategy::decompress - Z_STREAM_ERROR" << std::endl;
                throw CompressionError("zlib inflate failed: Z_STREAM_ERROR");
            case Z_NEED_DICT:
                // This shouldn't happen with standard gzip
                std::cerr << "GzipStrategy::decompress - Z_NEED_DICT" << std::endl;
                ret = Z_DATA_ERROR; // Treat as data error
                [[fallthrough]];
            case Z_DATA_ERROR:
                std::cerr << "GzipStrategy::decompress - Z_DATA_ERROR" << std::endl;
                inflateEnd(&strm);
                throw CompressionError("zlib inflate failed: Z_DATA_ERROR");
            case Z_MEM_ERROR:
                std::cerr << "GzipStrategy::decompress - Z_MEM_ERROR" << std::endl;
                inflateEnd(&strm);
                throw CompressionError("zlib inflate failed: Z_MEM_ERROR");
            case Z_BUF_ERROR:
                // This is not always fatal - it can mean we need more input or output space
                std::cerr << "GzipStrategy::decompress - Z_BUF_ERROR (may be recoverable)" << std::endl;
                break;
        }

        size_t have = CHUNK_SIZE - strm.avail_out;
        std::cerr << "  Decompressed " << have << " bytes this iteration" << std::endl;
        decompressedData.insert(decompressedData.end(), outBuffer.begin(), outBuffer.begin() + have);

    } while (strm.avail_out == 0); // Continue if output buffer was full

    // Check for clean end of stream
    if (ret != Z_STREAM_END) {
         // It might just mean more input is needed if we were streaming, but here we expect full data.
         // However, sometimes inflate returns Z_OK if the buffer ends exactly. Let's allow Z_OK too.
         if (ret != Z_OK) {
            std::cerr << "GzipStrategy::decompress - Failed to finish cleanly: " << ret << std::endl;
            inflateEnd(&strm);
            throw CompressionError("zlib inflate failed to finish cleanly: " + std::to_string(ret));
         }
    }

    // Clean up
    inflateEnd(&strm);

    // Optional: Final check against originalSize if provided
    if (originalSize > 0 && decompressedData.size() != originalSize) {
         std::cerr << "Warning: Gzip decompressed size (" << decompressedData.size()
                   << ") does not match expected original size (" << originalSize << ")." << std::endl;
         // Could throw here if strict size matching is required.
    }
    
    // Special handling for empty or suspicious results
    if (decompressedData.empty()) {
        std::cerr << "Warning: Decompression produced empty result!" << std::endl;
        // For GRAPH_STRUCTURE_PROTO, if decompression failed, let the (empty) result propagate
        if (originalType == SegmentType::GRAPH_STRUCTURE_PROTO) {
             std::cerr << "Warning: Decompression for GRAPH_STRUCTURE_PROTO resulted in empty data." << std::endl;
             // Let the empty vector be returned below
        }
    } else if (decompressedData.size() > 0) {
        bool allZeros = true;
        for (size_t i = 0; i < std::min(size_t(20), decompressedData.size()); i++) {
            if (static_cast<uint8_t>(decompressedData[i]) != 0) {
                allZeros = false;
                break;
            }
        }
        
        if (allZeros) {
            std::cerr << "Warning: Decompression produced all zeros in first 20 bytes!" << std::endl;
            // For GRAPH_STRUCTURE_PROTO, if decompression produced all zeros, let the result propagate
            if (originalType == SegmentType::GRAPH_STRUCTURE_PROTO) {
                 std::cerr << "Warning: Decompression for GRAPH_STRUCTURE_PROTO resulted in all zeros." << std::endl;
                 // Let the zero-filled vector be returned below
            }
        }
    }

    return decompressedData;
}

} // namespace CortexAICompression
