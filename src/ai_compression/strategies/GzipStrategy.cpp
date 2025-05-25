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
    if (compressedData.empty()) {
        throw CompressionError("Empty compressed data");
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

        switch (ret) {
            case Z_STREAM_ERROR:
                inflateEnd(&strm);
                throw CompressionError("zlib inflate failed: Z_STREAM_ERROR");
            case Z_NEED_DICT:
                ret = Z_DATA_ERROR; // Treat as data error
                [[fallthrough]];
            case Z_DATA_ERROR:
                inflateEnd(&strm);
                throw CompressionError("zlib inflate failed: Z_DATA_ERROR");
            case Z_MEM_ERROR:
                inflateEnd(&strm);
                throw CompressionError("zlib inflate failed: Z_MEM_ERROR");
            case Z_BUF_ERROR:
                break;
        }

        size_t have = CHUNK_SIZE - strm.avail_out;
        decompressedData.insert(decompressedData.end(), outBuffer.begin(), outBuffer.begin() + have);

    } while (strm.avail_out == 0); // Continue if output buffer was full

    // Check for clean end of stream
    if (ret != Z_STREAM_END && ret != Z_OK) {
            inflateEnd(&strm);
            throw CompressionError("zlib inflate failed to finish cleanly: " + std::to_string(ret));
    }

    // Clean up
    inflateEnd(&strm);

    // Verify size if provided
    if (originalSize > 0 && decompressedData.size() != originalSize) {
        throw CompressionError("Decompressed size does not match expected original size");
    }
    
    if (decompressedData.empty()) {
        throw CompressionError("Decompression produced empty result");
    }

    return decompressedData;
}

} // namespace CortexAICompression
