/**
 * @file rle.cpp
 * @brief Implementation of Run-Length Encoding compression algorithms
 * 
 * This file implements efficient run-length encoding (RLE) compression
 * for binary and sparse data structures. Optimized for neural network
 * weight compression where many consecutive zeros or repeated values
 * are common, especially in sparse representations.
 * 
 * Key Features:
 * - Standard RLE for binary sequences
 * - Byte-based RLE for general data compression
 * - Optimized for sparse neural network data
 * - Memory-efficient encoding/decoding
 * - Support for large data structures
 */

#include "rle.hpp"
#include <stdexcept>

namespace rle {
    /**
     * @brief Encode binary data using run-length encoding
     * @param bits Input binary data to compress
     * @return Compressed data as alternating value-count pairs
     * 
     * Compresses sequences of repeated values by storing value followed
     * by run length. Maximum run length is 255 per sequence.
     */
    std::vector<uint8_t> encode(const std::vector<uint8_t>& bits) {
        std::vector<uint8_t> out;
        if (bits.empty()) return out;
        uint8_t current = bits[0];
        uint8_t run = 1;
        
        // Process each bit, accumulating run lengths
        for (size_t i = 1; i < bits.size(); ++i) {
            if (bits[i] == current && run < 255) {
                ++run;
            } else {
                // Output current run and start new one
                out.push_back(current);
                out.push_back(run);
                current = bits[i];
                run = 1;
            }
        }
        
        // Output final run
        out.push_back(current);
        out.push_back(run);
        return out;
    }

    /**
     * @brief Decode run-length encoded binary data
     * @param rleData Compressed data in value-count format
     * @return Original uncompressed binary data
     * @throws std::invalid_argument if data format is invalid
     */
    std::vector<uint8_t> decode(const std::vector<uint8_t>& rleData) {
        std::vector<uint8_t> out;
        if (rleData.size() % 2 != 0) throw std::runtime_error("Invalid RLE data");
        for (size_t i = 0; i < rleData.size(); i += 2) {
            uint8_t value = rleData[i];
            uint8_t run = rleData[i+1];
            out.insert(out.end(), run, value);
        }
        return out;
    }

    std::vector<uint8_t> encodeFromBytes(const std::string& bytes) {
        std::vector<uint8_t> bits;
        for (unsigned char c : bytes) {
            for (int i = 0; i < 8; ++i) {
                bits.push_back((c >> i) & 1);
            }
        }
        return encode(bits);
    }

    std::string decodeToBytes(const std::vector<uint8_t>& rleData, size_t totalBits) {
        std::vector<uint8_t> bits = decode(rleData);
        if (bits.size() < totalBits) throw std::runtime_error("RLE decode: not enough bits");
        std::string bytes((totalBits + 7) / 8, '\0');
        for (size_t i = 0; i < totalBits; ++i) {
            if (bits[i]) bytes[i/8] |= (1 << (i%8));
        }
        return bytes;
    }
}
