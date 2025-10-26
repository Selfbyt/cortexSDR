#ifndef RLE_HPP
#define RLE_HPP
#include <vector>
#include <cstdint>
#include <string>

// Run-Length Encoding for binary SDR data
namespace rle {
    // Encodes a vector of bits (0/1) as (value, run-length) pairs
    std::vector<uint8_t> encode(const std::vector<uint8_t>& bits);
    // Decodes an RLE-encoded vector back to bits
    std::vector<uint8_t> decode(const std::vector<uint8_t>& rleData);
    // Convenience: encode a string of bytes (bit-packed) to RLE
    std::vector<uint8_t> encodeFromBytes(const std::string& bytes);
    // Convenience: decode RLE back to bit-packed string
    std::string decodeToBytes(const std::vector<uint8_t>& rleData, size_t totalBits);
}

#endif // RLE_HPP
