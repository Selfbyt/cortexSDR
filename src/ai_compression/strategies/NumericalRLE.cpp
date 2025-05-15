#include "NumericalRLE.hpp"
#include "../core/ModelSegment.hpp"
#include <stdexcept>       // For std::runtime_error
#include <map>             // For mapping types to sizes

namespace CortexAICompression {

// Helper to get the size of the element based on the segment type
size_t NumericalRLEStrategy::getElementSize(SegmentType type) const {
    static const std::map<SegmentType, size_t> type_sizes = {
        {SegmentType::WEIGHTS_FP32, sizeof(float)},
        {SegmentType::WEIGHTS_FP16, sizeof(uint16_t)}, // Assuming FP16 is stored as uint16_t
        {SegmentType::WEIGHTS_INT8, sizeof(int8_t)},
        {SegmentType::WEIGHTS_INT4, 0} // Special handling needed for INT4 (packed)
        // Add other numerical types if needed
    };

    auto it = type_sizes.find(type);
    if (it != type_sizes.end()) {
        if (it->second == 0) {
             throw CompressionError("NumericalRLEStrategy does not directly support packed types like INT4 yet.");
        }
        return it->second;
    }
    throw CompressionError("Unsupported segment type for NumericalRLEStrategy.");
}


std::vector<std::byte> NumericalRLEStrategy::compress(const ModelSegment& segment) const {
    size_t elementSize = getElementSize(segment.type); // Throws if unsupported type

    // Dispatch to the correct typed implementation based on element size
    // Note: This assumes the byte representation matches standard types.
    // For FP16 or other custom types, more specific handling might be needed.
    switch (elementSize) {
        case sizeof(float): // Assuming FP32
            return compressTyped<float>(segment.data);
        case sizeof(uint16_t): // Assuming FP16 stored as uint16_t
            // Caution: This treats FP16 bits as uint16_t for comparison.
            // This is generally okay for lossless RLE but relies on exact bit patterns.
            return compressTyped<uint16_t>(segment.data);
        case sizeof(int8_t): // Assuming INT8
            return compressTyped<int8_t>(segment.data);
        // Add cases for other supported sizes (e.g., int32, double) if necessary
        default:
            // This case should ideally not be reached due to getElementSize check,
            // but included for safety.
            throw CompressionError("Internal error: Unhandled element size in NumericalRLE compress.");
    }
}

std::vector<std::byte> NumericalRLEStrategy::decompress(const std::vector<std::byte>& compressedData, SegmentType originalType, size_t originalSize) const {
    // We need to know the original type to decompress correctly.
    // The archive format MUST store the original SegmentType.
    // The originalType is now passed as a parameter.
    // Use the provided originalType to determine the element size and call the correct template specialization.
    size_t elementSize = getElementSize(originalType); // Throws if unsupported type

    // Dispatch to the correct typed implementation
    switch (elementSize) {
        case sizeof(float): // Assuming FP32
            return decompressTyped<float>(compressedData);
        case sizeof(uint16_t): // Assuming FP16 stored as uint16_t
            return decompressTyped<uint16_t>(compressedData);
        case sizeof(int8_t): // Assuming INT8
            return decompressTyped<int8_t>(compressedData);
        // Add cases for other supported sizes if necessary
        default:
            // This case should ideally not be reached due to getElementSize check,
            // but included for safety.
             throw CompressionError("Internal error: Unhandled element size in NumericalRLE decompress.");
    }

    // Note: We still rely on originalSize for potential validation within decompressTyped,
    // although the primary type dispatch is now based on originalType.
    // A final check could be added here:
    // std::vector<std::byte> decompressed = ...;
    // if (decompressed.size() != originalSize) {
    //     throw CompressionError("Decompressed size does not match original size.");
    // }
    // return decompressed;
}

} // namespace CortexAICompression
