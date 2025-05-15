#ifndef NUMERICAL_RLE_HPP
#define NUMERICAL_RLE_HPP

#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"
#include <vector>
#include <cstdint>
#include <type_traits> // For std::is_arithmetic

namespace CortexAICompression {

// Implementation of RLE for sequences of arithmetic types (int, float, etc.)
class NumericalRLEStrategy : public ICompressionStrategy {
public:
    // Compresses numerical data from a ModelSegment using RLE.
    // Assumes segment.data contains a sequence of a specific arithmetic type.
    // The type needs to be known or inferred (e.g., from segment.type).
    std::vector<std::byte> compress(const ModelSegment& segment) const override;

    // Decompresses RLE data back into a sequence of raw bytes.
    // Requires the original segment type to correctly interpret the data.
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData, SegmentType originalType, size_t originalSize) const override;

private:
    // Templated helper for compressing sequences of a specific type T
    template <typename T>
    std::vector<std::byte> compressTyped(const std::vector<std::byte>& rawData) const;

    // Templated helper for decompressing sequences of a specific type T
    template <typename T>
    std::vector<std::byte> decompressTyped(const std::vector<std::byte>& rleData) const;

    // Helper to determine element size based on segment type
    size_t getElementSize(SegmentType type) const;
};

// --- Template Implementations (can be moved to .cpp if preferred) ---

template <typename T>
std::vector<std::byte> NumericalRLEStrategy::compressTyped(const std::vector<std::byte>& rawData) const {
    static_assert(std::is_arithmetic_v<T>, "NumericalRLE only supports arithmetic types.");
    if (rawData.empty() || rawData.size() % sizeof(T) != 0) {
        throw CompressionError("Invalid data size for NumericalRLE compression with the specified type.");
    }

    const T* dataPtr = reinterpret_cast<const T*>(rawData.data());
    size_t numElements = rawData.size() / sizeof(T);
    std::vector<std::byte> compressedOutput;

    if (numElements == 0) return compressedOutput;

    T currentValue = dataPtr[0];
    uint8_t runLength = 1; // Use uint8_t for run length, handle longer runs if needed

    for (size_t i = 1; i < numElements; ++i) {
        if (dataPtr[i] == currentValue && runLength < 255) {
            runLength++;
        } else {
            // Write the run: (value, length)
            const std::byte* valueBytes = reinterpret_cast<const std::byte*>(&currentValue);
            compressedOutput.insert(compressedOutput.end(), valueBytes, valueBytes + sizeof(T));
            compressedOutput.push_back(static_cast<std::byte>(runLength));

            // Start new run
            currentValue = dataPtr[i];
            runLength = 1;
        }
    }

    // Write the last run
    const std::byte* valueBytes = reinterpret_cast<const std::byte*>(&currentValue);
    compressedOutput.insert(compressedOutput.end(), valueBytes, valueBytes + sizeof(T));
    compressedOutput.push_back(static_cast<std::byte>(runLength));

    return compressedOutput;
}

template <typename T>
std::vector<std::byte> NumericalRLEStrategy::decompressTyped(const std::vector<std::byte>& rleData) const {
    static_assert(std::is_arithmetic_v<T>, "NumericalRLE only supports arithmetic types.");
    std::vector<std::byte> decompressedOutput;
    size_t elementSize = sizeof(T);
    size_t pairSize = elementSize + sizeof(uint8_t); // Size of (value, runLength)

    if (rleData.size() % pairSize != 0) {
         throw CompressionError("Invalid RLE data size for decompression.");
    }

    for (size_t i = 0; i < rleData.size(); i += pairSize) {
        T value = *reinterpret_cast<const T*>(&rleData[i]);
        uint8_t runLength = static_cast<uint8_t>(rleData[i + elementSize]);

        if (runLength == 0) {
             throw CompressionError("Invalid run length (0) encountered during RLE decompression.");
        }

        const std::byte* valueBytes = reinterpret_cast<const std::byte*>(&value);
        for (uint8_t j = 0; j < runLength; ++j) {
            decompressedOutput.insert(decompressedOutput.end(), valueBytes, valueBytes + elementSize);
        }
    }

    return decompressedOutput;
}


} // namespace CortexAICompression

#endif // NUMERICAL_RLE_HPP
