#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

#include <iostream>
#include <vector>
#include <string>

// Debug utility functions for SDR compression/decompression

namespace DebugUtils {

// Print vocabulary information
inline void printVocabulary(const std::vector<std::string>& vocabulary, const std::string& prefix) {
    // Debug output disabled
}

// Print active positions in SDR
inline void printActivePositions(const std::vector<size_t>& positions, const std::string& prefix) {
    // Debug output disabled
}

// Print indices by region
inline void printIndicesByRegion(const std::vector<size_t>& wordIndices, 
                                 const std::vector<size_t>& specialCharIndices,
                                 const std::vector<size_t>& numberIndices) {
    // Debug output disabled
}

// Print fingerprint overlap information
inline void printFingerprintOverlap(size_t windowIndex, int bestOverlap, const std::string& bestWord) {
    // Debug output disabled
}

// Print serialization information
inline void printSerializationInfo(const std::vector<std::string>& vocabulary, size_t dataSize) {
    // Debug output disabled
}

// Print deserialization information
inline void printDeserializationInfo(const std::vector<std::string>& vocabulary) {
    // Debug output disabled
}

} // namespace DebugUtils

#endif // DEBUG_UTILS_HPP
