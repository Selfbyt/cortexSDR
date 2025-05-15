#ifndef DEBUG_UTILS_HPP
#define DEBUG_UTILS_HPP

#include <iostream>
#include <vector>
#include <string>

// Debug utility functions for SDR compression/decompression

namespace DebugUtils {

// Print vocabulary information
inline void printVocabulary(const std::vector<std::string>& vocabulary, const std::string& prefix) {
    std::cout << "[DEBUG-" << prefix << "] Vocabulary size: " << vocabulary.size() << std::endl;
    std::cout << "[DEBUG-" << prefix << "] First 10 words in vocabulary: ";
    size_t count = 0;
    for (const auto& word : vocabulary) {
        if (count++ < 10) {
            std::cout << "'" << word << "' ";
        } else {
            break;
        }
    }
    std::cout << std::endl;
}

// Print active positions in SDR
inline void printActivePositions(const std::vector<size_t>& positions, const std::string& prefix) {
    std::cout << "[DEBUG-" << prefix << "] Total active positions: " << positions.size() << std::endl;
    std::cout << "[DEBUG-" << prefix << "] First 20 active positions: ";
    size_t count = 0;
    for (const auto& pos : positions) {
        if (count++ < 20) {
            std::cout << pos << " ";
        } else {
            break;
        }
    }
    std::cout << std::endl;
}

// Print indices by region
inline void printIndicesByRegion(const std::vector<size_t>& wordIndices, 
                                 const std::vector<size_t>& specialCharIndices,
                                 const std::vector<size_t>& numberIndices) {
    std::cout << "[DEBUG-DECODE] Word indices: " << wordIndices.size() 
              << ", Special char indices: " << specialCharIndices.size()
              << ", Number indices: " << numberIndices.size() << std::endl;
    
    if (!wordIndices.empty()) {
        std::cout << "[DEBUG-DECODE] First 10 word indices: ";
        for (size_t i = 0; i < std::min(wordIndices.size(), size_t(10)); ++i) {
            std::cout << wordIndices[i] << " ";
        }
        std::cout << std::endl;
    }
}

// Print fingerprint overlap information
inline void printFingerprintOverlap(size_t windowIndex, int bestOverlap, const std::string& bestWord) {
    std::cout << "[DEBUG-DECODE] Window " << windowIndex 
              << ": bestOverlap=" << bestOverlap 
              << ", bestWord=" << bestWord << std::endl;
}

// Print serialization information
inline void printSerializationInfo(const std::vector<std::string>& vocabulary, size_t dataSize) {
    std::cout << "[DEBUG-SERIALIZE] Serializing vocabulary of size: " << vocabulary.size() 
              << ", data size: " << dataSize << " bytes" << std::endl;
}

// Print deserialization information
inline void printDeserializationInfo(const std::vector<std::string>& vocabulary) {
    std::cout << "[DEBUG-DESERIALIZE] Deserialized vocabulary of size: " << vocabulary.size() << std::endl;
    if (!vocabulary.empty()) {
        std::cout << "[DEBUG-DESERIALIZE] First 5 words: ";
        for (size_t i = 0; i < std::min(vocabulary.size(), size_t(5)); ++i) {
            std::cout << "'" << vocabulary[i] << "' ";
        }
        std::cout << std::endl;
    }
}

} // namespace DebugUtils

#endif // DEBUG_UTILS_HPP
