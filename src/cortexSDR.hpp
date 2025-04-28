#ifndef CORTEXSDR_HPP
#define CORTEXSDR_HPP

#include <iostream>
#include <bitset>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <cstdint>

// Include encoder headers instead of forward declarations
#include "encoders/text/WordEncoding.hpp"
#include "encoders/numeric/DateTimeEncoding.hpp"
#include "encoders/numeric/NumberEncoding.hpp"
#include "encoders/media/ImageEncoding.hpp"
#include "encoders/media/VideoEncoding.hpp"
#include "encoders/media/AudioEncoding.hpp"
#include "encoders/text/CharacterEncoding.hpp"
#include "encoders/text/SpecialCharEncoding.hpp" // Correct path confirmed by file listing
#include "encoders/numeric/GeoEncoding.hpp"
#include "encoders/adapters/SpecialCharEncodingAdapter.hpp"
#include "encoders/adapters/GeoEncodingAdapter.hpp"

// Define encoding ranges
struct EncodingRanges {
    static constexpr size_t WORD_START = 0;
    static constexpr size_t WORD_END = 9999;
    static constexpr size_t SPECIAL_CHAR_START = 10000;
    static constexpr size_t SPECIAL_CHAR_END = 14999;
    static constexpr size_t NUMBER_START = 15000;
    static constexpr size_t GEO_START = 18000;
    static constexpr size_t GEO_END = 18999;
    static constexpr size_t NUMBER_END = 19999;
    static constexpr size_t MAX_VECTOR_SIZE = 20000;
};

class SparseDistributedRepresentation {
public:
    // Sparsity: fraction of active bits (0.0–1.0)
    void setSparsity(double sparsity);
    double getSparsity() const;
public:

public:
    struct EncodedData {
        std::vector<size_t> activePositions;
        size_t totalSize;

        EncodedData() = default;
        EncodedData(std::vector<size_t> positions, size_t size)
            : activePositions(std::move(positions)), totalSize(size) {}

        size_t getMemorySize() const {
            // On-disk SDR size: totalSize (size_t) + numActivePositions (uint32_t) + indices (uint16_t each)
            return sizeof(totalSize) + sizeof(uint32_t) + activePositions.size() * sizeof(uint16_t);
        }

        EncodedData& operator|=(const EncodedData& other) {
            std::vector<size_t> combined;
            std::set_union(
                activePositions.begin(), activePositions.end(),
                other.activePositions.begin(), other.activePositions.end(),
                std::back_inserter(combined));
            activePositions = std::move(combined);
            return *this;
        }

        double calculateOverlap(const EncodedData& other) const {
            std::vector<size_t> intersection;
            std::set_intersection(
                activePositions.begin(), activePositions.end(),
                other.activePositions.begin(), other.activePositions.end(),
                std::back_inserter(intersection));
            return static_cast<double>(intersection.size()) /
                   std::min(activePositions.size(), other.activePositions.size());
        }
    };

    SparseDistributedRepresentation(); // Updated constructor (removed explicit and vocabulary param)
    EncodedData encodeText(const std::string& text);
    EncodedData encodeNumber(double number);
    EncodedData encodeImage(const std::string& imagePath);
    std::string decode() const;
    std::string decodeImage() const;
    void setEncoding(const EncodedData& data); // Added for decompression
    void printStats() const;
    
    // Method to check if the encoded data is an image
    bool isImageData() const;

    // Methods to access word encoder's vocabulary state for serialization
    const std::vector<std::string>& getWordVocabulary() const;
    // Removed getWordToIndexMap as it's no longer needed
    void setWordVocabulary(const std::vector<std::string>& vocab); // Updated signature
    EncodedData getEncodedData() const; // Moved to public
    const std::vector<EncodedData>& getTokenEncodings() const;

private:
    EncodedData currentEncoding_;
    double sparsity_ = 0.002; // Default 0.2% active bits (very sparse for high compression)

    std::unique_ptr<WordEncoding> wordEncoder_;
    std::unique_ptr<DateTimeEncoding> dateTimeEncoder_;
    std::unique_ptr<GeoEncodingAdapter> geoEncoder_;
    std::unique_ptr<NumberEncoding> numberEncoder_;
    std::unique_ptr<ImageEncoding> imageEncoder_;
    std::unique_ptr<VideoEncoding> videoEncoder_;
    std::unique_ptr<AudioEncoding> audioEncoder_;
    std::bitset<EncodingRanges::MAX_VECTOR_SIZE> encodedVector_;
    std::vector<EncodedData> tokenEncodings_;

    // Removed validateVocabularySize() and initializeWordMap() declarations
    std::vector<std::string> tokenizeText(const std::string& text) const;
    void resetEncodedVector();
    void setIndices(const std::vector<size_t>& indices);
    // Removed getEncodedData() declaration from private section

    struct SeparatedIndices {
        std::vector<size_t> wordIndices;
        std::vector<size_t> specialCharIndices;
        std::vector<size_t> numberIndices;
    };

    SeparatedIndices separateIndices(const EncodedData& data) const;
    std::string joinComponents(const std::vector<std::string>& components) const;
    double calculateSparsity(const EncodedData& data) const;
};

// New classes for Brain-Inspired Optimizations and related enhancements
// (Implementation inspired by BaMI-SDR: https://www.numenta.com/assets/pdf/biological-and-machine-intelligence/BaMI-SDR.pdf)

class BrainInspiredSDR {
public:
    struct Synapse {
        uint16_t position;
        float strength;  // Hebbian update strength
    };
    struct Pattern {
        std::vector<Synapse> connections;
        float frequency;  // Updated based on occurrence
        uint32_t context; // Encoded contextual info
    };
    struct HierarchicalPattern {
        std::vector<Pattern> lowLevel;
        std::vector<Pattern> midLevel;
        std::vector<Pattern> highLevel;
    };

    void optimizePatterns() {
        // TODO: Implement hierarchical pattern optimization inspired by cortical columns (BaMI-SDR)
    }
};

class AdaptiveEncoder {
public:
    void learnPatterns(const std::string& text) {
        // TODO: Update frequency, adapt encoding, merge similar patterns per BaMI-SDR principles.
    }
};

class ContextualSDR {
public:
    using EncodedData = SparseDistributedRepresentation::EncodedData;
    
    EncodedData encodeWithContext(const std::string& text) {
        // TODO: Analyze local context and encode only unexpected deviations (BaMI-SDR)
        return EncodedData({}, EncodingRanges::MAX_VECTOR_SIZE);
    }
};

class SemanticEncoder {
public:
    using EncodedData = SparseDistributedRepresentation::EncodedData;
    
    std::vector<EncodedData> semanticClusters;
    void clusterRelatedConcepts() {
        // TODO: Group semantically similar patterns to reduce redundancy (based on BaMI-SDR)
    }
};

class PredictiveEncoder {
public:
    using EncodedData = SparseDistributedRepresentation::EncodedData;
    
    EncodedData encode(const std::string& text) {
        // TODO: Compare predicted and actual patterns and encode the difference (BaMI-SDR)
        return EncodedData({}, EncodingRanges::MAX_VECTOR_SIZE);
    }
};

class OptimizedSDR {
public:
    // Forward declaration of Pattern type
    using Pattern = BrainInspiredSDR::Pattern;
    
    struct CompressedPosition {
        uint8_t prefix;
        uint16_t offset;
    };
    struct PatternPool {
        std::unordered_map<Pattern, uint16_t> commonPatterns;
    };
    struct TemporalCache {
        std::vector<Pattern> recentPatterns;
    };
};

class BrainLikeCompression {
public:
    using EncodedData = SparseDistributedRepresentation::EncodedData;
    
    static constexpr float SPARSITY_TARGET = 0.02f; // 2% active neurons
    void strengthenConnections(const EncodedData& pattern) {
        // TODO: Apply Hebbian learning to fortify common connections (BaMI-SDR)
    }
    EncodedData predictiveEncode(const std::string& text) {
        // TODO: Encode only unexpected information using predictive error (BaMI-SDR)
        return EncodedData({}, EncodingRanges::MAX_VECTOR_SIZE);
    }
};

#endif // CORTEXSDR_HPP
