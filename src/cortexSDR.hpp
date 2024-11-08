#ifndef CORTEXSDR_HPP
#define CORTEXSDR_HPP

#include <iostream>
#include <bitset>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>

// Include encoder headers instead of forward declarations
#include "encoders/WordEncoding.hpp"
#include "encoders/DateTimeEncoding.hpp"
#include "encoders/SpecialCharacterEncoding.hpp"
#include "encoders/NumberEncoding.hpp"
#include "encoders/ImageEncoding.hpp"
#include "encoders/VideoEncoding.hpp"
#include "encoders/AudioEncoding.hpp"

// Define encoding ranges
struct EncodingRanges {
    static constexpr size_t WORD_START = 0;
    static constexpr size_t WORD_END = 999;
    static constexpr size_t SPECIAL_CHAR_START = 1000;
    static constexpr size_t SPECIAL_CHAR_END = 1499;
    static constexpr size_t NUMBER_START = 1500;
    static constexpr size_t NUMBER_END = 1999;
    static constexpr size_t MAX_VECTOR_SIZE = 2000;
};

class SparseDistributedRepresentation {
public:
    struct EncodedData {
        std::vector<size_t> activePositions;
        size_t totalSize;

        EncodedData() = default;
        EncodedData(std::vector<size_t> positions, size_t size)
            : activePositions(std::move(positions)), totalSize(size) {}

        size_t getMemorySize() const {
            return activePositions.size() * sizeof(size_t) + sizeof(totalSize);
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

    explicit SparseDistributedRepresentation(std::initializer_list<std::string> vocabulary);
    EncodedData encodeText(const std::string& text);
    EncodedData encodeNumber(double number);
    std::string decode() const;
    void printStats() const;

private:
    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, size_t> wordToIndex_;
    EncodedData currentEncoding_;
    std::unique_ptr<WordEncoding> wordEncoder_;
    std::unique_ptr<DateTimeEncoding> dateTimeEncoder_;
    std::unique_ptr<SpecialCharacterEncoding> specialCharEncoder_;
    std::unique_ptr<NumberEncoding> numberEncoder_;
    std::unique_ptr<ImageEncoding> imageEncoder_;
    std::unique_ptr<VideoEncoding> videoEncoder_;
    std::unique_ptr<AudioEncoding> audioEncoder_;
    std::bitset<EncodingRanges::MAX_VECTOR_SIZE> encodedVector_;

    void validateVocabularySize() const;
    void initializeWordMap();
    std::vector<std::string> tokenizeText(const std::string& text) const;
    void resetEncodedVector();
    void setIndices(const std::vector<size_t>& indices);
    EncodedData getEncodedData() const;

    struct SeparatedIndices {
        std::vector<size_t> wordIndices;
        std::vector<size_t> specialCharIndices;
        std::vector<size_t> numberIndices;
    };

    SeparatedIndices separateIndices(const EncodedData& data) const;
    std::string joinComponents(const std::vector<std::string>& components) const;
    double calculateSparsity(const EncodedData& data) const;
};

#endif // CORTEXSDR_HPP 