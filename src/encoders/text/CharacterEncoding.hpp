#ifndef CHARACTER_ENCODING_H
#define CHARACTER_ENCODING_H

#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <unordered_map>
#include <cmath>

/**
 * @brief A class for encoding and decoding text using various methods
 *
 * Supports multiple encoding/decoding schemes including character-based,
 * base64, URL encoding, and hex encoding with adaptive compression.
 */
class CharacterEncoding {
public:
    struct EncodingConfig {
        size_t contextWindowSize = 8;
        float compressionLevel = 0.8f;
        bool enablePrediction = true;
        size_t maxPatternLength;
        EncodingConfig() : contextWindowSize(8), compressionLevel(0.8f), enablePrediction(true), maxPatternLength(16) {}

    };

    struct EncodingStats {
        size_t originalSize;
        size_t encodedSize;
        float compressionRatio;
        float patternUtilization;
    };

    /**
     * @brief Constructor initializes encoding tables
     * @param config Optional encoding configuration
     * @throws std::runtime_error if initialization fails
     */
    explicit CharacterEncoding(const EncodingConfig& config = EncodingConfig());

    /**
     * @brief Destructor
     */
    ~CharacterEncoding();

    // Enhanced character encoding/decoding
    std::vector<size_t> encodeText(const std::string& text) const;
    std::vector<size_t> encodeWithContext(const std::string& text, const std::string& context) const;
    std::vector<size_t> encodePredictive(const std::string& text) const;
    std::string decodeIndices(const std::vector<size_t>& indices) const;
    std::string decodeWithContext(const std::vector<size_t>& indices, const std::string& context) const;

    // Base64 encoding/decoding
    std::string encodeBase64(const std::string& input) const;
    std::string decodeBase64(const std::string& encoded) const;

    // URL encoding/decoding
    std::string encodeURL(const std::string& input) const;
    std::string decodeURL(const std::string& encoded) const;

    // Statistics and optimization
    EncodingStats getEncodingStats() const;
    void updateStatistics(const std::string& text, const std::vector<size_t>& encoded);
    void learnPatterns(const std::string& text);
    void optimizePatterns();

    static constexpr size_t getCharsPerPosition() { return CHARS_PER_POSITION; }

private:
    static constexpr size_t CHARS_PER_POSITION = 4;
    static constexpr float ENTROPY_THRESHOLD = 0.5f;

    class Impl;
    std::unique_ptr<Impl> pimpl;
    EncodingConfig config;
    mutable std::vector<EncodingStats> historicalStats;

    // Enhanced helper methods
    size_t getCharacterPosition(char c) const;
    std::vector<size_t> encodePattern(const std::string& pattern) const;
    std::vector<size_t> predictNextPositions(const std::string& context) const;
    float calculatePatternEntropy(const std::string& pattern) const;
    static bool isBase64Char(char c);
    static unsigned char hexToByte(char first, char second);
    static bool isHexChar(char c);
    bool shouldOptimizeDictionary() const;
    size_t encodeDelta(size_t delta) const;
    size_t findBestPrediction(const std::vector<size_t>& predictions, char c) const;
    size_t encodePredictionIndex(size_t index) const;
};

#endif // CHARACTER_ENCODING_H
