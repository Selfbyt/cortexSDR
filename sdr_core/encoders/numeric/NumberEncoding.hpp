#ifndef NUMBER_ENCODING_HPP
#define NUMBER_ENCODING_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <random>

class NumberEncoding {
public:
    struct EncodingConfig {
        size_t startIndex = 1000;
        size_t bucketCount = 100;
        double minValue = -1000.0;
        double maxValue = 1000.0;
        bool adaptiveBuckets = true;
        float adaptationRate = 0.1f;
    };

    explicit NumberEncoding(const EncodingConfig& config);

    // Enhanced number encoding methods
    std::vector<size_t> encodeNumber(double number) const;
    std::vector<size_t> encodeNumberString(const std::string& numbers) const;
    std::vector<size_t> encodeRange(double start, double end, double step) const;

    // Advanced decoding methods
    std::string decodeIndices(const std::vector<size_t>& indices) const;
    double decodeExact(const std::vector<size_t>& indices) const;
    std::pair<double, double> decodeRange(const std::vector<size_t>& indices) const;

    // Optimization methods
    void updateBuckets(double number);
    void optimizeQuantization();
    float getCompressionRatio() const;

private:
    // BucketStats struct definition
    struct BucketStats {
        size_t hitCount = 0;
        double mean = 0.0;
        double variance = 0.0;
        std::vector<double> values; // Store values for more complex optimization
    };
    
    // Implementation struct definition
    struct Impl {
        // Adaptive learning rate parameters
        static constexpr float INITIAL_LEARNING_RATE = 0.01f;
        static constexpr float MIN_LEARNING_RATE = 0.001f;
        static constexpr float LEARNING_RATE_DECAY = 0.995f;

        // Optimization parameters
        static constexpr size_t MIN_SAMPLES_FOR_ADAPTATION = 100;
        static constexpr float VARIANCE_THRESHOLD = 0.01f;
        static constexpr size_t MAX_ITERATIONS = 1000;

        float currentLearningRate = INITIAL_LEARNING_RATE;
        std::mt19937 rng{std::random_device{}()};
    };
    std::unique_ptr<Impl> pimpl;
    
    EncodingConfig config_;
    std::vector<double> bucketBoundaries_;
    std::unordered_map<size_t, BucketStats> bucketStats_;
    mutable std::vector<double> recentValues_;

    // Renamed from findBucket
    size_t getBucketIndex(double value) const;
    void redistributeBuckets();
    double interpolateValue(size_t bucket, double position) const;
    // Renamed from updateBucketStats
    void updateStatistics(double value, size_t bucket);

    // Special value handling helpers
    std::vector<size_t> encodeSpecialValue(double number) const;
    bool isSpecialValue(size_t index) const;
    double decodeSpecialValue(size_t index) const;

    // Optimization helpers (declarations needed if used internally)
    void optimizeBoundary(size_t boundaryIndex);
    void resetBucketStats();
    bool shouldOptimizeBuckets() const;
};

#endif // NUMBER_ENCODING_HPP
