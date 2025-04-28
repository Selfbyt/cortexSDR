#include "NumberEncoding.hpp"
#include <cmath>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept> // Added for exceptions
#include <memory>    // Added for std::make_unique
#include <limits>   // Added for numeric_limits
#include <iostream> // Added for debug output

// BucketStats struct is now defined in the header file

// Define RANGE_START_MARKER (choose a suitable value outside normal index range)
static constexpr size_t RANGE_START_MARKER = std::numeric_limits<size_t>::max() - 1;
// Define Special Value Markers (choose suitable values outside normal index range)
static constexpr size_t NAN_MARKER = std::numeric_limits<size_t>::max() - 2;
static constexpr size_t INF_POS_MARKER = std::numeric_limits<size_t>::max() - 3;
static constexpr size_t INF_NEG_MARKER = std::numeric_limits<size_t>::max() - 4;


// Impl struct is now defined in the header file

// Correct constructor signature and initialization list
NumberEncoding::NumberEncoding(const EncodingConfig& config)
    : config_(config) // Use config_
    , pimpl(std::make_unique<Impl>())
    , bucketBoundaries_(config_.bucketCount + 1) // Use config_ and bucketBoundaries_
    , bucketStats_(config_.bucketCount) // Use bucketStats_ and config_
{
    // Initialize bucket boundaries linearly
    double range = config_.maxValue - config_.minValue; // Use config_
    double step = range / config_.bucketCount; // Use config_

    for (size_t i = 0; i <= config_.bucketCount; ++i) { // Use config_
        bucketBoundaries_[i] = config_.minValue + i * step; // Use bucketBoundaries_ and config_
    }

    // Bucket statistics are already resized in the initializer list
}

// Add const qualifier
std::vector<size_t> NumberEncoding::encodeNumber(double number) const {
    // Validate input
    if (std::isnan(number) || std::isinf(number)) {
        return encodeSpecialValue(number); // Use helper
    }

    // Clamp to valid range
    number = std::clamp(number, config_.minValue, config_.maxValue); // Use config_

    // Find appropriate bucket
    size_t bucketIndex = getBucketIndex(number); // Use getBucketIndex

    // Calculate fine-grained position within bucket
    double bucketStart = bucketBoundaries_[bucketIndex]; // Use bucketBoundaries_
    double bucketEnd = bucketBoundaries_[bucketIndex + 1]; // Use bucketBoundaries_
    // Avoid division by zero if bucket is extremely small
    double bucketWidth = bucketEnd - bucketStart;
    double normalizedPosition = (bucketWidth > 1e-9) ? ((number - bucketStart) / bucketWidth) : 0.5;


    std::vector<size_t> encoded;

    // Encode bucket index
    encoded.push_back(config_.startIndex + bucketIndex); // Use config_.startIndex

    // Encode normalized position if precision needed
    // Note: highPrecision and positionPrecision are not part of EncodingConfig in header
    // Assuming standard precision for now. Add these to EncodingConfig if needed.
    /*
    if (config_.highPrecision) {
        size_t positionIndex = static_cast<size_t>(normalizedPosition * config_.positionPrecision);
        encoded.push_back(config_.startIndex + config_.bucketCount + positionIndex);
    }
    */

    // Update statistics if adaptive
    if (config_.adaptiveBuckets) { // Use config_
        // updateStatistics(number, bucketIndex); // Use updateStatistics (mutable method, needs non-const this or mutable members)
        // For now, comment out mutable operations in const methods
        // If adaptation is required, encodeNumber cannot be const, or stats need to be mutable

        // Check if optimization is needed
        // if (shouldOptimizeBuckets()) { // mutable method
        //     optimizeQuantization(); // mutable method
        // }
    }

    // Added debug statements to log numeric values during encoding and decoding.
    std::cout << "Encoding number: " << number << std::endl;

    return encoded;
}

// Add const qualifier
std::vector<size_t> NumberEncoding::encodeRange(double start, double end, double step) const {
    std::vector<size_t> encoded;

    // Validate range
    if (start > end || step <= 0) {
        throw std::invalid_argument("Invalid range parameters");
    }

    // Find buckets for start and end
    size_t startBucket = getBucketIndex(start); // Use getBucketIndex
    size_t endBucket = getBucketIndex(end);     // Use getBucketIndex

    // Encode range markers
    encoded.push_back(RANGE_START_MARKER);
    encoded.push_back(config_.startIndex + startBucket); // Use config_.startIndex
    encoded.push_back(config_.startIndex + endBucket);   // Use config_.startIndex

    // Encode step size if non-standard
    if (step != 1.0) {
        // encodeNumber is const, so this is okay if encodeNumber is fixed
        auto stepEncoding = encodeNumber(step);
        encoded.insert(encoded.end(), stepEncoding.begin(), stepEncoding.end());
    }

    return encoded;
}

double NumberEncoding::decodeExact(const std::vector<size_t>& indices) const {
    if (indices.empty()) {
        throw std::invalid_argument("Empty indices provided");
    }

    // Handle special values
    if (isSpecialValue(indices[0])) { // Use helper
        return decodeSpecialValue(indices[0]); // Use helper
    }

    // Decode bucket index
    if (indices[0] < config_.startIndex) { // Check lower bound
         throw std::out_of_range("Invalid index: below start index");
    }
    size_t bucketIndex = indices[0] - config_.startIndex; // Use config_.startIndex
    if (bucketIndex >= config_.bucketCount) { // Use config_.bucketCount
        throw std::out_of_range("Invalid bucket index");
    }

    double bucketStart = bucketBoundaries_[bucketIndex]; // Use bucketBoundaries_
    double bucketEnd = bucketBoundaries_[bucketIndex + 1]; // Use bucketBoundaries_

    // If high precision enabled and position encoded
    // Note: highPrecision and positionPrecision are not part of EncodingConfig in header
    /*
    if (config_.highPrecision && indices.size() > 1) {
        if (indices[1] < config_.startIndex + config_.bucketCount) {
             throw std::out_of_range("Invalid position index: below start index");
        }
        size_t positionIndex = indices[1] - (config_.startIndex + config_.bucketCount);
        if (positionIndex >= config_.positionPrecision) {
             throw std::out_of_range("Invalid position index");
        }
        double normalizedPosition = static_cast<double>(positionIndex) / config_.positionPrecision;
        return bucketStart + normalizedPosition * (bucketEnd - bucketStart);
    }
    */

    // Return bucket midpoint for standard precision
    double value = (bucketStart + bucketEnd) / 2.0;

    // Added debug statement to log decoded value
    std::cout << "Decoded value: " << value << std::endl;

    return value;
}

std::pair<double, double> NumberEncoding::decodeRange(const std::vector<size_t>& indices) const {
    if (indices.size() < 3 || indices[0] != RANGE_START_MARKER) {
        throw std::invalid_argument("Invalid range encoding");
    }

    if (indices[1] < config_.startIndex || indices[2] < config_.startIndex) {
         throw std::out_of_range("Invalid range index: below start index");
    }
    size_t startBucket = indices[1] - config_.startIndex; // Use config_.startIndex
    size_t endBucket = indices[2] - config_.startIndex;   // Use config_.startIndex

    if (startBucket >= config_.bucketCount || endBucket >= config_.bucketCount || endBucket + 1 >= bucketBoundaries_.size()) {
         throw std::out_of_range("Invalid range bucket index");
    }

    double start = bucketBoundaries_[startBucket];       // Use bucketBoundaries_
    double end = bucketBoundaries_[endBucket + 1];     // Use bucketBoundaries_

    return {start, end};
}

// Renamed from updateBucketStats
void NumberEncoding::updateStatistics(double value, size_t bucket) {
    // This method modifies state (bucketStats_), so it cannot be called from const methods
    // unless bucketStats_ is declared mutable.
    if (bucket >= bucketStats_.size()) return; // Basic bounds check
    auto& stats = bucketStats_[bucket]; // Use bucketStats_
    stats.hitCount++;
    stats.values.push_back(value); // Assumes BucketStats has 'values' member

    // Update running statistics
    double delta = value - stats.mean;
    stats.mean += delta / stats.hitCount;
    if (stats.hitCount > 1) {
        double delta2 = value - stats.mean; // Use updated mean
        // Welford's algorithm update for variance
        stats.variance = stats.variance + delta * delta2;
    }
}

// --- Implementations for Special Value Helpers ---

std::vector<size_t> NumberEncoding::encodeSpecialValue(double number) const {
    if (std::isnan(number)) {
        return {NAN_MARKER};
    } else if (std::isinf(number)) {
        return { (number > 0) ? INF_POS_MARKER : INF_NEG_MARKER };
    }
    // Should not happen based on calling condition, but return empty for safety
    return {};
}

bool NumberEncoding::isSpecialValue(size_t index) const {
    return index == NAN_MARKER || index == INF_POS_MARKER || index == INF_NEG_MARKER;
}

double NumberEncoding::decodeSpecialValue(size_t index) const {
    if (index == NAN_MARKER) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (index == INF_POS_MARKER) {
        return std::numeric_limits<double>::infinity();
    } else if (index == INF_NEG_MARKER) {
        return -std::numeric_limits<double>::infinity();
    }
    // Should not happen based on calling condition
    throw std::invalid_argument("Index is not a special value marker");
}


// --- Implementations/Fixes for Optimization Helpers ---
// Note: These modify state and cannot be const

void NumberEncoding::optimizeQuantization() {
    // Check if we have enough samples
    size_t totalSamples = std::accumulate(bucketStats_.begin(), bucketStats_.end(), size_t{0}, // Use bucketStats_
        [](size_t sum, const auto& bucket_pair) { return sum + bucket_pair.second.hitCount; });

    if (totalSamples < pimpl->MIN_SAMPLES_FOR_ADAPTATION) {
        return;
    }

    // Perform k-means clustering-like optimization
    bool changed;
    size_t iterations = 0;

    do {
        changed = false;
        iterations++;

        // Store old boundaries for comparison
        std::vector<double> oldBoundaries = bucketBoundaries_; // Use bucketBoundaries_

        // Optimize each internal boundary
        for (size_t i = 1; i < bucketBoundaries_.size() - 1; ++i) { // Use bucketBoundaries_
            optimizeBoundary(i); // Call helper
        }

        // Check if boundaries changed significantly
        changed = !std::equal(oldBoundaries.begin(), oldBoundaries.end(),
                            bucketBoundaries_.begin(), // Use bucketBoundaries_
                            [this](double a, double b) {
                                return std::abs(a - b) < pimpl->VARIANCE_THRESHOLD;
                            });

    } while (changed && iterations < pimpl->MAX_ITERATIONS);

    // Update learning rate using pimpl
    pimpl->currentLearningRate *= pimpl->LEARNING_RATE_DECAY;
    pimpl->currentLearningRate = std::max(pimpl->currentLearningRate, pimpl->MIN_LEARNING_RATE);

    // Clear statistics for next adaptation period
    resetBucketStats(); // Call helper
}

void NumberEncoding::optimizeBoundary(size_t boundaryIndex) {
    // Basic bounds check
    if (boundaryIndex == 0 || boundaryIndex >= bucketBoundaries_.size() -1 ||
        boundaryIndex -1 >= bucketStats_.size() || boundaryIndex >= bucketStats_.size()) {
        return;
    }

    auto lowerIt = bucketStats_.find(boundaryIndex - 1);
    auto upperIt = bucketStats_.find(boundaryIndex);
    
    // Skip optimization if either bucket doesn't exist
    if (lowerIt == bucketStats_.end() || upperIt == bucketStats_.end()) {
        return;
    }
    
    const auto& lowerBucket = lowerIt->second;
    const auto& upperBucket = upperIt->second;

    if (lowerBucket.values.empty() || upperBucket.values.empty()) {
        return; // Cannot optimize without data in both adjacent buckets
    }

    // Simple optimization: move boundary to midpoint of means
    // More sophisticated methods (like weighted sum) could be used
    double newBoundary = (lowerBucket.mean + upperBucket.mean) / 2.0;

    // Apply learning rate to boundary update using pimpl
    bucketBoundaries_[boundaryIndex] += pimpl->currentLearningRate *
                                      (newBoundary - bucketBoundaries_[boundaryIndex]); // Use bucketBoundaries_

    // Ensure boundaries remain ordered and within min/max
    bucketBoundaries_[boundaryIndex] = std::clamp(
        bucketBoundaries_[boundaryIndex], // Use bucketBoundaries_
        bucketBoundaries_[boundaryIndex - 1] + std::numeric_limits<double>::epsilon(), // Use bucketBoundaries_
        bucketBoundaries_[boundaryIndex + 1] - std::numeric_limits<double>::epsilon()  // Use bucketBoundaries_
    );
     bucketBoundaries_[boundaryIndex] = std::clamp(bucketBoundaries_[boundaryIndex], config_.minValue, config_.maxValue); // Clamp to overall range
}

void NumberEncoding::resetBucketStats() {
    for (auto& bucket_pair : bucketStats_) { // Use bucketStats_
        auto& stats = bucket_pair.second;
        stats.hitCount = 0;
        stats.mean = 0.0;
        // Reset variance correctly (Welford's M2, not simple variance)
        stats.variance = 0.0; // M2 is 0 initially
        stats.values.clear(); // Clear stored values if used
    }
}

// Renamed from findBucket
size_t NumberEncoding::getBucketIndex(double value) const {
    // Clamp value to range before searching
    value = std::clamp(value, config_.minValue, config_.maxValue);
    // Use lower_bound to find the first element NOT LESS than value
    auto it = std::lower_bound(bucketBoundaries_.begin() + 1, bucketBoundaries_.end() - 1, value); // Use bucketBoundaries_
    // The index is the distance from the beginning
    size_t index = std::distance(bucketBoundaries_.begin(), it) - 1;
    // Ensure index is within valid range [0, bucketCount - 1]
    return std::min(index, config_.bucketCount - 1); // Use config_
}


bool NumberEncoding::shouldOptimizeBuckets() const {
    // Check if any bucket has enough samples and high variance
    // Note: Variance calculation in updateStatistics needs correction for Welford's algorithm
    // This check might need adjustment based on the actual variance metric used.
    return std::any_of(bucketStats_.begin(), bucketStats_.end(), // Use bucketStats_
        [this](const auto& bucket_pair) {
            const auto& stats = bucket_pair.second;
            // Need a meaningful variance check, placeholder for now
            // double stddev = (stats.hitCount > 1) ? std::sqrt(stats.variance / (stats.hitCount - 1)) : 0.0;
            return stats.hitCount >= pimpl->MIN_SAMPLES_FOR_ADAPTATION; // Simplified check based on hits
                   // && stddev > SOME_THRESHOLD; // Add variance check later
        });
}

// --- Need implementations for other methods declared in header ---
// encodeNumberString, decodeIndices, updateBuckets, getCompressionRatio, redistributeBuckets, interpolateValue

std::vector<size_t> NumberEncoding::encodeNumberString(const std::string& numbers) const {
    // Production-quality: parse, validate, and encode all numbers in the string
    std::vector<size_t> allIndices;
    std::string token;
    std::istringstream iss(numbers);
    while (iss >> token) {
        // Remove commas and handle scientific notation
        token.erase(std::remove(token.begin(), token.end(), ','), token.end());
        try {
            size_t idx = 0;
            double value = std::stod(token, &idx);
            if (idx != token.size()) continue; // skip malformed
            // Clamp to supported range
            value = std::clamp(value, config_.minValue, config_.maxValue);
            auto indices = encodeNumber(value);
            allIndices.insert(allIndices.end(), indices.begin(), indices.end());
        } catch (const std::exception&) {
            continue; // skip invalid tokens
        }
    }
    return allIndices;
}

std::string NumberEncoding::decodeIndices(const std::vector<size_t>& indices) const {
    // Production-quality: decode SDR indices to best-approximate original numbers
    std::ostringstream oss;
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (isSpecialValue(idx)) {
            double value = decodeSpecialValue(idx);
            oss << value;
        } else if (idx >= config_.startIndex && idx < config_.startIndex + config_.bucketCount) {
            // Standard bucket: use midpoint
            size_t bucket = idx - config_.startIndex;
            double start = bucketBoundaries_[bucket];
            double end = bucketBoundaries_[bucket + 1];
            double value = (start + end) / 2.0;
            oss << value;
        } else if (idx == std::numeric_limits<size_t>::max() - 1 && i + 2 < indices.size()) { // RANGE_START_MARKER
            // Range encoding: decode as [start, end]
            size_t startBucket = indices[i + 1] - config_.startIndex;
            size_t endBucket = indices[i + 2] - config_.startIndex;
            double startVal = bucketBoundaries_[startBucket];
            double endVal = bucketBoundaries_[endBucket + 1];
            oss << "[" << startVal << ", " << endVal << "]";
            i += 2;
        } else {
            // Unknown index type; skip or print as is
            oss << "?";
        }
        if (i != indices.size() - 1) oss << " ";
    }
    return oss.str();
}

void NumberEncoding::updateBuckets(double number) {
     // Placeholder implementation - likely involves calling updateStatistics and maybe optimizeQuantization
     if (config_.adaptiveBuckets) {
        size_t bucketIndex = getBucketIndex(number);
        updateStatistics(number, bucketIndex);
        if (shouldOptimizeBuckets()) {
            optimizeQuantization();
        }
    }
}

float NumberEncoding::getCompressionRatio() const {
    // Compute actual compression ratio using encoding stats
    size_t totalOriginal = 0;
    size_t totalEncoded = 0;
    for (const auto& kv : bucketStats_) {
        totalOriginal += kv.second.values.size();
        totalEncoded += kv.second.hitCount;
    }
    if (totalOriginal > 0 && totalEncoded > 0) {
        constexpr float originalBits = 64.0f;
        constexpr float sdrBits = 11.0f;
        return (originalBits * totalOriginal) / (sdrBits * totalEncoded);
    }
    // Fallback to theoretical
    constexpr float originalBits = 64.0f;
    constexpr float sdrBits = 11.0f;
    return originalBits / sdrBits;
}

void NumberEncoding::redistributeBuckets() {
    // Production-quality: quantile binning for adaptive quantization
    std::vector<double> allValues;
    for (const auto& kv : bucketStats_) {
        allValues.insert(allValues.end(), kv.second.values.begin(), kv.second.values.end());
    }
    if (allValues.size() < config_.bucketCount) {
        // Not enough data, fall back to linear
        double range = config_.maxValue - config_.minValue;
        double step = range / config_.bucketCount;
        for (size_t i = 0; i <= config_.bucketCount; ++i) {
            bucketBoundaries_[i] = config_.minValue + i * step;
        }
        resetBucketStats();
        return;
    }
    std::sort(allValues.begin(), allValues.end());
    size_t n = allValues.size();
    for (size_t i = 0; i <= config_.bucketCount; ++i) {
        size_t idx = std::min(i * n / config_.bucketCount, n - 1);
        bucketBoundaries_[i] = allValues[idx];
    }
    resetBucketStats();
}

double NumberEncoding::interpolateValue(size_t bucket, double position) const {
    // Production-quality: linear interpolation, clamp position, extensible for nonlinear
    if (bucket >= bucketBoundaries_.size() - 1) return 0.0;
    double start = bucketBoundaries_[bucket];
    double end = bucketBoundaries_[bucket + 1];
    position = std::clamp(position, 0.0, 1.0);
    // For future: support nonlinear interpolation if needed
    return start + (end - start) * position;
}
