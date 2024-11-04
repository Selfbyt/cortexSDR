#include "NumberEncoding.hpp"
#include <sstream>
#include <cmath>
#include <limits>
#include <iomanip>

NumberEncoding::NumberEncoding(size_t startIndex, size_t bucketCount, 
                             double minValue, double maxValue)
    : startIndex_(startIndex)
    , bucketCount_(bucketCount)
    , minValue_(minValue)
    , maxValue_(maxValue) {
    // Ensure we don't divide by zero if bucketCount is 1
    bucketSize_ = (bucketCount > 1) ? 
                 (maxValue - minValue) / static_cast<double>(bucketCount - 1) :
                 1.0;
}

std::vector<size_t> NumberEncoding::encodeNumber(double number) const {
    std::vector<size_t> indices;
    
    // Handle special floating point cases
    if (std::isnan(number)) {
        number = 0.0; // Map NaN to zero
    } else if (std::isinf(number)) {
        number = (number > 0) ? maxValue_ : minValue_;
    }
    
    // Clamp the number to our range using nextafter to handle edge cases
    number = std::max(minValue_, std::min(maxValue_, number));
    
    // Get the bucket index with proper rounding
    size_t bucketIdx = getBucketIndex(number);
    
    // Add the main bucket index
    indices.push_back(startIndex_ + bucketIdx);
    
    // Calculate the position within the bucket for interpolation
    double normalizedPos = (number - (minValue_ + bucketIdx * bucketSize_)) / bucketSize_;
    
    // Add adjacent buckets with weight based on position
    if (normalizedPos < 0.5 && bucketIdx > 0) {
        indices.push_back(startIndex_ + bucketIdx - 1);
    }
    if (normalizedPos >= 0.5 && bucketIdx < bucketCount_ - 1) {
        indices.push_back(startIndex_ + bucketIdx + 1);
    }
    
    return indices;
}

std::vector<size_t> NumberEncoding::encodeNumberString(const std::string& numbers) const {
    std::vector<size_t> indices;
    std::istringstream iss(numbers);
    std::string token;
    
    while (iss >> token) {
        try {
            double number = std::stod(token);
            auto numberIndices = encodeNumber(number);
            indices.insert(indices.end(), numberIndices.begin(), numberIndices.end());
        } catch (const std::invalid_argument&) {
            // Skip invalid numbers
            continue;
        } catch (const std::out_of_range&) {
            // Handle out of range by encoding as max/min value
            auto numberIndices = encodeNumber(token[0] == '-' ? minValue_ : maxValue_);
            indices.insert(indices.end(), numberIndices.begin(), numberIndices.end());
        }
    }
    
    return indices;
}

size_t NumberEncoding::getBucketIndex(double value) const {
    // Use proper rounding to nearest bucket
    double bucketDbl = (value - minValue_) / bucketSize_;
    size_t bucket = static_cast<size_t>(std::round(bucketDbl));
    
    // Handle floating point precision issues near boundaries
    if (std::abs(bucketDbl - std::round(bucketDbl)) < std::numeric_limits<double>::epsilon() * 100) {
        bucket = static_cast<size_t>(std::floor(bucketDbl));
    }
    
    return std::min(bucket, bucketCount_ - 1);
}

std::string NumberEncoding::decodeIndices(const std::vector<size_t>& indices) const {
    std::ostringstream result;
    result << std::fixed << std::setprecision(6); // Set reasonable precision for floating point output
    
    // Process indices in groups of up to 3 (main bucket + possible adjacent buckets)
    for (size_t i = 0; i < indices.size();) {
        std::vector<size_t> group;
        
        // Collect adjacent indices
        size_t currentIdx = indices[i];
        if (currentIdx >= startIndex_ && currentIdx < startIndex_ + bucketCount_) {
            // Find consecutive indices that could form a group
            while (i < indices.size() && group.size() < 3 && 
                   indices[i] >= startIndex_ && indices[i] < startIndex_ + bucketCount_) {
                group.push_back(indices[i] - startIndex_);
                i++;
            }
            
            // Find the middle (main) bucket
            size_t mainBucket;
            if (group.size() == 3) {
                // For three consecutive buckets, take the middle one
                mainBucket = group[1];
            } else if (group.size() == 2) {
                // For two buckets, interpolate between them
                double bucket1 = static_cast<double>(group[0]);
                double bucket2 = static_cast<double>(group[1]);
                mainBucket = static_cast<size_t>(std::round((bucket1 + bucket2) / 2.0));
            } else {
                // Single bucket
                mainBucket = group[0];
            }
            
            // Calculate the value from the main bucket with proper interpolation
            double value = minValue_ + (mainBucket * bucketSize_);
            
            // Add interpolation adjustment if we have adjacent buckets
            if (group.size() > 1) {
                double adjustment = 0.0;
                if (group.size() == 2) {
                    // Interpolate between two buckets
                    adjustment = (group[1] > group[0]) ? bucketSize_ * 0.25 : -bucketSize_ * 0.25;
                }
                value += adjustment;
            }
            
            if (result.tellp() > 0) result << " ";
            result << value;
        } else {
            i++; // Skip invalid indices
        }
    }
    return result.str();
}