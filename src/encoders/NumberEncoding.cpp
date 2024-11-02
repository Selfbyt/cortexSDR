#include "NumberEncoding.hpp"
#include <sstream>
#include <cmath>

NumberEncoding::NumberEncoding(size_t startIndex, size_t bucketCount, 
                             double minValue, double maxValue)
    : startIndex_(startIndex)
    , bucketCount_(bucketCount)
    , minValue_(minValue)
    , maxValue_(maxValue) {
    bucketSize_ = (maxValue - minValue) / bucketCount;
}

std::vector<size_t> NumberEncoding::encodeNumber(double number) const {
    std::vector<size_t> indices;
    
    // Clamp the number to our range
    number = std::max(minValue_, std::min(maxValue_, number));
    
    // Get the bucket index
    size_t bucketIdx = getBucketIndex(number);
    
    // Add the main bucket index
    indices.push_back(startIndex_ + bucketIdx);
    
    // Add adjacent buckets for smoother representation
    if (bucketIdx > 0) {
        indices.push_back(startIndex_ + bucketIdx - 1);
    }
    if (bucketIdx < bucketCount_ - 1) {
        indices.push_back(startIndex_ + bucketIdx + 1);
    }
    
    return indices;
}

std::vector<size_t> NumberEncoding::encodeNumberString(const std::string& numbers) const {
    std::vector<size_t> indices;
    std::istringstream iss(numbers);
    double number;
    
    while (iss >> number) {
        auto numberIndices = encodeNumber(number);
        indices.insert(indices.end(), numberIndices.begin(), numberIndices.end());
    }
    
    return indices;
}

size_t NumberEncoding::getBucketIndex(double value) const {
    size_t bucket = static_cast<size_t>((value - minValue_) / bucketSize_);
    return std::min(bucket, bucketCount_ - 1);
} 