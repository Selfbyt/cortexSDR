#ifndef NUMBER_ENCODING_HPP
#define NUMBER_ENCODING_HPP

#include <vector>
#include <string> 
#include <cstddef>

class NumberEncoding {
public:
    NumberEncoding(size_t startIndex = 1000, 
                  size_t bucketCount = 100,
                  double minValue = -1000.0,
                  double maxValue = 1000.0);

    // Encode a single number
    std::vector<size_t> encodeNumber(double number) const;
    
    // Encode multiple numbers from a string (e.g., "42.5 -123.4 789")
    std::vector<size_t> encodeNumberString(const std::string& numbers) const;

private:
    size_t startIndex_;      // Starting index in the SDR for number encoding
    size_t bucketCount_;     // Number of buckets for value discretization
    double minValue_;        // Minimum value in range
    double maxValue_;        // Maximum value in range
    double bucketSize_;      // Size of each bucket

    size_t getBucketIndex(double value) const;
};

#endif // NUMBER_ENCODING_HPP 