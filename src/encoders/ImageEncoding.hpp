#ifndef IMAGE_ENCODING_HPP
#define IMAGE_ENCODING_HPP

#include <string>
#include <vector>

class ImageEncoding {
public:
    std::vector<size_t> encodeImage(const std::string& imagePath) const;
    std::string decodeIndices(const std::vector<size_t>& indices) const;
};

#endif // IMAGE_ENCODING_HPP 