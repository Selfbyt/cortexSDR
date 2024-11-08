#ifndef VIDEO_ENCODING_HPP
#define VIDEO_ENCODING_HPP

#include <string>
#include <vector>

class VideoEncoding {
public:
    std::vector<size_t> encodeVideo(const std::string& videoPath) const;
    std::string decodeIndices(const std::vector<size_t>& indices) const;
};

#endif // VIDEO_ENCODING_HPP 