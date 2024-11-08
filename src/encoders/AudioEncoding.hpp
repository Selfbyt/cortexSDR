#ifndef AUDIO_ENCODING_HPP
#define AUDIO_ENCODING_HPP

#include <string>
#include <vector>

class AudioEncoding {
public:
    std::vector<size_t> encodeAudio(const std::string& audioPath) const;
    std::string decodeIndices(const std::vector<size_t>& indices) const;
};

#endif // AUDIO_ENCODING_HPP 