#ifndef SPECIAL_CHARACTER_ENCODING_HPP
#define SPECIAL_CHARACTER_ENCODING_HPP

#include <string>
#include <vector>

class SpecialCharacterEncoding {
public:
    std::vector<size_t> encodeText(const std::string& text) const;
};

#endif // SPECIAL_CHARACTER_ENCODING_HPP
