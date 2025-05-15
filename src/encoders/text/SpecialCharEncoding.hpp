#pragma once
#include <unordered_map>
#include <vector>
#include <string>

class SpecialCharEncoding {
public:
    SpecialCharEncoding();
    std::vector<size_t> encode(const std::string& text) const;
    std::string decode(const std::vector<size_t>& indices) const;
private:
    std::unordered_map<char, size_t> specialToIndex_;
    std::unordered_map<size_t, char> indexToSpecial_;
};
