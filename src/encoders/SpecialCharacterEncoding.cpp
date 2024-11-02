#include "SpecialCharacterEncoding.hpp"

std::vector<size_t> SpecialCharacterEncoding::encodeText(const std::string& text) const {
    std::vector<size_t> indices;
    for (char c : text) {
        if (!std::isalnum(c)) {
            // Encode special characters to indices starting at 1000
            indices.push_back(1000 + static_cast<size_t>(c));
        }
    }
    return indices;
} 