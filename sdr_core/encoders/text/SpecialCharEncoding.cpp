#include "SpecialCharEncoding.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <cctype>

SpecialCharEncoding::SpecialCharEncoding() {
    // Populate with common ASCII special chars and symbols
    const std::string specials = "!@#$%^&*()_+-=[]{}`~|;:'\"<>,.?/\\";
    for (char c : specials) {
        if (specialToIndex_.count(c) == 0) {
            size_t idx = specialToIndex_.size();
            specialToIndex_[c] = idx;
            indexToSpecial_[idx] = c;
        }
    }
}

std::vector<size_t> SpecialCharEncoding::encode(const std::string& text) const {
    std::vector<size_t> indices;
    for (char c : text) {
        auto it = specialToIndex_.find(c);
        if (it != specialToIndex_.end()) {
            indices.push_back(it->second);
        }
    }
    return indices;
}

std::string SpecialCharEncoding::decode(const std::vector<size_t>& indices) const {
    std::string result;
    for (size_t idx : indices) {
        auto it = indexToSpecial_.find(idx);
        if (it != indexToSpecial_.end()) {
            result += it->second;
        } else {
            result += '?';
        }
    }
    return result;
}
