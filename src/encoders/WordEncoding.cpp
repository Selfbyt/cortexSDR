#include "WordEncoding.hpp"

WordEncoding::WordEncoding(std::initializer_list<std::string> vocabulary)
    : vocabulary_(vocabulary) {
    
    size_t index = 0;
    for (const auto& word : vocabulary_) {
        wordToIndex_[word] = index++;
    }
}

std::vector<size_t> WordEncoding::encodeWord(const std::string& text) const {
    std::vector<size_t> indices;
    auto it = wordToIndex_.find(text);
    if (it != wordToIndex_.end()) {
        indices.push_back(it->second);
    }
    return indices;
} 