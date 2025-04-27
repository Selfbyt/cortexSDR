#include "WordEncoding.hpp"
#include <iostream> 
#include <algorithm> 
#include <string>

// Default constructor - initializes empty vocabulary and map
WordEncoding::WordEncoding() {
    // No explicit initialization needed for vector and map members
}

// Encodes a single word. If the word is new, adds it to the vocabulary.
// Now non-const because it modifies the internal vocabulary state.
std::vector<size_t> WordEncoding::encodeWord(const std::string& word) {
    std::vector<size_t> indices;
    std::string normWord = word;
    // Normalize: lowercase and strip punctuation
    std::transform(normWord.begin(), normWord.end(), normWord.begin(), ::tolower);
    normWord.erase(std::remove_if(normWord.begin(), normWord.end(), [](char c) { return !std::isalnum(c); }), normWord.end());
    if (normWord.empty()) return indices;
    auto it = wordToIndex_.find(normWord);
    if (it != wordToIndex_.end()) {
        indices.push_back(it->second);
    } else {
        size_t newIndex = vocabulary_.size();
        vocabulary_.push_back(normWord);
        wordToIndex_[normWord] = newIndex;
        indices.push_back(newIndex);
    }
    return indices;
}

// Encodes an entire text string: splits into normalized words, encodes each
std::vector<size_t> WordEncoding::encodeText(const std::string& text) {
    std::vector<size_t> indices;
    std::string word;
    for (char c : text) {
        if (std::isalnum(c)) {
            word += std::tolower(c);
        } else if (!word.empty()) {
            auto idx = encodeWord(word);
            indices.insert(indices.end(), idx.begin(), idx.end());
            word.clear();
        }
    }
    if (!word.empty()) {
        auto idx = encodeWord(word);
        indices.insert(indices.end(), idx.begin(), idx.end());
    }
    return indices;
}

// Decodes a vector of indices back into a string of words.
// Remains const as it doesn't modify the state.
std::string WordEncoding::decodeIndices(const std::vector<size_t>& indices) const {
    std::string result;
    for (size_t index : indices) {
        if (index < vocabulary_.size()) {
            if (!result.empty()) {
                result += " ";
            }
            result += vocabulary_[index];
        } else {
            // Handle invalid index: insert placeholder
            if (!result.empty()) result += " ";
            result += "<UNK>";
        }
    }
    return result;
}

// Getter for the vocabulary vector
const std::vector<std::string>& WordEncoding::getVocabulary() const {
    return vocabulary_;
}

// Removed getWordToIndexMap implementation as it's no longer needed

// Setter to restore vocabulary state (used during deserialization)
// Rebuilds the wordToIndex_ map internally.
void WordEncoding::setVocabulary(const std::vector<std::string>& vocab) {
    vocabulary_ = vocab;
    wordToIndex_.clear(); // Clear the existing map
    for (size_t i = 0; i < vocabulary_.size(); ++i) {
        wordToIndex_[vocabulary_[i]] = i; // Rebuild the map
    }
}
