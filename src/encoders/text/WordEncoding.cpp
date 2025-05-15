#include "WordEncoding.hpp"
#include <iostream>
#include <algorithm>
#include <cstdint>  // For uint32_t
#include <string>
#include <cctype>

// Default constructor - initializes empty vocabulary and map
WordEncoding::WordEncoding() {
    // No explicit initialization needed for vector and map members
}

// Encodes a single word. If the word is new, adds it to the vocabulary.
// Now non-const because it modifies the internal vocabulary state.
// Simple, direct fingerprint encoding: use word index directly
std::vector<size_t> WordEncoding::encodeWord(const std::string& word) {
    constexpr size_t K = 8; // Number of active bits per word
    std::vector<size_t> indices;
    std::string normWord = word;
    std::transform(normWord.begin(), normWord.end(), normWord.begin(), ::tolower);
    normWord.erase(std::remove_if(normWord.begin(), normWord.end(), [](char c) { return !std::isalnum(c); }), normWord.end());
    if (normWord.empty()) return indices;
    
    // Add word to vocabulary if not present
    if (wordToIndex_.find(normWord) == wordToIndex_.end()) {
        wordToIndex_[normWord] = vocabulary_.size();
        vocabulary_.push_back(normWord); // Always store normalized
    }
    
    // Get the word's index in the vocabulary
    size_t wordIndex = wordToIndex_[normWord];
    
    // CRITICAL: Use a very simple, direct encoding scheme
    // Each word gets a unique range of K consecutive indices starting at wordIndex*K
    // This ensures no overlap between different words' fingerprints
    for (size_t i = 0; i < K; ++i) {
        // Generate indices in the range [0, 9999] to stay within word region
        size_t idx = (wordIndex * K + i) % 10000;
        indices.push_back(idx);
    }
    
    // Debug output for first few words
    if (wordIndex < 5) {
        std::cout << "[DEBUG-ENCODE] Word '" << normWord << "' (index " << wordIndex << ") fingerprint: ";
        for (size_t i = 0; i < indices.size(); ++i) {
            std::cout << indices[i] << " ";
        }
        std::cout << std::endl;
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
// Robust fingerprint decoding: for each K-bit set, find word with maximal overlap
std::string WordEncoding::decodeIndices(const std::vector<size_t>& indices) const {
    constexpr size_t K = 8;
    std::string result;
    if (vocabulary_.empty() || indices.empty()) return result;
    // Build reverse map: for each word, precompute its fingerprint using same logic as encodeWord
    std::vector<std::vector<size_t>> fingerprints;
    for (size_t wordIndex = 0; wordIndex < vocabulary_.size(); ++wordIndex) {
        std::vector<size_t> bits;
        // Use the same simple, direct encoding scheme as in encodeWord
        for (size_t i = 0; i < K; ++i) {
            // Generate indices in the range [0, 9999] to stay within word region
            size_t idx = (wordIndex * K + i) % 10000;
            bits.push_back(idx);
        }
        fingerprints.push_back(bits);
        
        // Debug output for first few words
        if (wordIndex < 5) {
            std::cout << "[DEBUG-DECODE] Word '" << vocabulary_[wordIndex] << "' (index " << wordIndex << ") fingerprint: ";
            for (size_t i = 0; i < bits.size(); ++i) {
                std::cout << bits[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    // Debug: print first 5 windows and their best overlaps
    for (size_t i = 0; i + K <= indices.size() && i < 5 * K; i += K) {
        int best = -1, bestOverlap = -1;
        for (size_t w = 0; w < fingerprints.size(); ++w) {
            int overlap = 0;
            for (size_t k = 0; k < K; ++k) {
                for (size_t j = 0; j < K; ++j) {
                    if (indices[i+k] == fingerprints[w][j]) ++overlap;
                }
            }
            if (overlap > bestOverlap) {
                best = w;
                bestOverlap = overlap;
            }
        }
        std::cout << "[DEBUG] Window " << (i/K) << ": bestOverlap=" << bestOverlap << ", bestWord=";
        if (best != -1 && bestOverlap > K/2) std::cout << vocabulary_[best] << std::endl;
        else std::cout << "<UNK>" << std::endl;
    }
    // SIMPLIFIED DECODING APPROACH
    // With our new direct encoding scheme, we can directly map indices to words
    // Create a map from index to word
    std::unordered_map<size_t, size_t> indexToWord;
    
    // For each word in our vocabulary
    for (size_t wordIndex = 0; wordIndex < vocabulary_.size(); ++wordIndex) {
        // Each word has K consecutive indices
        for (size_t i = 0; i < K; ++i) {
            size_t idx = (wordIndex * K + i) % 10000;
            indexToWord[idx] = wordIndex;
        }
    }
    
    // Count occurrences of each word in the input indices
    std::unordered_map<size_t, int> wordCounts;
    for (size_t idx : indices) {
        auto it = indexToWord.find(idx);
        if (it != indexToWord.end()) {
            wordCounts[it->second]++;
        }
    }
    
    // Sort words by frequency (most frequent first)
    std::vector<std::pair<size_t, int>> sortedWords;
    for (const auto& pair : wordCounts) {
        sortedWords.push_back(pair);
    }
    std::sort(sortedWords.begin(), sortedWords.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Build result string with the most frequent words
    for (size_t i = 0; i < sortedWords.size(); ++i) {
        // Only include words that have at least 2 matching indices
        if (sortedWords[i].second >= 2) {
            if (!result.empty()) result += " ";
            result += vocabulary_[sortedWords[i].first];
        }
    }
    
    // If we didn't match anything, return <UNK>
    if (result.empty()) {
        result = "<UNK>";
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
    vocabulary_.clear();
    wordToIndex_.clear();
    for (const auto& word : vocab) {
        std::string normWord = word;
        std::transform(normWord.begin(), normWord.end(), normWord.begin(), ::tolower);
        normWord.erase(std::remove_if(normWord.begin(), normWord.end(), [](char c) { return !std::isalnum(c); }), normWord.end());
        if (!normWord.empty() && wordToIndex_.find(normWord) == wordToIndex_.end()) {
            wordToIndex_[normWord] = vocabulary_.size();
            vocabulary_.push_back(normWord);
        }
    }
}
