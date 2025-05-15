#ifndef WORD_ENCODING_HPP
#define WORD_ENCODING_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <vector>
#include <unordered_map>

class WordEncoding {
public:
    WordEncoding(); // Default constructor

    // encodeWord is now non-const as it modifies the internal state
    std::vector<size_t> encodeWord(const std::string& text);
    std::vector<size_t> encodeText(const std::string& text);
    std::string decodeIndices(const std::vector<size_t>& indices) const;

    // Methods to get/set vocabulary state for serialization/deserialization
    const std::vector<std::string>& getVocabulary() const;
    // No need to expose map getter if setter rebuilds it
    // const std::unordered_map<std::string, size_t>& getWordToIndexMap() const; 
    void setVocabulary(const std::vector<std::string>& vocab); // Only needs the vector

private:
    std::vector<std::string> vocabulary_; // Stores words as they are encountered
    std::unordered_map<std::string, size_t> wordToIndex_; // Maps words to their indices
};

#endif // WORD_ENCODING_HPP
