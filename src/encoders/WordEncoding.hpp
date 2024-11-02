#ifndef WORD_ENCODING_HPP
#define WORD_ENCODING_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <initializer_list>

class WordEncoding {
public:
    WordEncoding(std::initializer_list<std::string> vocabulary);

    std::vector<size_t> encodeWord(const std::string& text) const;
    std::string decodeIndices(const std::vector<size_t>& indices) const;

private:
    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, size_t> wordToIndex_;
    std::vector<std::string> tokenizeText(const std::string& text) const;
};

#endif // WORD_ENCODING_HPP
