#ifndef WORD_ENCODING_HPP
#define WORD_ENCODING_HPP

#include <string>
#include <unordered_map>

class WordEncoding {
public:
    WordEncoding(std::initializer_list<std::string> vocabulary);

    size_t encodeWord(const std::string& word) const;

private:
    std::unordered_map<std::string, size_t> wordIndices_;
};

#endif // WORD_ENCODING_HPP
