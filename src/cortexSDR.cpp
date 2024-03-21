#include <iostream>
#include <bitset>
#include <vector>
#include <string>
#include "WordEncoding.hpp"
#include "DateTimeEncoding.hpp"
#include "SpecialCharacterEncoding.hpp"

class SparseDistributedRepresentation {
public:
    SparseDistributedRepresentation(std::initializer_list<std::string> vocabulary)
        : vocabulary_(vocabulary), vectorSize_(vocabulary.size()) {
        wordEncoder_ = WordEncoding(vocabulary);
    }

    void encodeText(const std::string& text) {
        std::bitset<MAX_VECTOR_SIZE> encodedVector;
        auto encodedText = specialCharEncoder_.encodeText(text);
        for (size_t index : encodedText) {
            if (index < vectorSize_) {
                encodedVector.set(index);
            }
        }
        encodedVector_ = encodedVector;
    }

private:
    static const size_t MAX_VECTOR_SIZE = 2000;
    std::bitset<MAX_VECTOR_SIZE> encodedVector_;
    std::vector<std::string> vocabulary_;
    size_t vectorSize_;
    WordEncoding wordEncoder_;
    SpecialCharacterEncoding specialCharEncoder_;
};
