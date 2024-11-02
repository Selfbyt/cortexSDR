#include <iostream>
#include <bitset>
#include <vector>
#include <string>
#include "WordEncoding.hpp"
#include "DateTimeEncoding.hpp"
#include "SpecialCharacterEncoding.hpp"
#include "NumberEncoding.hpp"

static const size_t MAX_VECTOR_SIZE = 2000;

class SparseDistributedRepresentation {
public:
    SparseDistributedRepresentation(std::initializer_list<std::string> vocabulary)
        : vocabulary_(vocabulary), vectorSize_(vocabulary.size()),
          wordEncoder_(vocabulary),
          dateTimeEncoder_(),
          specialCharEncoder_(),
          numberEncoder_() {
    }

    void encodeText(const std::string& text) {
        std::bitset<MAX_VECTOR_SIZE> encodedVector;
        
        // Encode words
        auto wordIndices = wordEncoder_.encodeWord(text);
        setIndices(encodedVector, wordIndices);
        
        // Encode special characters
        auto specialCharIndices = specialCharEncoder_.encodeText(text);
        setIndices(encodedVector, specialCharIndices);
        
        encodedVector_ = encodedVector;
    }

    void encodeDateTime(const std::string& dateTime) {
        std::bitset<MAX_VECTOR_SIZE> encodedVector;
        auto dateTimeIndices = dateTimeEncoder_.encodeDateTime(dateTime);
        setIndices(encodedVector, dateTimeIndices);
        encodedVector_ = encodedVector;
    }

    // Method to combine multiple encodings
    void combineEncodings(const std::vector<std::bitset<MAX_VECTOR_SIZE>>& encodings) {
        std::bitset<MAX_VECTOR_SIZE> combined;
        for (const auto& encoding : encodings) {
            combined |= encoding;
        }
        encodedVector_ = combined;
    }

    void encodeNumber(double number) {
        std::bitset<MAX_VECTOR_SIZE> encodedVector;
        auto numberIndices = numberEncoder_.encodeNumber(number);
        setIndices(encodedVector, numberIndices);
        encodedVector_ = encodedVector;
    }

    void encodeNumberString(const std::string& numbers) {
        std::bitset<MAX_VECTOR_SIZE> encodedVector;
        auto numberIndices = numberEncoder_.encodeNumberString(numbers);
        setIndices(encodedVector, numberIndices);
        encodedVector_ = encodedVector;
    }

    const std::bitset<MAX_VECTOR_SIZE>& getEncodedVector() const {
        return encodedVector_;
    }

    size_t getActiveCount() const {
        return encodedVector_.count();
    }

    void printStats() const {
        std::cout << "Active bits: " << getActiveCount() << "/" << MAX_VECTOR_SIZE 
                  << " (" << (static_cast<double>(getActiveCount()) / MAX_VECTOR_SIZE * 100)
                  << "%)\n";
        
        // Print first few active positions
        std::cout << "First active positions: ";
        int shown = 0;
        for (size_t i = 0; i < MAX_VECTOR_SIZE && shown < 5; ++i) {
            if (encodedVector_[i]) {
                std::cout << i << " ";
                shown++;
            }
        }
        std::cout << "\n";
    }

private:
    void setIndices(std::bitset<MAX_VECTOR_SIZE>& vector, const std::vector<size_t>& indices) {
        for (size_t index : indices) {
            if (index < MAX_VECTOR_SIZE) {
                vector.set(index);
            }
        }
    }

    std::bitset<MAX_VECTOR_SIZE> encodedVector_;
    std::vector<std::string> vocabulary_;
    size_t vectorSize_;
    WordEncoding wordEncoder_;
    DateTimeEncoding dateTimeEncoder_;
    SpecialCharacterEncoding specialCharEncoder_;
    NumberEncoding numberEncoder_;
};
