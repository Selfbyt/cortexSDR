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
    struct EncodedData {
        std::vector<size_t> activePositions;
        size_t totalSize;  // Store the total size for reconstruction
        
        // Calculate actual memory usage
        size_t getMemorySize() const {
            return activePositions.size() * sizeof(size_t) + sizeof(totalSize);
        }
    };

    SparseDistributedRepresentation(std::initializer_list<std::string> vocabulary)
        : vocabulary_(vocabulary), vectorSize_(vocabulary.size()),
          wordEncoder_(vocabulary),
          dateTimeEncoder_(),
          specialCharEncoder_(),
          numberEncoder_() {
    }

    EncodedData encodeText(const std::string& text) {
        std::bitset<MAX_VECTOR_SIZE> tempVector;
        
        // Encode words
        auto wordIndices = wordEncoder_.encodeWord(text);
        setIndices(tempVector, wordIndices);
        
        // Encode special characters
        auto specialCharIndices = specialCharEncoder_.encodeText(text);
        setIndices(tempVector, specialCharIndices);
        
        // Store the bitset internally
        encodedVector_ = tempVector;
        
        // Return only active positions
        return getEncodedData();
    }

    EncodedData encodeDateTime(const std::string& dateTime) {
        std::bitset<MAX_VECTOR_SIZE> tempVector;
        auto dateTimeIndices = dateTimeEncoder_.encodeDateTime(dateTime);
        setIndices(tempVector, dateTimeIndices);
        encodedVector_ = tempVector;
        return getEncodedData();
    }

    // Method to combine multiple encodings
    void combineEncodings(const std::vector<std::bitset<MAX_VECTOR_SIZE>>& encodings) {
        std::bitset<MAX_VECTOR_SIZE> combined;
        for (const auto& encoding : encodings) {
            combined |= encoding;
        }
        encodedVector_ = combined;
    }

    EncodedData encodeNumber(double number) {
        std::bitset<MAX_VECTOR_SIZE> tempVector;
        auto numberIndices = numberEncoder_.encodeNumber(number);
        setIndices(tempVector, numberIndices);
        encodedVector_ = tempVector;
        return getEncodedData();
    }

    EncodedData encodeNumberString(const std::string& numbers) {
        std::bitset<MAX_VECTOR_SIZE> tempVector;
        auto numberIndices = numberEncoder_.encodeNumberString(numbers);
        setIndices(tempVector, numberIndices);
        encodedVector_ = tempVector;
        return getEncodedData();
    }

    EncodedData getEncodedData() const {
        EncodedData data;
        data.totalSize = MAX_VECTOR_SIZE;
        
        // Store only active bit positions
        for(size_t i = 0; i < MAX_VECTOR_SIZE; i++) {
            if(encodedVector_[i]) {
                data.activePositions.push_back(i);
            }
        }
        
        return data;
    }

    static std::bitset<MAX_VECTOR_SIZE> reconstruct(const EncodedData& data) {
        std::bitset<MAX_VECTOR_SIZE> result;
        for(size_t pos : data.activePositions) {
            if(pos < MAX_VECTOR_SIZE) {
                result.set(pos);
            }
        }
        return result;
    }

    void printStats() const {
        auto data = getEncodedData();
        std::cout << "Active bits: " << data.activePositions.size() << "/" << MAX_VECTOR_SIZE 
                  << " (" << (static_cast<double>(data.activePositions.size()) / MAX_VECTOR_SIZE * 100)
                  << "%)\n";
        
        // Print first few active positions
        std::cout << "First active positions: ";
        for(size_t i = 0; i < std::min(size_t(5), data.activePositions.size()); ++i) {
            std::cout << data.activePositions[i] << " ";
        }
        std::cout << "\n";
        
        // Print memory usage
        std::cout << "Memory usage: " << data.getMemorySize() << " bytes\n";
    }

    // Decode the current encoded vector
    std::string decode() const {
        std::string result;
        
        // Get active positions
        auto data = getEncodedData();
        
        // Separate indices by their ranges
        std::vector<size_t> wordIndices, specialCharIndices, numberIndices, dateTimeIndices;
        
        for (size_t pos : data.activePositions) {
            if (pos < 1000) {  // Word indices
                wordIndices.push_back(pos);
            } else if (pos >= 1000 && pos < 1500) {  // Special characters
                specialCharIndices.push_back(pos);
            } else if (pos >= 1500 && pos < 2000) {  // Numbers
                numberIndices.push_back(pos);
            }
        }
        
        // Decode words
        if (!wordIndices.empty()) {
            result += wordEncoder_.decodeIndices(wordIndices);
        }
        
        // Decode special characters
        if (!specialCharIndices.empty()) {
            result += specialCharEncoder_.decodeIndices(specialCharIndices);
        }
        
        // Decode numbers (if present)
        if (!numberIndices.empty()) {
            if (!result.empty()) result += " ";
            result += numberEncoder_.decodeIndices(numberIndices);
        }
        
        return result;
    }
    
    // Decode from EncodedData
    std::string decodeFromData(const EncodedData& data) {
        encodedVector_ = reconstruct(data);
        return decode();
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
