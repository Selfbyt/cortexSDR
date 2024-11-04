#include <iostream>
#include <bitset>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include "WordEncoding.hpp"
#include "DateTimeEncoding.hpp"
#include "SpecialCharacterEncoding.hpp"
#include "NumberEncoding.hpp"

// Define encoding ranges
struct EncodingRanges
{
    static constexpr size_t WORD_START = 0;
    static constexpr size_t WORD_END = 999;
    static constexpr size_t SPECIAL_CHAR_START = 1000;
    static constexpr size_t SPECIAL_CHAR_END = 1499;
    static constexpr size_t NUMBER_START = 1500;
    static constexpr size_t NUMBER_END = 1999;
    static constexpr size_t MAX_VECTOR_SIZE = 2000;
};

class SparseDistributedRepresentation
{
public:
    struct EncodedData
    {
        std::vector<size_t> activePositions;
        size_t totalSize;

        // Constructors
        EncodedData() = default;
        EncodedData(std::vector<size_t> positions, size_t size)
            : activePositions(std::move(positions)), totalSize(size) {}

        // Get memory size
        size_t getMemorySize() const
        {
            return activePositions.size() * sizeof(size_t) + sizeof(totalSize);
        }

        // Union operator
        EncodedData &operator|=(const EncodedData &other)
        {
            std::vector<size_t> combined;
            std::set_union(
                activePositions.begin(), activePositions.end(),
                other.activePositions.begin(), other.activePositions.end(),
                std::back_inserter(combined));
            activePositions = std::move(combined);
            return *this;
        }

        // Calculate overlap
        double calculateOverlap(const EncodedData &other) const
        {
            std::vector<size_t> intersection;
            std::set_intersection(
                activePositions.begin(), activePositions.end(),
                other.activePositions.begin(), other.activePositions.end(),
                std::back_inserter(intersection));
            return static_cast<double>(intersection.size()) /
                   std::min(activePositions.size(), other.activePositions.size());
        }
    };

    explicit SparseDistributedRepresentation(std::initializer_list<std::string> vocabulary)
        : vocabulary_(vocabulary),
          currentEncoding_({}, EncodingRanges::MAX_VECTOR_SIZE),
          wordEncoder_(std::make_unique<WordEncoding>(vocabulary)),
          dateTimeEncoder_(std::make_unique<DateTimeEncoding>()),
          specialCharEncoder_(std::make_unique<SpecialCharacterEncoding>()),
          numberEncoder_(std::make_unique<NumberEncoding>(
              EncodingRanges::NUMBER_START,
              EncodingRanges::NUMBER_END - EncodingRanges::NUMBER_START + 1,
              -1000.0,
              1000.0))
    {
        validateVocabularySize();
        initializeWordMap();
    }

    EncodedData encodeText(const std::string &text)
    {
        resetEncodedVector();
        std::vector<std::string> tokens = tokenizeText(text);

        for (const auto &token : tokens)
        {
            // Try to encode as a word
            auto wordIndices = wordEncoder_->encodeWord(token);
            if (!wordIndices.empty())
            {
                setIndices(wordIndices);
                continue;
            }

            // Try to encode as a number
            try
            {
                double number = std::stod(token);
                auto numberIndices = numberEncoder_->encodeNumber(number);
                setIndices(numberIndices);
                continue;
            }
            catch (const std::invalid_argument &)
            {
            }

            // Encode special characters
            auto specialCharIndices = specialCharEncoder_->encodeText(token);
            setIndices(specialCharIndices);
        }

        currentEncoding_ = getEncodedData();
        return currentEncoding_;
    }

    EncodedData encodeNumber(double number)
    {
        resetEncodedVector();
        auto indices = numberEncoder_->encodeNumber(number);
        setIndices(indices);
        currentEncoding_ = getEncodedData();
        return currentEncoding_;
    }

    std::string decode() const
    {
        auto [wordIndices, specialCharIndices, numberIndices] = separateIndices(currentEncoding_);

        std::vector<std::string> components;

        if (!wordIndices.empty())
        {
            components.push_back(wordEncoder_->decodeIndices(wordIndices));
        }

        if (!specialCharIndices.empty())
        {
            components.push_back(specialCharEncoder_->decodeIndices(specialCharIndices));
        }

        if (!numberIndices.empty())
        {
            components.push_back(numberEncoder_->decodeIndices(numberIndices));
        }

        return joinComponents(components);
    }

    void printStats() const
    {
        double sparsity = calculateSparsity(currentEncoding_);
        auto [wordIndices, specialCharIndices, numberIndices] = separateIndices(currentEncoding_);

        std::cout << "\nSDR Statistics:\n";
        std::cout << "----------------\n";
        std::cout << "Active bits: " << currentEncoding_.activePositions.size()
                  << "/" << EncodingRanges::MAX_VECTOR_SIZE
                  << " (Sparsity: " << sparsity << "%)\n";

        std::cout << "Word encodings: " << wordIndices.size() << " active bits\n";
        std::cout << "Special characters: " << specialCharIndices.size() << " active bits\n";
        std::cout << "Number encodings: " << numberIndices.size() << " active bits\n";
        std::cout << "Memory usage: " << currentEncoding_.getMemorySize() << " bytes\n";
    }

private:
    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, size_t> wordToIndex_;
    EncodedData currentEncoding_;
    std::unique_ptr<WordEncoding> wordEncoder_;
    std::unique_ptr<DateTimeEncoding> dateTimeEncoder_;
    std::unique_ptr<SpecialCharacterEncoding> specialCharEncoder_;
    std::unique_ptr<NumberEncoding> numberEncoder_;
    std::bitset<EncodingRanges::MAX_VECTOR_SIZE> encodedVector_;

    void validateVocabularySize() const
    {
        if (vocabulary_.size() > EncodingRanges::WORD_END - EncodingRanges::WORD_START + 1)
        {
            throw std::runtime_error("Vocabulary size exceeds available word encoding range");
        }
    }

    void initializeWordMap()
    {
        for (size_t i = 0; i < vocabulary_.size(); ++i)
        {
            std::string word = vocabulary_[i];
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            wordToIndex_[word] = i;
        }
    }

    std::vector<std::string> tokenizeText(const std::string &text) const
    {
        std::vector<std::string> tokens;
        std::string currentToken;

        for (char c : text)
        {
            if (std::isspace(c) || std::ispunct(c))
            {
                if (!currentToken.empty())
                {
                    tokens.push_back(currentToken);
                    currentToken.clear();
                }
                if (std::ispunct(c))
                {
                    tokens.push_back(std::string(1, c));
                }
            }
            else
            {
                currentToken += std::tolower(c);
            }
        }

        if (!currentToken.empty())
        {
            tokens.push_back(currentToken);
        }

        return tokens;
    }

    void resetEncodedVector()
    {
        encodedVector_.reset();
    }

    void setIndices(const std::vector<size_t> &indices)
    {
        for (size_t index : indices)
        {
            if (index < EncodingRanges::MAX_VECTOR_SIZE)
            {
                encodedVector_.set(index);
            }
        }
    }

    EncodedData getEncodedData() const
    {
        EncodedData data;
        data.totalSize = EncodingRanges::MAX_VECTOR_SIZE;
        for (size_t i = 0; i < encodedVector_.size(); ++i)
        {
            if (encodedVector_[i])
            {
                data.activePositions.push_back(i);
            }
        }
        return data;
    }

    struct SeparatedIndices
    {
        std::vector<size_t> wordIndices;
        std::vector<size_t> specialCharIndices;
        std::vector<size_t> numberIndices;
    };

    SeparatedIndices separateIndices(const EncodedData &data) const
    {
        SeparatedIndices result;

        for (size_t pos : data.activePositions)
        {
            if (pos <= EncodingRanges::WORD_END)
            {
                result.wordIndices.push_back(pos);
            }
            else if (pos <= EncodingRanges::SPECIAL_CHAR_END)
            {
                result.specialCharIndices.push_back(pos);
            }
            else if (pos <= EncodingRanges::NUMBER_END)
            {
                result.numberIndices.push_back(pos);
            }
        }

        return result;
    }

    std::string joinComponents(const std::vector<std::string> &components) const
    {
        std::string result;
        for (size_t i = 0; i < components.size(); ++i)
        {
            if (i > 0 && !components[i].empty() && !components[i - 1].empty())
            {
                result += " ";
            }
            result += components[i];
        }
        return result;
    }

    double calculateSparsity(const EncodedData &data) const
    {
        return 100.0 * (1.0 - static_cast<double>(data.activePositions.size()) /
                                  EncodingRanges::MAX_VECTOR_SIZE);
    }
};