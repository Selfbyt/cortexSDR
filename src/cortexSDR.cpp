#include "cortexSDR.hpp"

SparseDistributedRepresentation::SparseDistributedRepresentation(std::initializer_list<std::string> vocabulary)
    : vocabulary_(vocabulary),
    currentEncoding_({}, EncodingRanges::MAX_VECTOR_SIZE),
    wordEncoder_(std::make_unique<WordEncoding>(vocabulary)),
    dateTimeEncoder_(std::make_unique<DateTimeEncoding>(DateTimeEncoding::EncodingConfig{})),
    specialCharEncoder_(std::make_unique<SpecialCharacterEncoding>()),
    numberEncoder_(std::make_unique<NumberEncoding>(NumberEncoding::EncodingConfig{})),
    imageEncoder_(std::make_unique<ImageEncoding>()),
    videoEncoder_(std::make_unique<VideoEncoding>()),
    audioEncoder_(std::make_unique<AudioEncoding>())
{
}

SparseDistributedRepresentation::EncodedData SparseDistributedRepresentation::encodeText(const std::string &text)
{
    resetEncodedVector();
    std::vector<std::string> tokens = tokenizeText(text);

    // Initialize empty combined encoding
    EncodedData combinedEncoding({}, EncodingRanges::MAX_VECTOR_SIZE);

    for (const auto &token : tokens)
    {
        // Try to encode as a number first
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

        // Handle special characters
        if (token.length() == 1 && !std::isalnum(token[0])) {
            auto specialCharIndices = specialCharEncoder_->encodeText(token);
            setIndices(specialCharIndices);
            continue;
        }

        // Encode regular text using character-level encoding
        auto charIndices = wordEncoder_->encodeWord(token);
        setIndices(charIndices);
    }

    currentEncoding_ = getEncodedData();
    // No need to combine with empty encoding
    // currentEncoding_ |= combinedEncoding;
    return currentEncoding_;
}

SparseDistributedRepresentation::EncodedData SparseDistributedRepresentation::encodeNumber(double number)
{
    resetEncodedVector();
    auto indices = numberEncoder_->encodeNumber(number);
    setIndices(indices);
    currentEncoding_ = getEncodedData();
    return currentEncoding_;
}

std::string SparseDistributedRepresentation::decode() const
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

void SparseDistributedRepresentation::printStats() const
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

void SparseDistributedRepresentation::validateVocabularySize() const {
    if (vocabulary_.size() > EncodingRanges::WORD_END - EncodingRanges::WORD_START + 1) {
        throw std::runtime_error("Vocabulary size exceeds available word encoding range");
    }
}

void SparseDistributedRepresentation::initializeWordMap() {
    for (size_t i = 0; i < vocabulary_.size(); ++i) {
        std::string word = vocabulary_[i];
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        wordToIndex_[word] = i;
    }
}

std::vector<std::string> SparseDistributedRepresentation::tokenizeText(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string currentToken;

    for (char c : text) {
        if (std::isspace(c) || std::ispunct(c)) {
            if (!currentToken.empty()) {
                tokens.push_back(currentToken);
                currentToken.clear();
            }
            if (std::ispunct(c)) {
                tokens.push_back(std::string(1, c));
            }
        } else {
            currentToken += std::tolower(c);
        }
    }

    if (!currentToken.empty()) {
        tokens.push_back(currentToken);
    }

    return tokens;
}

void SparseDistributedRepresentation::resetEncodedVector() {
    encodedVector_.reset();
}

void SparseDistributedRepresentation::setIndices(const std::vector<size_t>& indices) {
    for (size_t index : indices) {
        if (index < EncodingRanges::MAX_VECTOR_SIZE) {
            encodedVector_.set(index);
        }
    }
}

SparseDistributedRepresentation::EncodedData SparseDistributedRepresentation::getEncodedData() const {
    EncodedData data;
    data.totalSize = EncodingRanges::MAX_VECTOR_SIZE;
    for (size_t i = 0; i < encodedVector_.size(); ++i) {
        if (encodedVector_[i]) {
            data.activePositions.push_back(i);
        }
    }
    return data;
}

SparseDistributedRepresentation::SeparatedIndices
SparseDistributedRepresentation::separateIndices(const EncodedData& data) const {
    SeparatedIndices result;

    for (size_t pos : data.activePositions) {
        if (pos <= EncodingRanges::WORD_END) {
            result.wordIndices.push_back(pos);
        } else if (pos <= EncodingRanges::SPECIAL_CHAR_END) {
            result.specialCharIndices.push_back(pos);
        } else if (pos <= EncodingRanges::NUMBER_END) {
            result.numberIndices.push_back(pos);
        }
    }

    return result;
}

std::string SparseDistributedRepresentation::joinComponents(
    const std::vector<std::string>& components) const {
    std::string result;
    for (size_t i = 0; i < components.size(); ++i) {
        if (i > 0 && !components[i].empty() && !components[i - 1].empty()) {
            result += " ";
        }
        result += components[i];
    }
    return result;
}

double SparseDistributedRepresentation::calculateSparsity(const EncodedData& data) const {
    return 100.0 * (1.0 - static_cast<double>(data.activePositions.size()) /
                              EncodingRanges::MAX_VECTOR_SIZE);
}

// These implementations are placeholders and not needed for basic functionality
/*
void BrainInspiredSDR::optimizePatterns() {
    // Implementation removed for simplicity
}
*/

/*
void AdaptiveEncoder::learnPatterns(const std::string& text) {
    // Implementation removed for simplicity
}
*/

/*
EncodedData ContextualSDR::encodeWithContext(const std::string& text) {
    // Implementation removed for simplicity
    return EncodedData({}, EncodingRanges::MAX_VECTOR_SIZE);
}
*/

/*
void SemanticEncoder::clusterRelatedConcepts() {
    // Implementation removed for simplicity
}
*/

/*
EncodedData PredictiveEncoder::encode(const std::string& text) {
    // Implementation removed for simplicity
    return EncodedData({}, EncodingRanges::MAX_VECTOR_SIZE);
}
*/

/*
void BrainLikeCompression::strengthenConnections(const EncodedData& pattern) {
    // Implementation removed for simplicity
}
*/
