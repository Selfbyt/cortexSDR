#include "cortexSDR.hpp"

// Constructor no longer takes vocabulary
void SparseDistributedRepresentation::setSparsity(double sparsity) {
    if (sparsity < 0.0) sparsity_ = 0.0;
    else if (sparsity > 1.0) sparsity_ = 1.0;
    else sparsity_ = sparsity;
}

double SparseDistributedRepresentation::getSparsity() const {
    return sparsity_;
}

SparseDistributedRepresentation::SparseDistributedRepresentation() 
    : currentEncoding_({}, EncodingRanges::MAX_VECTOR_SIZE),
      sparsity_(0.002),
    wordEncoder_(std::make_unique<WordEncoding>()),
    dateTimeEncoder_(std::make_unique<DateTimeEncoding>(DateTimeEncoding::EncodingConfig{})),
    geoEncoder_(std::make_unique<GeoEncodingAdapter>()),
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

        // Handle special characters (SDR region)
        if (token.length() == 1 && !std::isalnum(token[0])) {
            // Offset indices to SDR region
            std::vector<size_t> specialCharIndices{static_cast<size_t>(token[0])};
            for (auto& idx : specialCharIndices) idx += EncodingRanges::SPECIAL_CHAR_START;
            setIndices(specialCharIndices);
            continue;
        }

        // Geo encoding: detect lat,lon pattern (basic)
        if (token.find(",") != std::string::npos) {
            size_t comma = token.find(",");
            try {
                double lat = std::stod(token.substr(0, comma));
                double lon = std::stod(token.substr(comma + 1));
                auto geoIndices = geoEncoder_->encode(lat, lon);
                // Offset geo indices to the defined region
                for (auto& idx : geoIndices) idx += EncodingRanges::GEO_START;
                setIndices(geoIndices);
                continue;
            } catch (...) {}
        }

        // Encode regular text using word encoding
        auto wordIndices = wordEncoder_->encodeWord(token);
        setIndices(wordIndices);
    }

    // Enforce sparsity by randomly dropping active bits if needed
    EncodedData raw = getEncodedData();
    size_t desiredActive = static_cast<size_t>(sparsity_ * EncodingRanges::MAX_VECTOR_SIZE);
    if (desiredActive > 0 && raw.activePositions.size() > desiredActive) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(raw.activePositions.begin(), raw.activePositions.end(), g);
        raw.activePositions.resize(desiredActive);
        std::sort(raw.activePositions.begin(), raw.activePositions.end());
    }
    currentEncoding_ = raw;
    return currentEncoding_;
}

SparseDistributedRepresentation::EncodedData SparseDistributedRepresentation::encodeNumber(double number)
{
    resetEncodedVector();
    auto indices = numberEncoder_->encodeNumber(number);
    setIndices(indices);
    // Enforce sparsity for numbers as well
    EncodedData raw = getEncodedData();
    size_t desiredActive = static_cast<size_t>(sparsity_ * EncodingRanges::MAX_VECTOR_SIZE);
    if (desiredActive > 0 && raw.activePositions.size() > desiredActive) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(raw.activePositions.begin(), raw.activePositions.end(), g);
        raw.activePositions.resize(desiredActive);
        std::sort(raw.activePositions.begin(), raw.activePositions.end());
    }
    currentEncoding_ = raw;
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
        // Remove SDR region offset for decoding
        std::vector<std::string> adjIndices;
        for (auto idx : specialCharIndices) adjIndices.push_back(std::string(1, static_cast<char>(idx - EncodingRanges::SPECIAL_CHAR_START)));
        components.push_back(joinComponents(adjIndices));
    }

    if (!numberIndices.empty())
    {
        components.push_back(numberEncoder_->decodeIndices(numberIndices));
    }

    // Geo decoding: look for indices in the defined geo region
    std::vector<size_t> geoIndices;
    for (size_t idx : currentEncoding_.activePositions) {
        if (idx >= EncodingRanges::GEO_START && idx <= EncodingRanges::GEO_END) {
             geoIndices.push_back(idx - EncodingRanges::GEO_START);
        }
    }
    if (!geoIndices.empty()) {
        auto geo = geoEncoder_->decode(geoIndices);
        components.push_back("geo:" + std::to_string(geo.first) + "," + std::to_string(geo.second));
    }

    return joinComponents(components);
}

void SparseDistributedRepresentation::setEncoding(const EncodedData& data) {
    currentEncoding_ = data;
    // Optionally, could also update the internal bitset if needed for other operations,
    // but decode() primarily uses currentEncoding_.activePositions.
    // resetEncodedVector();
    // setIndices(data.activePositions);
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

// Removed validateVocabularySize() and initializeWordMap() as they are no longer needed

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
    // Calculate sparsity as the percentage of *active* bits
    return 100.0 * (static_cast<double>(data.activePositions.size()) /
                              EncodingRanges::MAX_VECTOR_SIZE);
}

// --- Vocabulary Access Methods ---

const std::vector<std::string>& SparseDistributedRepresentation::getWordVocabulary() const {
    return wordEncoder_->getVocabulary();
}

// Removed getWordToIndexMap implementation

void SparseDistributedRepresentation::setWordVocabulary(const std::vector<std::string>& vocab) { // Updated signature
    wordEncoder_->setVocabulary(vocab); // Call updated WordEncoding method
}

// --- End Vocabulary Access Methods ---


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
