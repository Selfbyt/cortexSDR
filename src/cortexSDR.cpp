#include "cortexSDR.hpp"
#include "debug_utils.hpp"
#include <fstream>

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

        // Only encode as word if token is alphanumeric and not empty
        bool isWord = !token.empty() && std::all_of(token.begin(), token.end(), ::isalnum);
        if (isWord) {
            auto wordIndices = wordEncoder_->encodeWord(token);
            setIndices(wordIndices);
        }
    }

    // CRITICAL: For decompression to work correctly, we must preserve ALL word fingerprints exactly
    // We will only apply sparsity to non-word regions
    
    EncodedData raw = getEncodedData();
    size_t desiredActive = static_cast<size_t>(sparsity_ * EncodingRanges::MAX_VECTOR_SIZE);
    
    if (desiredActive > 0 && raw.activePositions.size() > desiredActive) {
        // Separate indices by region
        std::vector<size_t> wordRegion, otherRegions;
        
        for (size_t pos : raw.activePositions) {
            if (pos <= EncodingRanges::WORD_END) {
                wordRegion.push_back(pos);
            } else {
                otherRegions.push_back(pos);
            }
        }
        
        // NEVER drop word bits - they are essential for correct decompression
        size_t wordBitsToKeep = wordRegion.size();
        
        // Calculate how many non-word bits we can keep
        size_t remainingBudget = 0;
        if (desiredActive > wordBitsToKeep) {
            remainingBudget = desiredActive - wordBitsToKeep;
        } else {
            // If we can't even keep all word bits, we must keep them anyway
            // This means we'll exceed the desired sparsity, but it's necessary
            remainingBudget = 0;
        }
        
        // Keep as many non-word bits as the budget allows
        size_t otherBitsToKeep = std::min(otherRegions.size(), remainingBudget);
        
        // Randomly select which non-word bits to keep
        if (otherBitsToKeep < otherRegions.size()) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(otherRegions.begin(), otherRegions.end(), g);
            otherRegions.resize(otherBitsToKeep);
        }
        
        // Combine and sort the preserved indices
        raw.activePositions.clear();
        raw.activePositions.insert(raw.activePositions.end(), wordRegion.begin(), wordRegion.end());
        raw.activePositions.insert(raw.activePositions.end(), otherRegions.begin(), otherRegions.end());
        std::sort(raw.activePositions.begin(), raw.activePositions.end());
        
        std::cout << "[DEBUG] Sparsity applied: Kept ALL " << wordRegion.size() << " word bits and " 
                  << otherRegions.size() << " other bits (total: " << raw.activePositions.size() << ")." << std::endl;
    }
    currentEncoding_ = raw;
    return currentEncoding_;
}

SparseDistributedRepresentation::EncodedData SparseDistributedRepresentation::encodeNumber(double number) {
    resetEncodedVector();
    // Encode number using NumberEncoding
    std::vector<size_t> indices = numberEncoder_->encodeNumber(number);
    std::cout << "Encoding number: " << number << std::endl;
    setIndices(indices);
    return getEncodedData();
}

SparseDistributedRepresentation::EncodedData SparseDistributedRepresentation::encodeImage(const std::string& imagePath) {
    resetEncodedVector();
    
    // Read the image file directly
    std::ifstream file(imagePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open image file: " + imagePath);
    }
    
    // Read the file content into a buffer
    std::vector<unsigned char> imageData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    // Add a special marker to indicate this is image data
    std::vector<size_t> indices;
    indices.push_back(0xFFFFFFFF); // Special marker for image data
    
    // Store the image size
    indices.push_back(imageData.size());
    
    // Encode the image data using a simple quantization approach
    constexpr unsigned char quantLevels = 16;
    for (size_t i = 0; i < imageData.size(); ++i) {
        unsigned char quantized = static_cast<unsigned char>((imageData[i] * quantLevels) / 256);
        if (quantized == 0) continue; // sparsify: skip zeros
        // Map (position, quantized value) to a unique index
        size_t idx = i * quantLevels + quantized;
        indices.push_back(idx);
    }
    
    // Set the indices in the encoded vector
    setIndices(indices);
    return getEncodedData();
}

bool SparseDistributedRepresentation::isImageData() const {
    const auto& data = currentEncoding_;
    // Check if the encoded data has our special image marker
    return !data.activePositions.empty() && data.activePositions[0] == 0xFFFFFFFF;
}

std::string SparseDistributedRepresentation::decode() const {
    // Check if this is image data
    if (isImageData()) {
        return decodeImage();
    }
    
    // Separate the active positions into word, special character, and number indices
    SeparatedIndices separated = separateIndices(currentEncoding_);
    
    // Debug: Print the separated indices
    std::cout << "[DEBUG-SEPARATE] Total active positions: " << currentEncoding_.activePositions.size() << std::endl;
    std::cout << "[DEBUG-SEPARATE] First 20 active positions: ";
    for (size_t i = 0; i < std::min(currentEncoding_.activePositions.size(), size_t(20)); ++i) {
        std::cout << currentEncoding_.activePositions[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "[DEBUG-SEPARATE] Word indices: " << separated.wordIndices.size() 
              << ", Special char indices: " << separated.specialCharIndices.size() 
              << ", Number indices: " << separated.numberIndices.size() << std::endl;
    
    std::cout << "[DEBUG-SEPARATE] First 20 word indices: ";
    for (size_t i = 0; i < std::min(separated.wordIndices.size(), size_t(20)); ++i) {
        std::cout << separated.wordIndices[i] << " ";
    }
    std::cout << std::endl;

    // Decode the components
    std::vector<std::string> components;
    
    // Decode words (using WordEncoding)
    if (!separated.wordIndices.empty()) {
        std::string decodedWords = wordEncoder_->decodeIndices(separated.wordIndices);
        components.push_back(decodedWords);
    }
    
    // Join the components into a single string
    return joinComponents(components);
}

std::string SparseDistributedRepresentation::decodeImage() const {
    if (!isImageData()) {
        throw std::runtime_error("Cannot decode as image: data is not in image format");
    }
    
    const auto& data = currentEncoding_;
    
    // Skip the first index (our special marker)
    if (data.activePositions.size() < 2) {
        throw std::runtime_error("Invalid image data format");
    }
    
    // Second index is image size
    size_t imageSize = data.activePositions[1];
    std::vector<unsigned char> decoded(imageSize, 0);
    constexpr unsigned char quantLevels = 16;
    
    // Each index after the second encodes (position, quantized value)
    for (size_t k = 2; k < data.activePositions.size(); ++k) {
        size_t idx = data.activePositions[k];
        size_t pos = idx / quantLevels;
        unsigned char quantized = static_cast<unsigned char>(idx % quantLevels);
        // Dequantize
        unsigned char value = static_cast<unsigned char>((quantized * 256) / quantLevels);
        if (pos < decoded.size()) decoded[pos] = value;
    }
    
    // Convert the decoded binary data to a string
    return std::string(decoded.begin(), decoded.end());
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
    
    // Debug output for active positions
    std::cout << "[DEBUG-SEPARATE] Total active positions: " << data.activePositions.size() << std::endl;
    std::cout << "[DEBUG-SEPARATE] First 20 active positions: ";
    size_t count = 0;
    for (size_t pos : data.activePositions) {
        if (count++ < 20) {
            std::cout << pos << " ";
        } else {
            break;
        }
    }
    std::cout << std::endl;

    for (size_t pos : data.activePositions) {
        if (pos <= EncodingRanges::WORD_END) {
            result.wordIndices.push_back(pos);
        } else if (pos <= EncodingRanges::SPECIAL_CHAR_END) {
            result.specialCharIndices.push_back(pos);
        } else if (pos <= EncodingRanges::NUMBER_END) {
            result.numberIndices.push_back(pos);
        }
    }
    
    // Debug output for separated indices
    std::cout << "[DEBUG-SEPARATE] Word indices: " << result.wordIndices.size() 
              << ", Special char indices: " << result.specialCharIndices.size()
              << ", Number indices: " << result.numberIndices.size() << std::endl;
              
    if (!result.wordIndices.empty()) {
        std::cout << "[DEBUG-SEPARATE] First 20 word indices: ";
        count = 0;
        for (size_t pos : result.wordIndices) {
            if (count++ < 20) {
                std::cout << pos << " ";
            } else {
                break;
            }
        }
        std::cout << std::endl;
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
