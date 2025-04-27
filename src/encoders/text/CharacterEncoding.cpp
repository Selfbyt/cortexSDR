#include "CharacterEncoding.hpp"
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <bitset>
#include <string_view>

namespace {
    const std::string VALID_CHARS =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,!?-'\"();:";

    const std::string BASE64_CHARS =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    constexpr size_t CHUNK_SIZE = 8;
    constexpr float COMPRESSION_RATIO = 5.0f;
}

class CharacterEncoding::Impl {
public:
    std::unordered_map<char, unsigned char> base64Table;
    std::unordered_map<std::string_view, size_t> patternTable;
    std::vector<std::pair<std::string, float>> frequencyTable;

    Impl() {
        // Initialize base64 decoding table
        for (int i = 0; i < 64; ++i) {
            base64Table[BASE64_CHARS[i]] = i;
        }

        // Initialize pattern recognition
        initializePatterns();
    }

    void initializePatterns() {
        patternTable.reserve(1024);
        frequencyTable.reserve(1024);
    }

    void updatePatterns(const std::string& text) {
        for (size_t i = 0; i < text.length() - CHUNK_SIZE + 1; i++) {
            std::string_view pattern(text.data() + i, CHUNK_SIZE);
            patternTable[pattern]++;

            if (patternTable[pattern] > 1) {
                updateFrequencyTable(pattern);
            }
        }
    }

    void updateFrequencyTable(std::string_view pattern) {
        auto freq = patternTable[pattern];
        frequencyTable.emplace_back(std::string(pattern),
                                  static_cast<float>(freq) / pattern.length());

        std::sort(frequencyTable.begin(), frequencyTable.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });

        if (frequencyTable.size() > 1024) {
            frequencyTable.resize(1024);
        }
    }
};



CharacterEncoding::~CharacterEncoding() = default;

std::vector<size_t> CharacterEncoding::encodeText(const std::string& text) const {
    std::vector<size_t> encoded;
    encoded.reserve(text.length() / COMPRESSION_RATIO);

    // First pass: pattern recognition
    pimpl->updatePatterns(text);

    // Second pass: encoding with pattern optimization
    for (size_t i = 0; i < text.length(); i += CHARS_PER_POSITION) {
        size_t encodedValue = 0;
        size_t multiplier = 1;
        bool patternEncoded = false;

        // Try pattern matching first
        if (i + CHUNK_SIZE <= text.length()) {
            std::string_view chunk(text.data() + i, CHUNK_SIZE);
            auto it = pimpl->patternTable.find(chunk);
            if (it != pimpl->patternTable.end() && it->second > 1) {
                // TODO: Implement pattern encoding (encodePattern)
                // For now, skip pattern encoding and use fallback character encoding
                // encodedValue = encodePattern(std::string(chunk));
                // i += CHUNK_SIZE - 1;
                // patternEncoded = true;
            }
        }

        // Fall back to character encoding if no pattern match
        if (!patternEncoded) {
            for (size_t j = 0; j < CHARS_PER_POSITION && (i + j) < text.length(); ++j) {
                char c = text[i + j];
                size_t pos = getCharacterPosition(c);
                encodedValue += pos * multiplier;
                multiplier *= VALID_CHARS.length();
            }
        }

        encoded.push_back(encodedValue);
    }

    return encoded;
}

// TODO: Properly implement pattern index detection and decoding
static bool isPatternIndex(size_t /*index*/) {
    // Stub: No pattern indices encoded for now
    return false;
}

static std::string decodePattern(size_t /*index*/) {
    // Stub: No pattern decoding implemented
    return "";
}

std::string CharacterEncoding::decodeIndices(const std::vector<size_t>& indices) const {
    std::string decoded;
    const size_t base = VALID_CHARS.length();

    for (size_t index : indices) {
        if (isPatternIndex(index)) {
            decoded += decodePattern(index);
            continue;
        }

        size_t remaining = index;
        std::string chunk;

        for (size_t i = 0; i < CHARS_PER_POSITION; ++i) {
            if (remaining == 0 && i > 0) break;

            size_t charIndex = remaining % base;
            if (charIndex >= VALID_CHARS.length()) {
                throw std::invalid_argument("Invalid index value encountered during decoding");
            }

            chunk = VALID_CHARS[charIndex] + chunk;
            remaining /= base;
        }

        decoded += chunk;
    }

    return decoded;
}

std::string CharacterEncoding::encodeBase64(const std::string& input) const {
    std::string encoded;
    encoded.reserve(((input.length() + 2) / 3) * 4);

    std::bitset<24> buffer;
    int bits = 0;

    for (unsigned char c : input) {
        buffer <<= 8;
        buffer |= c;
        bits += 8;

        while (bits >= 6) {
            bits -= 6;
            size_t index = (buffer >> bits).to_ulong() & 0x3F;
            encoded += BASE64_CHARS[index];
        }
    }

    if (bits > 0) {
        buffer <<= (6 - bits);
        encoded += BASE64_CHARS[buffer.to_ulong() & 0x3F];
    }

    while (encoded.length() % 4) {
        encoded += '=';
    }

    return encoded;
}

std::string CharacterEncoding::decodeBase64(const std::string& encoded) const {
    std::string decoded;
    decoded.reserve(encoded.length() * 3 / 4);

    std::bitset<24> buffer;
    int bits = 0;

    for (char c : encoded) {
        if (c == '=') break;
        if (!isBase64Char(c)) {
            throw std::invalid_argument("Invalid base64 character encountered");
        }

        buffer <<= 6;
        buffer |= pimpl->base64Table[c];
        bits += 6;

        if (bits >= 8) {
            bits -= 8;
            decoded += static_cast<char>((buffer >> bits).to_ulong() & 0xFF);
        }
    }

    return decoded;
}

std::string CharacterEncoding::encodeURL(const std::string& input) const {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (char c : input) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
        } else if (c == ' ') {
            escaped << '+';
        } else {
            escaped << '%' << std::setw(2) << int(static_cast<unsigned char>(c));
        }
    }

    return escaped.str();
}

std::string CharacterEncoding::decodeURL(const std::string& encoded) const {
    std::string decoded;
    decoded.reserve(encoded.length());

    for (size_t i = 0; i < encoded.length(); ++i) {
        if (encoded[i] == '%' && i + 2 < encoded.length()) {
            decoded += hexToByte(encoded[i + 1], encoded[i + 2]);
            i += 2;
        }
        else if (encoded[i] == '+') {
            decoded += ' ';
        }
        else {
            decoded += encoded[i];
        }
    }

    return decoded;
}

size_t CharacterEncoding::getCharacterPosition(char c) const {
    size_t pos = VALID_CHARS.find(c);
    if (pos == std::string::npos) {
        throw std::invalid_argument(std::string("Invalid character encountered: ") + c);
    }
    return pos;
}

bool CharacterEncoding::isBase64Char(char c) {
    return (isalnum(c) || c == '+' || c == '/' || c == '=');
}

unsigned char CharacterEncoding::hexToByte(char first, char second) {
    unsigned char byte;
    std::istringstream iss(std::string{first, second});
    int value;
    iss >> std::hex >> value;
    byte = static_cast<unsigned char>(value);
    return byte;
}

bool CharacterEncoding::isHexChar(char c) {
    return (isdigit(c) || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f'));
}
