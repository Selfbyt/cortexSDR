#include "LLMTokenizer.hpp"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace CortexAICompression {

std::vector<std::string> LLMTokenizer::splitLines(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    return lines;
}

std::string LLMTokenizer::toPieceText(const std::string& text) {
    std::string out;
    out.reserve(text.size() * 2);
    bool at_word_start = true;
    for (char ch : text) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isspace(uch)) {
            at_word_start = true;
            continue;
        }
        if (at_word_start) {
            out += "\xE2\x96\x81"; // U+2581 '▁'
            at_word_start = false;
        }
        out.push_back(ch);
    }
    return out;
}

std::optional<int> LLMTokenizer::tokenToId(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it == token_to_id_.end()) {
        return std::nullopt;
    }
    return it->second;
}

int LLMTokenizer::fallbackTokenForChar(char ch) const {
    const std::string s(1, ch);
    auto id = tokenToId(s);
    if (id.has_value()) {
        return id.value();
    }
    return unk_id_;
}

void LLMTokenizer::loadFromArchive(SDRModelLoader& loader) {
    ModelSegment vocab_segment = loader.loadSegmentByName("gguf_tokenizer_vocab");
    if (vocab_segment.data.empty()) {
        throw std::runtime_error("Tokenizer vocab segment is empty");
    }

    const std::string vocab_text(
        reinterpret_cast<const char*>(vocab_segment.data.data()),
        vocab_segment.data.size());
    const auto lines = splitLines(vocab_text);
    if (lines.empty()) {
        throw std::runtime_error("Tokenizer vocab has no entries");
    }

    vocab_.clear();
    token_to_id_.clear();
    vocab_.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        vocab_.push_back(lines[i]);
        token_to_id_.emplace(lines[i], static_cast<int>(i));
    }

    const auto mostly_spaces = [&]() {
        if (vocab_.empty()) return true;
        size_t printable = 0;
        for (const auto& t : vocab_) {
            for (unsigned char ch : t) {
                if (!std::isspace(ch)) {
                    ++printable;
                }
                if (printable > 8) {
                    return false;
                }
            }
        }
        return true;
    };
    if (vocab_.size() <= 1 || mostly_spaces()) {
        // Fallback when compressed archive tokenizer payload is not recoverable.
        vocab_.clear();
        token_to_id_.clear();
        vocab_.reserve(256);
        for (int i = 0; i < 256; ++i) {
            std::string s(1, static_cast<char>(i));
            vocab_.push_back(s);
            token_to_id_[s] = i;
        }
        unk_id_ = static_cast<int>('?');
        bos_id_ = -1;
        eos_id_ = -1;
        loaded_ = true;
        byte_fallback_ = true;
        return;
    }

    auto set_if = [&](const char* token, int& out) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            out = it->second;
        }
    };
    set_if("<unk>", unk_id_);
    set_if("<s>", bos_id_);
    set_if("</s>", eos_id_);
    loaded_ = true;
    byte_fallback_ = false;
}

std::vector<int> LLMTokenizer::encode(const std::string& text) const {
    if (byte_fallback_) {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char ch : text) {
            ids.push_back(static_cast<int>(ch));
        }
        if (ids.empty()) {
            ids.push_back(static_cast<int>(' '));
        }
        return ids;
    }
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    const std::string piece = toPieceText(text);
    std::vector<int> ids;
    ids.reserve(piece.size() + 4);
    if (bos_id_ >= 0) {
        ids.push_back(bos_id_);
    }

    size_t i = 0;
    while (i < piece.size()) {
        size_t best_len = 0;
        int best_id = unk_id_;
        const size_t max_len = std::min<size_t>(piece.size() - i, 24);
        for (size_t len = max_len; len > 0; --len) {
            const std::string_view sv(piece.data() + i, len);
            auto it = token_to_id_.find(std::string(sv));
            if (it != token_to_id_.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_len == 0) {
            best_len = 1;
            best_id = fallbackTokenForChar(piece[i]);
        }
        ids.push_back(best_id);
        i += best_len;
    }
    return ids;
}

std::string LLMTokenizer::decode(const std::vector<int>& token_ids) const {
    if (byte_fallback_) {
        std::string text;
        text.reserve(token_ids.size());
        for (int id : token_ids) {
            if (id < 0) continue;
            unsigned char ch = static_cast<unsigned char>(id & 0xFF);
            if (std::isprint(ch) || ch == '\n' || ch == '\t' || ch == ' ') {
                text.push_back(static_cast<char>(ch));
            }
        }
        return text;
    }
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    std::string out;
    for (int id : token_ids) {
        if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) {
            continue;
        }
        if (id == bos_id_ || id == eos_id_) {
            continue;
        }
        out += vocab_[static_cast<size_t>(id)];
    }

    std::string text;
    text.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        if (i + 2 < out.size() &&
            static_cast<unsigned char>(out[i]) == 0xE2 &&
            static_cast<unsigned char>(out[i + 1]) == 0x96 &&
            static_cast<unsigned char>(out[i + 2]) == 0x81) {
            if (!text.empty() && text.back() != ' ') {
                text.push_back(' ');
            }
            i += 2;
            continue;
        }
        text.push_back(out[i]);
    }
    return text;
}

} // namespace CortexAICompression
