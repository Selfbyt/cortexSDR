#include "LLMTokenizer.hpp"

#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace CortexAICompression {

namespace {
constexpr std::string_view kSentencePieceMarker = "\xE2\x96\x81"; // U+2581
constexpr std::string_view kGpt2Marker = "\xC4\xA0";              // U+0120

std::unordered_map<uint32_t, unsigned char> buildGpt2ByteDecoder() {
    std::vector<int> bs;
    bs.reserve(256);
    for (int c = 33; c <= 126; ++c) bs.push_back(c);
    for (int c = 161; c <= 172; ++c) bs.push_back(c);
    for (int c = 174; c <= 255; ++c) bs.push_back(c);

    std::vector<int> cs = bs;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + extra);
            ++extra;
        }
    }

    std::unordered_map<uint32_t, unsigned char> decoder;
    decoder.reserve(bs.size());
    for (size_t i = 0; i < bs.size(); ++i) {
        decoder.emplace(static_cast<uint32_t>(cs[i]), static_cast<unsigned char>(bs[i]));
    }
    return decoder;
}

uint32_t decodeUtf8Codepoint(const std::string& text, size_t& index) {
    const unsigned char lead = static_cast<unsigned char>(text[index]);
    if (lead < 0x80) {
        ++index;
        return lead;
    }
    if ((lead >> 5) == 0x6 && index + 1 < text.size()) {
        const uint32_t value =
            ((lead & 0x1F) << 6) |
            (static_cast<unsigned char>(text[index + 1]) & 0x3F);
        index += 2;
        return value;
    }
    if ((lead >> 4) == 0xE && index + 2 < text.size()) {
        const uint32_t value =
            ((lead & 0x0F) << 12) |
            ((static_cast<unsigned char>(text[index + 1]) & 0x3F) << 6) |
            (static_cast<unsigned char>(text[index + 2]) & 0x3F);
        index += 3;
        return value;
    }
    if ((lead >> 3) == 0x1E && index + 3 < text.size()) {
        const uint32_t value =
            ((lead & 0x07) << 18) |
            ((static_cast<unsigned char>(text[index + 1]) & 0x3F) << 12) |
            ((static_cast<unsigned char>(text[index + 2]) & 0x3F) << 6) |
            (static_cast<unsigned char>(text[index + 3]) & 0x3F);
        index += 4;
        return value;
    }
    ++index;
    return lead;
}

void appendUtf8Codepoint(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string decodeGpt2VisibleBytes(const std::string& text) {
    static const std::unordered_map<uint32_t, unsigned char> kByteDecoder = buildGpt2ByteDecoder();
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const uint32_t cp = decodeUtf8Codepoint(text, i);
        auto it = kByteDecoder.find(cp);
        if (it != kByteDecoder.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            appendUtf8Codepoint(out, cp);
        }
    }
    return out;
}
}

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

std::string LLMTokenizer::toPieceText(const std::string& text, std::string_view marker) {
    std::string out;
    out.reserve(text.size() * 2);
    bool at_word_start = true;
    for (char ch : text) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isspace(uch)) {
            at_word_start = true;
            continue;
        }
        if (at_word_start && !marker.empty()) {
            out.append(marker.data(), marker.size());
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

std::optional<std::string> LLMTokenizer::idToToken(int token_id) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_.size()) {
        return std::nullopt;
    }
    return vocab_[static_cast<size_t>(token_id)];
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
    unk_id_ = 0;
    bos_id_ = -1;
    eos_id_ = -1;
    space_marker_id_ = -1;
    word_boundary_marker_.clear();
    loaded_ = false;
    byte_fallback_ = false;

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
        space_marker_id_ = static_cast<int>(' ');
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

    try {
        ModelSegment model_segment = loader.loadSegmentByName("gguf_tokenizer_model");
        const std::string model_text(
            reinterpret_cast<const char*>(model_segment.data.data()),
            model_segment.data.size());
        if (model_text.find("gpt2") != std::string::npos) {
            word_boundary_marker_ = std::string(kGpt2Marker);
        } else if (model_text.find("sentencepiece") != std::string::npos) {
            word_boundary_marker_ = std::string(kSentencePieceMarker);
        }
    } catch (const std::exception&) {
    }

    if (word_boundary_marker_.empty()) {
        for (const auto& token : vocab_) {
            if (token == kGpt2Marker ||
                (token.size() >= kGpt2Marker.size() &&
                 std::string_view(token.data(), kGpt2Marker.size()) == kGpt2Marker)) {
                word_boundary_marker_ = std::string(kGpt2Marker);
                break;
            }
            if (token == kSentencePieceMarker ||
                (token.size() >= kSentencePieceMarker.size() &&
                 std::string_view(token.data(), kSentencePieceMarker.size()) == kSentencePieceMarker)) {
                word_boundary_marker_ = std::string(kSentencePieceMarker);
                break;
            }
        }
    }

    if (!word_boundary_marker_.empty()) {
        auto marker_id = tokenToId(word_boundary_marker_);
        if (marker_id.has_value()) {
            space_marker_id_ = marker_id.value();
        }
    }

    loaded_ = true;
    byte_fallback_ = false;
}

bool LLMTokenizer::matchSpecialToken(
    const std::string& text,
    size_t offset,
    int& token_id,
    size_t& token_len) const {
    token_id = unk_id_;
    token_len = 0;
    if (offset >= text.size() || text[offset] != '<') {
        return false;
    }
    const size_t end = text.find('>', offset);
    if (end == std::string::npos) {
        return false;
    }
    const std::string candidate = text.substr(offset, end - offset + 1);
    auto id = tokenToId(candidate);
    if (!id.has_value()) {
        return false;
    }
    token_id = id.value();
    token_len = candidate.size();
    return true;
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

    std::vector<int> ids;
    ids.reserve(text.size() + 4);
    if (bos_id_ >= 0) {
        ids.push_back(bos_id_);
    }

    const bool use_boundary_markers = !word_boundary_marker_.empty();
    const size_t marker_len = word_boundary_marker_.size();
    size_t text_offset = 0;
    while (text_offset < text.size()) {
        int special_token = unk_id_;
        size_t special_len = 0;
        if (matchSpecialToken(text, text_offset, special_token, special_len)) {
            ids.push_back(special_token);
            text_offset += special_len;
            continue;
        }

        if (text[text_offset] == '\n' || text[text_offset] == '\r' ||
            text[text_offset] == '\t') {
            ids.push_back(fallbackTokenForChar(text[text_offset]));
            ++text_offset;
            continue;
        }

        size_t next_special = text.find('<', text_offset);
        size_t next_control = text.find_first_of("\n\r\t", text_offset);
        size_t chunk_end = std::min(
            next_special == std::string::npos ? text.size() : next_special,
            next_control == std::string::npos ? text.size() : next_control);
        const std::string chunk = text.substr(text_offset, chunk_end - text_offset);
        const std::string piece = use_boundary_markers ? toPieceText(chunk, word_boundary_marker_) : chunk;

        size_t i = 0;
        while (i < piece.size()) {
            size_t best_len = 0;
            int best_id = unk_id_;
            const size_t max_len = std::min<size_t>(piece.size() - i, 32);
            for (size_t len = max_len; len > 0; --len) {
                const std::string_view sv(piece.data() + i, len);
                auto it = token_to_id_.find(std::string(sv));
                if (it != token_to_id_.end()) {
                    best_len = len;
                    best_id = it->second;
                    break;
                }
            }

            if (best_len == 0 && use_boundary_markers &&
                marker_len > 0 &&
                i + marker_len <= piece.size() &&
                std::string_view(piece.data() + i, marker_len) == word_boundary_marker_) {
                const size_t remainder = piece.size() - (i + marker_len);
                const size_t raw_max_len = std::min<size_t>(remainder, 32);
                for (size_t len = raw_max_len; len > 0; --len) {
                    const std::string_view sv(piece.data() + i + marker_len, len);
                    auto it = token_to_id_.find(std::string(sv));
                    if (it != token_to_id_.end()) {
                        if (space_marker_id_ >= 0) {
                            ids.push_back(space_marker_id_);
                        }
                        best_len = len + marker_len;
                        best_id = it->second;
                        break;
                    }
                }
                if (best_len == 0 && space_marker_id_ >= 0) {
                    ids.push_back(space_marker_id_);
                    i += marker_len;
                    continue;
                }
            }

            if (best_len == 0) {
                best_len = 1;
                best_id = fallbackTokenForChar(piece[i]);
            }
            ids.push_back(best_id);
            i += best_len;
        }
        text_offset = chunk_end;
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

    if (word_boundary_marker_.empty()) {
        return out;
    }

    std::string text;
    text.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        if (i + word_boundary_marker_.size() <= out.size() &&
            std::string_view(out.data() + i, word_boundary_marker_.size()) == word_boundary_marker_) {
            if (!text.empty() && text.back() != ' ') {
                text.push_back(' ');
            }
            i += word_boundary_marker_.size() - 1;
            continue;
        }
        text.push_back(out[i]);
    }
    if (word_boundary_marker_ == kGpt2Marker) {
        return decodeGpt2VisibleBytes(text);
    }
    return text;
}

} // namespace CortexAICompression
