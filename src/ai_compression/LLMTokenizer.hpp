#pragma once

#include "SparseInferenceEngine.hpp"
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace CortexAICompression {

/**
 * Lightweight GGUF tokenizer runtime backed by archive segments.
 * This is a pragmatic tokenizer for local inference/debug loops.
 */
class LLMTokenizer {
public:
    LLMTokenizer() = default;

    void loadFromArchive(SDRModelLoader& loader);
    bool isLoaded() const { return loaded_; }
    size_t vocabSize() const { return vocab_.size(); }
    int eosId() const { return eos_id_; }
    int bosId() const { return bos_id_; }

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& token_ids) const;

    std::optional<int> tokenToId(const std::string& token) const;
    std::optional<std::string> idToToken(int token_id) const;

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    int unk_id_ = 0;
    int bos_id_ = -1;
    int eos_id_ = -1;
    int space_marker_id_ = -1;
    std::string word_boundary_marker_;
    bool loaded_ = false;
    bool byte_fallback_ = false;

    static std::string toPieceText(const std::string& text, std::string_view marker);
    static std::vector<std::string> splitLines(const std::string& text);
    bool matchSpecialToken(const std::string& text, size_t offset, int& token_id, size_t& token_len) const;
    int fallbackTokenForChar(char ch) const;
};

} // namespace CortexAICompression
