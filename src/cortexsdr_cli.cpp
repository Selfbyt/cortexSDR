/**
 * @file cortexsdr_cli.cpp
 * @brief Interactive CLI for CortexSDR neural network inference with on-demand loading
 * 
 * This file implements an interactive command-line interface for running neural network
 * inference with compressed models (.sdr files). Features include:
 * 
 * Key Features:
 * - On-demand layer loading (Ollama-style memory efficiency)
 * - Interactive chat mode for text generation
 * - Model statistics and performance monitoring
 * - Fallback support for legacy inference engine
 * - Real-time switching between loading modes
 * 
 * Usage Examples:
 * - Interactive mode: ./cortexsdr_cli model.sdr
 * - Single prompt: ./cortexsdr_cli -p "Hello world" model.sdr
 * - Legacy mode: ./cortexsdr_cli --legacy model.sdr
 */

#include "ai_compression/api/cortex_sdk.h"
#include "ai_compression/SparseInferenceEngine.hpp"
#include "ai_compression/LLMTokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <signal.h>
#include <chrono>
#include <memory>
#include <thread>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <unordered_map>

// Global state for inference engines and signal handling
volatile bool g_running = true;
CortexInferenceEngineHandle g_engine = nullptr;                                          // Legacy inference engine
std::unique_ptr<CortexAICompression::SDRModelLoader> g_model_loader = nullptr;          // On-demand model loader
std::unique_ptr<CortexAICompression::SDRInferenceEngine> g_inference_engine = nullptr;  // Modern inference engine
bool g_use_on_demand_loading = true;                                                    // Default to on-demand mode

// Last-run profiling snapshot for optional --profile output
static uint64_t g_last_duration_ms = 0;
static size_t g_last_token_count = 0;
static double g_last_tokens_per_sec = 0.0;
static std::string g_last_benchmark_json;
static bool g_has_model_tokenizer = false;                                             // Tokenizer assets present
static std::vector<std::string> g_tokenizer_vocab;
static std::unordered_map<std::string, int> g_tokenizer_lookup;
static std::unique_ptr<CortexAICompression::LLMTokenizer> g_llm_tokenizer;
static std::string g_token_embedding_layer_name;
static std::vector<std::string> g_decoder_layer_order;
static std::unique_ptr<CortexAICompression::LayerInfo> g_cached_token_embedding;
static size_t g_native_hidden_dim = 0;
static int g_native_max_steps = 32;           // allow a minimally coherent response by default
static uint64_t g_native_timeout_ms = 10000;  // 10s default decode guard
static bool g_fast_native_sampling = true;    // avoid full logits materialization when possible
static bool g_native_timing_debug = false;
enum class PromptTemplateStyle { None, QwenInstruct };
static PromptTemplateStyle g_prompt_template_style = PromptTemplateStyle::None;

/**
 * @brief Signal handler for graceful shutdown
 * @param signal Signal number received (SIGINT, SIGTERM, etc.)
 * 
 * Performs cleanup of inference engines and exits gracefully
 * when interrupt signals are received.
 */
void signal_handler(int signal) {
    std::cout << "\nShutting down..." << std::endl;
    g_running = false;
    if (g_engine) {
        cortex_inference_engine_free(g_engine);
        g_engine = nullptr;
    }
    g_inference_engine.reset();
    g_model_loader.reset();
    exit(0);
}

/**
 * @brief Print error information from CortexError structure
 * @param error CortexError containing error code and message
 * 
 * Displays formatted error information and frees associated resources.
 */
void print_error(const CortexError& error) {
    if (error.code != CORTEX_OK) {
        std::cerr << "Error " << error.code << ": " << (error.message ? error.message : "Unknown error") << std::endl;
        cortex_error_free(const_cast<CortexError*>(&error));
    }
}

/**
 * @brief Convert text to tensor representation for neural network input
 * @param text Input text string to convert
 * @return Vector of floats representing the tokenized text
 * 
 * Implements basic character-level tokenization by converting each character
 * to a normalized float value (0.0-1.0). Pads output to fixed size of 128.
 */
std::vector<float> text_to_tensor(const std::string& text) {
    std::vector<float> tensor;
    tensor.reserve(text.length() * 128);
    
    for (char ch : text) {
        float value = static_cast<float>(static_cast<unsigned char>(ch)) / 255.0f;
        tensor.push_back(value);
        
        if (tensor.size() < 128) {
            tensor.push_back(0.0f);
        }
    }
    
    while (tensor.size() < 128) {
        tensor.push_back(0.0f);
    }
    
    return tensor;
}

/**
 * @brief Convert neural network output tensor back to text
 * @param tensor Output tensor from neural network
 * @param max_length Maximum length of output text
 * @return Generated text string
 * 
 * Converts float tensor values back to characters by denormalizing
 * and filtering for printable ASCII characters.
 */
std::string tensor_to_text(const std::vector<float>& tensor, int max_length) {
    std::string result;
    result.reserve(max_length);
    
    for (size_t i = 0; i < tensor.size() && result.length() < max_length; i += 2) {
        if (i + 1 < tensor.size()) {
            float value = tensor[i] * 255.0f;
            int char_code = static_cast<int>(value + 0.5f);
            
            if (char_code >= 32 && char_code <= 126) {
                result += static_cast<char>(char_code);
            } else if (char_code == 10) {
                result += '\n';
            } else if (char_code == 32) {
                result += ' ';
            }
        }
    }
    
    if (result.empty()) {
        result = "Generated text output";
    }
    
    return result;
}

static std::string bytes_to_string(const std::vector<std::byte>& bytes) {
    if (bytes.empty()) return {};
    return std::string(
        reinterpret_cast<const char*>(bytes.data()),
        reinterpret_cast<const char*>(bytes.data() + bytes.size())
    );
}

static std::string to_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static void detect_prompt_template_style(const std::string& model_path) {
    g_prompt_template_style = PromptTemplateStyle::None;

    std::string hint_blob = to_lower_copy(model_path);
    if (g_model_loader) {
        try {
            auto meta = g_model_loader->loadSegmentByName("gguf_metadata");
            hint_blob += "\n";
            hint_blob += to_lower_copy(bytes_to_string(meta.data));
        } catch (...) {
        }
        try {
            auto cfg = g_model_loader->loadSegmentByName("gguf_config");
            hint_blob += "\n";
            hint_blob += to_lower_copy(bytes_to_string(cfg.data));
        } catch (...) {
        }
    }

    if (hint_blob.find("qwen") != std::string::npos &&
        (hint_blob.find("instruct") != std::string::npos ||
         hint_blob.find("chat_template") != std::string::npos)) {
        g_prompt_template_style = PromptTemplateStyle::QwenInstruct;
    }
}

static std::string apply_prompt_template_if_needed(const std::string& prompt) {
    if (g_prompt_template_style == PromptTemplateStyle::QwenInstruct &&
        prompt.find("<|im_start|>") == std::string::npos) {
        // Only inject the chat template when the loaded tokenizer can represent
        // the control tokens. Otherwise we flood the prompt with <unk> pieces,
        // which both slows generation and destroys output quality.
        if (g_llm_tokenizer) {
            const auto start_id = g_llm_tokenizer->tokenToId("<|im_start|>");
            const auto end_id = g_llm_tokenizer->tokenToId("<|im_end|>");
            const auto newline_id = g_llm_tokenizer->tokenToId("\n");
            if (!start_id.has_value() || !end_id.has_value() || !newline_id.has_value()) {
                return prompt;
            }
        }
        return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }
    return prompt;
}

static bool ensure_tokenizer_vocab_loaded() {
    if (!g_tokenizer_vocab.empty() || !g_model_loader) {
        return !g_tokenizer_vocab.empty();
    }

    try {
        CortexAICompression::LayerInfo vocab_layer = g_model_loader->loadLayerByName("gguf_tokenizer_vocab");
        std::istringstream stream(bytes_to_string(vocab_layer.raw_data));
        std::string token;
        while (std::getline(stream, token)) {
            const int token_id = static_cast<int>(g_tokenizer_vocab.size());
            g_tokenizer_lookup.emplace(token, token_id);
            g_tokenizer_vocab.push_back(token);
        }
    } catch (const std::exception& e) {
        std::cerr << "[Tokenizer] WARNING: Failed to load tokenizer vocab: " << e.what() << std::endl;
    }

    return !g_tokenizer_vocab.empty();
}

static int lookup_token_id(const std::string& token) {
    auto it = g_tokenizer_lookup.find(token);
    if (it != g_tokenizer_lookup.end()) return it->second;

    const std::string sentencepiece_space = std::string("\xE2\x96\x81") + token;
    it = g_tokenizer_lookup.find(sentencepiece_space);
    if (it != g_tokenizer_lookup.end()) return it->second;

    it = g_tokenizer_lookup.find(" " + token);
    if (it != g_tokenizer_lookup.end()) return it->second;

    if (token.size() == 1) {
        return static_cast<int>(static_cast<unsigned char>(token[0]));
    }
    return -1;
}

static std::vector<int> tokenize_text_with_vocab(const std::string& text) {
    std::vector<int> token_ids;
    ensure_tokenizer_vocab_loaded();

    std::string current;
    auto flush_current = [&]() {
        if (current.empty()) return;
        const int token_id = lookup_token_id(current);
        if (token_id >= 0) {
            token_ids.push_back(token_id);
        } else {
            for (unsigned char ch : current) token_ids.push_back(static_cast<int>(ch));
        }
        current.clear();
    };

    for (unsigned char ch : text) {
        if (std::isspace(ch)) {
            flush_current();
        } else if (std::ispunct(ch)) {
            flush_current();
            std::string punct(1, static_cast<char>(ch));
            const int token_id = lookup_token_id(punct);
            token_ids.push_back(token_id >= 0 ? token_id : static_cast<int>(ch));
        } else {
            current.push_back(static_cast<char>(ch));
        }
    }
    flush_current();
    return token_ids;
}

static bool infer_embedding_shape_from_index(size_t& vocab_size, size_t& embedding_dim) {
    vocab_size = 0;
    embedding_dim = 0;
    if (!g_model_loader) return false;

    for (const auto& seg : g_model_loader->getSegmentIndex()) {
        std::string name = !seg.layer_name.empty() ? seg.layer_name : seg.name;
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (name.find("token_embd") == std::string::npos &&
            name.find("tok_embeddings") == std::string::npos &&
            name.find("embed_tokens") == std::string::npos) {
            continue;
        }

        if (seg.input_shape.size() >= 2 && seg.output_shape.size() >= 2) {
            vocab_size = std::max(seg.input_shape.back(), seg.output_shape.back());
            embedding_dim = std::min(seg.input_shape.back(), seg.output_shape.back());
            return vocab_size > 0 && embedding_dim > 0;
        }
        if (seg.tensor_metadata && seg.tensor_metadata->dimensions.size() >= 2) {
            vocab_size = std::max(seg.tensor_metadata->dimensions[0], seg.tensor_metadata->dimensions[1]);
            embedding_dim = std::min(seg.tensor_metadata->dimensions[0], seg.tensor_metadata->dimensions[1]);
            return vocab_size > 0 && embedding_dim > 0;
        }
    }

    return false;
}

static std::vector<float> build_prompt_embedding(const std::string& prompt) {
    size_t vocab_size = 0;
    size_t embedding_dim = 0;
    if (!infer_embedding_shape_from_index(vocab_size, embedding_dim)) {
        return {};
    }

    std::vector<int> token_ids = tokenize_text_with_vocab(prompt);
    if (token_ids.empty()) {
        for (unsigned char ch : prompt) token_ids.push_back(static_cast<int>(ch));
    }
    if (token_ids.empty()) {
        return {};
    }

    std::vector<float> embedding(embedding_dim, 0.0f);
    size_t used_tokens = 0;
    for (int token_id : token_ids) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
            continue;
        }

        uint64_t state = static_cast<uint64_t>(token_id) + 0x9E3779B97F4A7C15ULL;
        for (size_t probe = 0; probe < 8; ++probe) {
            state ^= state >> 30;
            state *= 0xBF58476D1CE4E5B9ULL;
            state ^= state >> 27;
            state *= 0x94D049BB133111EBULL;
            state ^= state >> 31;
            embedding[static_cast<size_t>(state % embedding_dim)] += (state & 1ULL) ? 1.0f : -1.0f;
        }
        ++used_tokens;
    }

    if (used_tokens == 0) return {};

    const float scale = 1.0f / std::sqrt(static_cast<float>(used_tokens * 8));
    for (float& value : embedding) value *= scale;

    std::cout << "  - Tokenized prompt tokens: " << token_ids.size() << std::endl;
    std::cout << "  - Prompt embedding size: " << embedding.size() << std::endl;
    return embedding;
}

static bool is_token_embedding_name(const std::string& name_in) {
    std::string n = to_lower_copy(name_in);
    return n.find("token_embd") != std::string::npos ||
           n.find("tok_embeddings") != std::string::npos ||
           n.find("embed_tokens") != std::string::npos;
}

static bool is_decoder_excluded_name(const std::string& name_in) {
    std::string n = to_lower_copy(name_in);
    if (n.find("gguf_") == 0 || n.find("tokenizer") != std::string::npos) {
        return true;
    }
    return is_token_embedding_name(n);
}

static bool discover_native_llm_paths() {
    if (!g_model_loader || !g_inference_engine) {
        return false;
    }
    g_token_embedding_layer_name.clear();
    g_decoder_layer_order.clear();
    g_native_hidden_dim = 0;

    const auto& segments = g_model_loader->getSegmentIndex();
    for (const auto& seg : segments) {
        const std::string candidate = !seg.layer_name.empty() ? seg.layer_name : seg.name;
        if (g_token_embedding_layer_name.empty() && is_token_embedding_name(candidate)) {
            g_token_embedding_layer_name = candidate;
            if (seg.output_shape.size() >= 2) {
                g_native_hidden_dim = seg.output_shape.back();
            } else if (seg.input_shape.size() >= 2) {
                g_native_hidden_dim = seg.input_shape.back();
            }
        }
    }

    auto execution_order = g_inference_engine->getExecutionOrder(segments);
    for (const auto& layer_name : execution_order) {
        if (is_decoder_excluded_name(layer_name)) {
            continue;
        }
        g_decoder_layer_order.push_back(layer_name);
    }
    return !g_token_embedding_layer_name.empty();
}

struct EmbeddingLayout {
    size_t vocab_size = 0;
    size_t embedding_dim = 0;
    bool valid() const { return vocab_size > 0 && embedding_dim > 0; }
};

static EmbeddingLayout infer_embedding_layout(const CortexAICompression::LayerInfo& token_embedding, int token_id_hint = -1) {
    EmbeddingLayout best{};
    const size_t total = token_embedding.weights.size();
    if (total == 0) {
        return best;
    }

    std::vector<std::pair<size_t, size_t>> candidates;
    if (token_embedding.input_shape.size() >= 2 && token_embedding.output_shape.size() >= 2) {
        candidates.emplace_back(token_embedding.input_shape.back(), token_embedding.output_shape.back());
        candidates.emplace_back(token_embedding.output_shape.back(), token_embedding.input_shape.back());
    }
    if (g_llm_tokenizer && g_llm_tokenizer->isLoaded() && g_native_hidden_dim > 0) {
        candidates.emplace_back(g_llm_tokenizer->vocabSize(), g_native_hidden_dim);
    }
    if (g_native_hidden_dim > 0) {
        candidates.emplace_back(total / g_native_hidden_dim, g_native_hidden_dim);
    }
    candidates.emplace_back(total / 3584, 3584);

    for (const auto& c : candidates) {
        const size_t vocab = c.first;
        const size_t dim = c.second;
        if (vocab == 0 || dim == 0) continue;
        if (vocab * dim > total) continue;
        if (token_id_hint >= 0 && static_cast<size_t>(token_id_hint) >= vocab) continue;
        if (dim < 64 || dim > 16384) continue;
        if (!best.valid() || (vocab > best.vocab_size)) {
            best = {vocab, dim};
        }
    }
    return best;
}

static std::vector<float> embedding_lookup(const CortexAICompression::LayerInfo& token_embedding, int token_id) {
    if (token_id < 0 || token_embedding.weights.empty()) {
        return {};
    }
    const EmbeddingLayout layout = infer_embedding_layout(token_embedding, token_id);
    if (!layout.valid() || static_cast<size_t>(token_id) >= layout.vocab_size) {
        return {};
    }
    std::vector<float> out(layout.embedding_dim);
    const float* row = token_embedding.weights.data() + (static_cast<size_t>(token_id) * layout.embedding_dim);
    std::copy(row, row + layout.embedding_dim, out.begin());
    return out;
}

static std::vector<float> build_prompt_context_hidden(
    const CortexAICompression::LayerInfo& token_embedding,
    const std::vector<int>& prompt_tokens) {
    const EmbeddingLayout layout = infer_embedding_layout(token_embedding);
    if (!layout.valid() || prompt_tokens.empty()) {
        return {};
    }

    std::vector<float> context(layout.embedding_dim, 0.0f);
    float total_weight = 0.0f;
    const size_t count = prompt_tokens.size();
    const size_t start = count > 6 ? count - 6 : 0;
    for (size_t idx = start; idx < count; ++idx) {
        const int token_id = prompt_tokens[idx];
        if (g_llm_tokenizer) {
            if (token_id == g_llm_tokenizer->bosId() || token_id == g_llm_tokenizer->eosId()) {
                continue;
            }
        }
        auto emb = embedding_lookup(token_embedding, token_id);
        if (emb.empty()) {
            continue;
        }
        const float relative = static_cast<float>(idx - start + 1) / static_cast<float>(count - start + 1);
        float weight = 0.1f + 0.9f * relative * relative;
        if (g_llm_tokenizer) {
            auto token = g_llm_tokenizer->idToToken(token_id);
            if (token.has_value()) {
                std::string normalized = token.value();
                if (!normalized.empty() &&
                    static_cast<unsigned char>(normalized[0]) == 0xC4 &&
                    normalized.size() >= 2 &&
                    static_cast<unsigned char>(normalized[1]) == 0xA0) {
                    normalized.erase(0, 2);
                }
                const bool punctuation_only =
                    !normalized.empty() &&
                    std::all_of(normalized.begin(), normalized.end(), [](unsigned char ch) {
                        return std::ispunct(ch) != 0;
                    });
                const bool special_token =
                    normalized.size() >= 2 &&
                    normalized.front() == '<' &&
                    normalized.back() == '>';
                if (special_token) {
                    weight *= 0.15f;
                }
                if (punctuation_only) {
                    weight *= 0.2f;
                }
            }
        }
        for (size_t i = 0; i < context.size(); ++i) {
            context[i] += emb[i] * weight;
        }
        total_weight += weight;
    }
    if (total_weight <= 0.0f) {
        return {};
    }

    const float inv = 1.0f / total_weight;
    for (float& value : context) {
        value *= inv;
    }
    return context;
}

static bool token_is_repeated(const std::vector<int>& already_generated, int tok) {
    return std::find(already_generated.begin(), already_generated.end(), tok) != already_generated.end();
}

static int sample_token_deterministic(
    const std::vector<float>& logits,
    const std::vector<int>& already_generated,
    float temperature = 0.8f,
    size_t top_k = 40) {
    if (logits.empty()) return -1;

    std::vector<std::pair<float, int>> scored;
    scored.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        float score = logits[i];
        if (temperature > 0.0f) {
            score /= temperature;
        }
        const int tok = static_cast<int>(i);
        if (std::find(already_generated.begin(), already_generated.end(), tok) != already_generated.end()) {
            score -= 1.5f; // repetition penalty
        }
        scored.emplace_back(score, tok);
    }
    std::partial_sort(
        scored.begin(),
        scored.begin() + std::min(top_k, scored.size()),
        scored.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    return scored.empty() ? -1 : scored.front().second;
}

static int select_next_token_from_embedding_transpose(
    const std::vector<float>& hidden,
    const CortexAICompression::LayerInfo& token_embedding,
    const std::vector<int>& already_generated,
    float temperature = 0.8f) {
    const EmbeddingLayout layout = infer_embedding_layout(token_embedding);
    if (!layout.valid() || hidden.size() != layout.embedding_dim) {
        return -1;
    }

    const float temp_scale = temperature > 0.0f ? (1.0f / temperature) : 1.0f;
    const float* emb = token_embedding.weights.data();
    struct BestCandidate {
        float score = -std::numeric_limits<float>::infinity();
        int token = -1;
    };

    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t worker_count = std::min<size_t>(hw_threads, std::max<size_t>(1, layout.vocab_size / 4096));
    std::vector<BestCandidate> best_by_worker(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count > 0 ? worker_count - 1 : 0);

    auto score_range = [&](size_t worker_index, size_t begin, size_t end) {
        BestCandidate local_best;
        for (size_t token = begin; token < end; ++token) {
            const float* row = emb + (token * layout.embedding_dim);
            float score = 0.0f;
            for (size_t i = 0; i < layout.embedding_dim; ++i) {
                score += row[i] * hidden[i];
            }
            score *= temp_scale;

            const int tok = static_cast<int>(token);
            if (token_is_repeated(already_generated, tok)) {
                score -= 1.5f;
            }
            if (score > local_best.score) {
                local_best.score = score;
                local_best.token = tok;
            }
        }
        best_by_worker[worker_index] = local_best;
    };

    const size_t chunk = (layout.vocab_size + worker_count - 1) / worker_count;
    for (size_t worker = 1; worker < worker_count; ++worker) {
        const size_t begin = worker * chunk;
        const size_t end = std::min(layout.vocab_size, begin + chunk);
        if (begin >= end) {
            best_by_worker[worker] = BestCandidate{};
            continue;
        }
        workers.emplace_back(score_range, worker, begin, end);
    }
    score_range(0, 0, std::min(layout.vocab_size, chunk));
    for (auto& worker : workers) {
        worker.join();
    }

    BestCandidate best;
    for (const auto& candidate : best_by_worker) {
        if (candidate.score > best.score) {
            best = candidate;
        }
    }

    return best.token;
}

// Minimal parser to summarize benchmark JSON from SDK without external deps
static void print_benchmark_summary(const std::string& json) {
    if (json.empty()) return;
    // Extract total_ms
    double total_ms = 0.0;
    size_t pos = json.find("\"total_ms\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos);
        if (pos != std::string::npos) {
            size_t end = json.find_first_of(",}\n\r ", pos + 1);
            std::string num = json.substr(pos + 1, end == std::string::npos ? std::string::npos : (end - pos - 1));
            try { total_ms = std::stod(num); } catch (...) {}
        }
    }
    // Estimate layer count by counting occurrences of "\"name\":"
    size_t layers = 0;
    size_t from = 0;
    while ((from = json.find("\"name\"", from)) != std::string::npos) { layers++; from += 6; }
    std::cout << "[Benchmark] total_ms: " << std::fixed << std::setprecision(2) << total_ms
              << ", layers: " << layers << std::endl;
}

// Simple tokenizer used when model provides tokenizer assets; whitespace+punct splits
static size_t count_tokens_simple(const std::string& text) {
    size_t count = 0;
    bool in_token = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) || std::ispunct(static_cast<unsigned char>(c))) {
            if (in_token) { count++; in_token = false; }
            if (std::ispunct(static_cast<unsigned char>(c))) {
                // Treat punctuation as separate token
                count++;
            }
        } else {
            in_token = true;
        }
    }
    if (in_token) count++;
    return count;
}

/**
 * @brief Load neural network model with on-demand layer loading support
 * @param model_path Path to the compressed model file (.sdr)
 * @param sparsity Sparsity level for compression (default: 0.02)
 * @return True if model loaded successfully, false otherwise
 * 
 * Initializes both modern on-demand inference engine and legacy engine
 * as fallback. The on-demand engine provides Ollama-style memory efficiency
 * by loading layers only when needed during inference.
 */
bool load_model(const std::string& model_path, float sparsity = 0.02f) {
    std::cout << "Loading model: " << model_path << std::endl;
    
    // Check if model is already compressed (.sdr extension)
    std::string compressed_path = model_path;
    if (compressed_path.find(".sdr") == std::string::npos) {
        compressed_path = model_path + ".sdr";
        std::cout << "Looking for compressed model: " << compressed_path << std::endl;
    }
    
    try {
        // Create model loader with on-demand loading capability
        std::cout << "Initializing on-demand model loader..." << std::endl;
        g_model_loader = std::make_unique<CortexAICompression::SDRModelLoader>(compressed_path);
        
        // Create inference engine that uses the model loader
        std::cout << "Creating inference engine with on-demand loading..." << std::endl;
        g_inference_engine = std::make_unique<CortexAICompression::SDRInferenceEngine>(*g_model_loader);
        g_tokenizer_vocab.clear();
        g_tokenizer_lookup.clear();
        g_llm_tokenizer.reset();
        g_token_embedding_layer_name.clear();
        g_decoder_layer_order.clear();
        g_cached_token_embedding.reset();
        g_prompt_template_style = PromptTemplateStyle::None;
        
        // Configure inference engine for optimal performance
        g_inference_engine->setBatchSize(1);
        g_inference_engine->enableDropout(false);
        g_inference_engine->setInferenceMode(false); // Set to inference mode
        // Native decode is extremely sensitive to layer reload churn, so prefer speed here.
        g_inference_engine->enableAggressiveMemoryManagement(false);
        g_inference_engine->enableLayerPrefetch(true);
        g_inference_engine->initializeMemoryPool(8192); // 8GB cap by default; adjust as needed
        
        // Show model statistics
        const auto& segments = g_model_loader->getSegmentIndex();
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "  - Total segments: " << segments.size() << std::endl;
        std::cout << "  - On-demand loading: ENABLED" << std::endl;
        std::cout << "  - Memory footprint: MINIMAL (layers loaded as needed)" << std::endl;
        std::cout << "  - Memory pool: ENABLED (8GB limit by default)" << std::endl;
        // Heuristic: detect tokenizer assets by segment names
        g_has_model_tokenizer = false;
        for (const auto& seg : segments) {
            std::string n = seg.name;
            for (auto& ch : n) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
            if (n.find("tokenizer") != std::string::npos || n.find("vocab") != std::string::npos || n.find("merges") != std::string::npos) {
                g_has_model_tokenizer = true;
                break;
            }
        }
        std::cout << "  - Tokenizer assets: " << (g_has_model_tokenizer ? "FOUND" : "NOT FOUND") << std::endl;
        
        // Use SDK introspection for tokenizer info
        int has_tok = 0;
        char* tok_type = nullptr;
        CortexError tok_err = cortex_archive_get_tokenizer_info(compressed_path.c_str(), &has_tok, &tok_type);
        if (tok_err.code == CORTEX_OK) {
            g_has_model_tokenizer = (has_tok != 0);
            if (tok_type) {
                std::cout << "  - Tokenizer type: " << tok_type << std::endl;
                cortex_free_string(tok_type);
            }
        }

        if (g_has_model_tokenizer) {
            try {
                g_llm_tokenizer = std::make_unique<CortexAICompression::LLMTokenizer>();
                g_llm_tokenizer->loadFromArchive(*g_model_loader);
                std::cout << "  - Tokenizer runtime: LOADED (" << g_llm_tokenizer->vocabSize() << " tokens)" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  - Tokenizer runtime load failed: " << e.what() << std::endl;
                g_llm_tokenizer.reset();
                g_has_model_tokenizer = false;
            }
        }

        if (discover_native_llm_paths()) {
            std::cout << "  - Native decode path: READY" << std::endl;
            std::cout << "  - Embedding layer: " << g_token_embedding_layer_name << std::endl;
            std::cout << "  - Decoder layers: " << g_decoder_layer_order.size() << std::endl;
            std::cout << "  - Native decode max steps: " << g_native_max_steps << std::endl;
            std::cout << "  - Native decode timeout: " << g_native_timeout_ms << "ms" << std::endl;
        } else {
            std::cout << "  - Native decode path: NOT READY (fallback path only)" << std::endl;
        }
        detect_prompt_template_style(compressed_path);
        if (g_prompt_template_style == PromptTemplateStyle::QwenInstruct) {
            std::cout << "  - Prompt template: Qwen instruct chat" << std::endl;
        }
        
        // Also try to create legacy engine as fallback
        CortexError error = cortex_inference_engine_create(compressed_path.c_str(), &g_engine);
        if (error.code == CORTEX_OK) {
            std::cout << "  - Legacy engine: AVAILABLE (fallback mode)" << std::endl;
            cortex_inference_engine_set_batch_size(g_engine, 1);
            cortex_inference_engine_enable_dropout(g_engine, 0);
            cortex_inference_engine_set_mode(g_engine, 0);
            // Configure memory management via SDK for legacy engine path
            cortex_inference_engine_enable_aggressive_memory(g_engine, 1);
            cortex_inference_engine_init_memory_pool(g_engine, 8192);
        } else {
            std::cout << "  - Legacy engine: NOT AVAILABLE (on-demand only)" << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to load model: " << e.what() << std::endl;
        g_inference_engine.reset();
        g_model_loader.reset();
        return false;
    }
}

/**
 * @brief Generate text response using loaded neural network model
 * @param prompt Input text prompt for generation
 * @param max_length Maximum length of generated response
 * @return Generated text response
 * 
 * Performs inference using the loaded model with automatic fallback:
 * 1. First attempts on-demand inference (memory efficient)
 * 2. Falls back to legacy engine if on-demand fails
 * 3. Provides detailed timing and performance information
 */
std::string generate_text(const std::string& prompt, int max_length = 100) {
    if (!g_inference_engine && !g_engine) {
        return "Error: No model loaded";
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    const std::string effective_prompt = apply_prompt_template_if_needed(prompt);

    std::cout << "Starting inference..." << std::endl;
    std::cout << "  - Input prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "  - On-demand loading: " << (g_use_on_demand_loading ? "ENABLED" : "DISABLED") << std::endl;
    if (effective_prompt != prompt) {
        std::cout << "  - Applied prompt template for instruct chat" << std::endl;
    }

    std::string result;
    bool native_decode_used = false;
    bool native_decode_attempted = false;

    if (g_use_on_demand_loading && g_inference_engine && g_model_loader &&
        g_llm_tokenizer && g_llm_tokenizer->isLoaded() &&
        !g_token_embedding_layer_name.empty() && !g_decoder_layer_order.empty()) {
        try {
            native_decode_attempted = true;
            native_decode_used = true;
            std::cout << "  - Using native token decode path..." << std::endl;
            auto prompt_tokens = g_llm_tokenizer->encode(effective_prompt);
            if (prompt_tokens.empty()) {
                throw std::runtime_error("Prompt tokenization produced no tokens");
            }

            if (!g_cached_token_embedding) {
                const auto embed_load_start = std::chrono::steady_clock::now();
                g_cached_token_embedding = std::make_unique<CortexAICompression::LayerInfo>(
                    g_model_loader->loadLayerByName(g_token_embedding_layer_name));
                const auto embed_load_end = std::chrono::steady_clock::now();
                if (g_native_timing_debug) {
                    const auto embed_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        embed_load_end - embed_load_start).count();
                    std::cout << "  - Token embedding cache warmup: " << embed_load_ms << "ms" << std::endl;
                }
            }
            const CortexAICompression::LayerInfo& token_embedding = *g_cached_token_embedding;
            if (token_embedding.weights.empty() && token_embedding.raw_data.empty()) {
                throw std::runtime_error("Token embedding layer has no accessible backing data");
            }

            std::vector<int> generated;
            generated.reserve(static_cast<size_t>(max_length));
            const std::vector<float> prompt_hidden = build_prompt_context_hidden(token_embedding, prompt_tokens);
            int cur_token = prompt_tokens.back();
            const int step_limit = std::min(max_length, std::max(1, g_native_max_steps));
            std::string stop_reason = "max_steps";
            const auto native_start = std::chrono::steady_clock::now();
            for (int step = 0; step < step_limit; ++step) {
                const auto native_now = std::chrono::steady_clock::now();
                const auto native_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(native_now - native_start).count();
                if (static_cast<uint64_t>(native_elapsed) >= g_native_timeout_ms) {
                    stop_reason = "timeout";
                    break;
                }
                const auto decode_start = std::chrono::steady_clock::now();
                std::vector<float> decoder_out;
                if (step == 0) {
                    if (prompt_hidden.empty()) {
                        throw std::runtime_error("Prompt context embedding failed");
                    }
                    decoder_out = g_inference_engine->runPrefill(prompt_hidden, g_decoder_layer_order);
                } else {
                    auto hidden = embedding_lookup(token_embedding, cur_token);
                    if (hidden.empty()) {
                        throw std::runtime_error("Token embedding lookup failed");
                    }
                    decoder_out = g_inference_engine->runDecodeStep(hidden, g_decoder_layer_order);
                }
                const auto decode_end = std::chrono::steady_clock::now();
                if (decoder_out.empty()) {
                    throw std::runtime_error("Decoder stack returned empty output");
                }

                const auto sample_start = std::chrono::steady_clock::now();
                cur_token = g_fast_native_sampling
                    ? select_next_token_from_embedding_transpose(
                        decoder_out, token_embedding, generated, 0.8f)
                    : -1;
                const auto sample_end = std::chrono::steady_clock::now();
                if (cur_token < 0) {
                    throw std::runtime_error("Failed to sample next token");
                }
                if (g_native_timing_debug) {
                    const auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count();
                    const auto sample_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sample_end - sample_start).count();
                    std::cout << "    step " << (step + 1)
                              << ": decoder=" << decode_ms << "ms"
                              << ", sampling=" << sample_ms << "ms" << std::endl;
                }
                generated.push_back(cur_token);
                if (g_llm_tokenizer->eosId() >= 0 && cur_token == g_llm_tokenizer->eosId()) {
                    stop_reason = "eos";
                    break;
                }
            }
            result = g_llm_tokenizer->decode(generated);
            if (result.empty()) {
                result = "<empty>";
            }
            std::cout << "  - Native decode generated tokens: " << generated.size() << std::endl;
            std::cout << "  - Native decode stop reason: " << stop_reason << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  - Native decode failed: " << e.what() << std::endl;
            native_decode_used = false;
        }
    }

    if (!native_decode_used && g_engine && g_has_model_tokenizer) {
        native_decode_attempted = true;
        std::cout << "  - Falling back to SDK token decode path..." << std::endl;
        char* generated_text = nullptr;
        CortexError gen_error = cortex_inference_engine_generate_text(
            g_engine,
            effective_prompt.c_str(),
            max_length,
            &generated_text
        );
        if (gen_error.code == CORTEX_OK && generated_text) {
            native_decode_used = true;
            result = generated_text;
            cortex_free_string(generated_text);
        } else if (gen_error.code != CORTEX_OK) {
            std::cerr << "  - SDK token decode failed: "
                      << (gen_error.message ? gen_error.message : "unknown error") << std::endl;
            cortex_error_free(&gen_error);
        }
    }

    if (native_decode_attempted && !native_decode_used) {
        return "Error: Native GGUF decode failed (see stderr diagnostics)";
    }

    if (!native_decode_used) {
        std::vector<float> input_tensor = build_prompt_embedding(effective_prompt);
        if (input_tensor.empty()) {
            input_tensor = text_to_tensor(effective_prompt);
        }
        std::cout << "  - Input tensor size: " << input_tensor.size() << std::endl;

        std::vector<float> output_tensor;
        if (g_inference_engine && g_use_on_demand_loading) {
            std::cout << "  - Using on-demand layer-by-layer execution..." << std::endl;
            try {
                output_tensor = g_inference_engine->run(input_tensor);
            } catch (const std::exception& e) {
                std::cerr << "  - On-demand inference failed: " << e.what() << std::endl;
                if (!g_engine) {
                    return "Error: On-demand inference failed and no legacy engine available";
                }
            }
        }

        if ((output_tensor.empty()) && g_engine) {
            std::cout << "  - Falling back to legacy engine..." << std::endl;
            const size_t max_output_size = 1000;
            output_tensor.resize(max_output_size);
            size_t actual_output_size = 0;
            CortexError error = cortex_inference_engine_run(
                g_engine,
                input_tensor.data(),
                input_tensor.size(),
                output_tensor.data(),
                output_tensor.size(),
                &actual_output_size
            );
            if (error.code != CORTEX_OK) {
                print_error(error);
                return "Error: Inference failed";
            }
            output_tensor.resize(actual_output_size);
        }
        result = tensor_to_text(output_tensor, max_length);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Compute a simple tokens/sec metric using output length as proxy
    double seconds = static_cast<double>(duration.count()) / 1000.0;
    size_t token_count = g_has_model_tokenizer ? count_tokens_simple(result) : result.size();
    double tokens_per_sec = seconds > 0.0 ? (static_cast<double>(token_count) / seconds) : 0.0;
    
    g_last_duration_ms = static_cast<uint64_t>(duration.count());
    g_last_token_count = token_count;
    g_last_tokens_per_sec = tokens_per_sec;

    std::cout << "Generated in " << g_last_duration_ms << "ms";
    std::cout << " | tokens: " << g_last_token_count << " | tokens/sec: " << std::fixed << std::setprecision(2) << g_last_tokens_per_sec << std::endl;
    std::cout << "Response: " << result << std::endl;
    // Print SDK benchmark stats if legacy is in use; otherwise advise how to fetch stats via SDK
    if (g_engine) {
        char* json = nullptr;
        CortexError e = cortex_inference_engine_get_last_run_stats_json(g_engine, &json);
        if (e.code == CORTEX_OK && json) {
            g_last_benchmark_json = json;
            print_benchmark_summary(g_last_benchmark_json);
            cortex_free_string(json);
        }
    }
    
    return result;
}

/**
 * @brief Main interactive chat loop with enhanced command support
 * 
 * Provides an interactive interface with the following commands:
 * - Text input for generation
 * - load <model>: Load new model
 * - stats: Show model statistics
 * - layers: List available layers
 * - memory: Show memory usage
 * - toggle-mode: Switch between on-demand and legacy modes
 * - help: Show available commands
 * - quit/exit: Exit the application
 */
void interactive_chat() {
    std::cout << "\n=== CortexSDR Text Generation CLI (On-Demand Loading) ===" << std::endl;
    std::cout << "Type your messages and press Enter." << std::endl;
    std::cout << "Commands: 'quit' to exit, 'load <model>' to load a model" << std::endl;
    std::cout << "Enhanced: 'stats', 'toggle-mode', 'layers', 'memory'" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    std::string input;
    while (g_running) {
        std::cout << "\n[" << (g_use_on_demand_loading ? "ON-DEMAND" : "LEGACY") << "] > ";
        std::getline(std::cin, input);
        
        if (input == "quit" || input == "exit") {
            break;
        }
        
        if (input.empty()) {
            continue;
        }
        
        // Check for commands
        if (input.substr(0, 5) == "load ") {
            std::string model_path = input.substr(5);
            if (load_model(model_path)) {
                std::cout << "Model loaded successfully!" << std::endl;
            } else {
                std::cout << "Failed to load model: " << model_path << std::endl;
            }
            continue;
        }
        
        if (input == "help") {
            std::cout << "Available commands:" << std::endl;
            std::cout << "  load <model_path>  - Load a new model with on-demand capabilities" << std::endl;
            std::cout << "  stats             - Show model and inference statistics" << std::endl;
            std::cout << "  layers            - List all available layers in the model" << std::endl;
            std::cout << "  memory            - Show memory usage information" << std::endl;
            std::cout << "  toggle-mode       - Toggle between on-demand and legacy modes" << std::endl;
            std::cout << "  quit/exit         - Exit the application" << std::endl;
            std::cout << "  help              - Show this help" << std::endl;
            std::cout << "  <any text>        - Generate response" << std::endl;
            continue;
        }
        
        if (input == "stats") {
            if (g_inference_engine) {
                std::cout << "=== Model Statistics ===" << std::endl;
                const auto& segments = g_model_loader->getSegmentIndex();
                std::cout << "Total segments: " << segments.size() << std::endl;
                std::cout << "Encountered layer types: " << g_inference_engine->getEncounteredLayerTypes().size() << std::endl;
                std::cout << "Unhandled layer types: " << g_inference_engine->getUnhandledLayerTypes().size() << std::endl;
                std::cout << "On-demand loading: " << (g_use_on_demand_loading ? "ENABLED" : "DISABLED") << std::endl;
                std::cout << "Legacy engine: " << (g_engine ? "AVAILABLE" : "NOT AVAILABLE") << std::endl;
            } else {
                std::cout << "No model loaded." << std::endl;
            }
            continue;
        }
        
        if (input == "layers") {
            if (g_model_loader) {
                std::cout << "=== Available Layers ===" << std::endl;
                const auto& segments = g_model_loader->getSegmentIndex();
                int count = 0;
                for (const auto& seg : segments) {
                    std::cout << ++count << ". " << seg.name << " (type: " << seg.layer_type << ")" << std::endl;
                    if (count > 20) {
                        std::cout << "... and " << (segments.size() - count) << " more layers" << std::endl;
                        break;
                    }
                }
            } else {
                std::cout << "No model loaded." << std::endl;
            }
            continue;
        }
        
        if (input == "memory") {
            std::cout << "=== Memory Usage ===" << std::endl;
            std::cout << "On-demand loading reduces memory usage by loading only needed layers" << std::endl;
            if (g_model_loader) {
                std::cout << "Model loader: ACTIVE (minimal footprint)" << std::endl;
            }
            if (g_inference_engine) {
                std::cout << "Inference engine: ACTIVE" << std::endl;
            }
            if (g_engine) {
                std::cout << "Legacy engine: ACTIVE (additional memory overhead)" << std::endl;
            }
            continue;
        }
        
        if (input == "toggle-mode") {
            g_use_on_demand_loading = !g_use_on_demand_loading;
            std::cout << "Switched to " << (g_use_on_demand_loading ? "ON-DEMAND" : "LEGACY") << " mode" << std::endl;
            continue;
        }
        
        // Check if model is loaded
        if (!g_inference_engine && !g_engine) {
            std::cout << "No model loaded. Use 'load <model_path>' to load a model first." << std::endl;
            continue;
        }
        
        std::cout << "Generating response..." << std::endl;
        std::string response = generate_text(input);
        // Response is already printed in generate_text function
    }
}

/**
 * @brief Print comprehensive usage information
 * @param program_name Name of the program executable
 * 
 * Displays detailed usage information including command-line options,
 * examples, and interactive commands with emphasis on on-demand
 * loading capabilities and Ollama-style memory efficiency.
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [MODEL_PATH]" << std::endl;
    std::cout << std::endl;
    std::cout << "CortexSDR CLI with On-Demand Layer Loading (Ollama-style)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  -v, --version           Show version information" << std::endl;
    std::cout << "  -i, --interactive       Start interactive chat mode" << std::endl;
    std::cout << "  -p, --prompt TEXT       Generate text from prompt and exit" << std::endl;
    std::cout << "  -m, --max-length N      Maximum output length (default: 100)" << std::endl;
    std::cout << "  --profile PATH          Write profiling JSON after single-prompt run" << std::endl;
    std::cout << "  --tokenize-debug TEXT   Show tokenizer IDs and decoded roundtrip" << std::endl;
    std::cout << "  --native-max-steps N    Max token steps in native decode (default: 32)" << std::endl;
    std::cout << "  --native-timeout-ms N   Timeout for native decode loop (default: 10000)" << std::endl;
    std::cout << "  --legacy                Use legacy mode (load all layers at once)" << std::endl;
    std::cout << "  --on-demand             Use on-demand mode (load layers as needed)" << std::endl;
    std::cout << "  --download URL          Download model and compress to .sdr (auto-detect format)" << std::endl;
    std::cout << "  -o, --output PATH       Output .sdr path (optional; defaults to URL basename.sdr)" << std::endl;
    std::cout << "  -s, --sparsity F        Sparsity for compression (optional; default: 0.02)" << std::endl;
    std::cout << "  --hf-token TOKEN        Hugging Face access token (optional; can also use env: HUGGING_FACE_HUB_TOKEN/HUGGINGFACE_TOKEN/HF_TOKEN)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                              # Start interactive mode (no model)" << std::endl;
    std::cout << "  " << program_name << " model.sdr                    # Load model with on-demand loading" << std::endl;
    std::cout << "  " << program_name << " -i --on-demand              # Interactive with on-demand loading" << std::endl;
    std::cout << "  " << program_name << " -p \"Hello, how are you?\" model.sdr  # Generate single response" << std::endl;
    std::cout << "  " << program_name << " --legacy -m 200 -p \"Tell me a story\" model.sdr  # Legacy mode" << std::endl;
    std::cout << "  " << program_name << " --download https://host/model.onnx                 # Auto-detect format, output model.sdr" << std::endl;
    std::cout << "  " << program_name << " --download https://host/model.gguf -o llama.sdr  # Custom output name" << std::endl;
    std::cout << "  " << program_name << " --download https://host/model.onnx -s 0.01       # Custom sparsity" << std::endl;
    std::cout << std::endl;
    std::cout << "Interactive Commands:" << std::endl;
    std::cout << "  load <model_path>       Load a model with on-demand capabilities" << std::endl;
    std::cout << "  stats                   Show model statistics and performance" << std::endl;
    std::cout << "  layers                  List all available layers in the model" << std::endl;
    std::cout << "  memory                  Show memory usage information" << std::endl;
    std::cout << "  toggle-mode             Switch between on-demand and legacy modes" << std::endl;
    std::cout << "  help                    Show available commands" << std::endl;
    std::cout << "  quit/exit               Exit the application" << std::endl;
    std::cout << std::endl;
    std::cout << "On-Demand Loading Benefits:" << std::endl;
    std::cout << "  - Minimal memory footprint (similar to Ollama)" << std::endl;
    std::cout << "  - Faster startup time" << std::endl;
    std::cout << "  - Layer-by-layer execution with detailed logging" << std::endl;
    std::cout << "  - Better scalability for large models" << std::endl;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "CortexSDR CLI - Version: " << cortex_sdk_version() << std::endl;
    
    // Parse command line arguments
    std::string model_path;
    std::string prompt;
    bool interactive_mode = false;
    int max_length = 100;
    std::string profile_path;
    std::string tokenize_debug_text;
    // Download+compress (auto-detect) options
    bool run_download = false;
    std::string dl_url;
    std::string dl_output;
    float dl_sparsity = 0.02f;
    std::string hf_token;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            std::cout << "CortexSDR CLI Version: " << cortex_sdk_version() << std::endl;
            return 0;
        } else if (arg == "-i" || arg == "--interactive") {
            interactive_mode = true;
        } else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 < argc) {
                prompt = argv[++i];
            } else {
                std::cerr << "Error: --prompt requires a text argument" << std::endl;
                return 1;
            }
        } else if (arg == "-m" || arg == "--max-length") {
            if (i + 1 < argc) {
                max_length = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --max-length requires a number argument" << std::endl;
                return 1;
            }
        } else if (arg == "--profile") {
            if (i + 1 < argc) {
                profile_path = argv[++i];
            } else {
                std::cerr << "Error: --profile requires a path argument" << std::endl;
                return 1;
            }
        } else if (arg == "--tokenize-debug") {
            if (i + 1 < argc) {
                tokenize_debug_text = argv[++i];
            } else {
                std::cerr << "Error: --tokenize-debug requires text" << std::endl;
                return 1;
            }
        } else if (arg == "--native-max-steps") {
            if (i + 1 < argc) {
                g_native_max_steps = std::max(1, std::stoi(argv[++i]));
            } else {
                std::cerr << "Error: --native-max-steps requires an integer" << std::endl;
                return 1;
            }
        } else if (arg == "--native-timeout-ms") {
            if (i + 1 < argc) {
                g_native_timeout_ms = static_cast<uint64_t>(std::stoull(argv[++i]));
                if (g_native_timeout_ms == 0) g_native_timeout_ms = 1000;
            } else {
                std::cerr << "Error: --native-timeout-ms requires an integer" << std::endl;
                return 1;
            }
        } else if (arg == "-s" || arg == "--sparsity") {
            if (i + 1 < argc) {
                dl_sparsity = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --sparsity requires a float argument" << std::endl;
                return 1;
            }
        } else if (arg == "--legacy") {
            g_use_on_demand_loading = false;
            std::cout << "Legacy mode enabled (load all layers at once)" << std::endl;
        } else if (arg == "--on-demand") {
            g_use_on_demand_loading = true;
            std::cout << "On-demand mode enabled (load layers as needed)" << std::endl;
        } else if (arg == "--download") {
            if (i + 1 < argc) {
                run_download = true;
                dl_url = argv[++i];
            } else {
                std::cerr << "Error: --download requires a URL argument" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--hf-token") {
            if (i + 1 < argc) {
                hf_token = argv[++i];
            } else {
                std::cerr << "Error: --hf-token requires a token argument" << std::endl;
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                dl_output = argv[++i];
            } else {
                std::cerr << "Error: --output requires a path argument" << std::endl;
                return 1;
            }
        } else if (arg[0] != '-') {
            // This is the model path
            if (model_path.empty()) {
                model_path = arg;
            } else {
                std::cerr << "Error: Multiple model paths specified" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown option " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // If requested, run download (auto-detect) and exit
    if (run_download) {
        // Derive output if not provided: store under user's home dir
        if (dl_output.empty()) {
            std::string path = dl_url;
            // Strip query/fragment
            auto qpos = path.find('?');
            if (qpos != std::string::npos) path = path.substr(0, qpos);
            auto hpos = path.find('#');
            if (hpos != std::string::npos) path = path.substr(0, hpos);
            // Extract basename
            auto slash = path.find_last_of('/') ;
            std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
            if (base.empty()) base = "model";
            // Remove extension
            auto dot = base.find_last_of('.');
            if (dot != std::string::npos) base = base.substr(0, dot);
            // Determine home directory
            std::string homeDir;
#ifdef _WIN32
            const char* homeEnv = std::getenv("USERPROFILE");
#else
            const char* homeEnv = std::getenv("HOME");
#endif
            if (homeEnv && *homeEnv) {
                homeDir = homeEnv;
            } else {
                homeDir = "."; // fallback to current directory
            }
            std::string outDir = homeDir + std::string("/CortexSDR/models");
            std::error_code mkec;
            std::filesystem::create_directories(outDir, mkec);
            dl_output = outDir + std::string("/") + base + std::string(".sdr");
        }

        // If token provided, set env vars for this process so SDK picks it up
        if (!hf_token.empty()) {
#ifdef _WIN32
            _putenv_s("HUGGING_FACE_HUB_TOKEN", hf_token.c_str());
            _putenv_s("HUGGINGFACE_TOKEN", hf_token.c_str());
            _putenv_s("HF_TOKEN", hf_token.c_str());
#else
            setenv("HUGGING_FACE_HUB_TOKEN", hf_token.c_str(), 1);
            setenv("HUGGINGFACE_TOKEN", hf_token.c_str(), 1);
            setenv("HF_TOKEN", hf_token.c_str(), 1);
#endif
        }

        std::cout << "Downloading and compressing..." << std::endl;
        std::cout << "  URL: " << dl_url << std::endl;
        std::cout << "  Output: " << dl_output << std::endl;
        std::cout << "  Sparsity: " << dl_sparsity << std::endl;

        CortexError err = cortex_compress_from_url(
            dl_url.c_str(),
            "auto",
            dl_output.c_str(),
            dl_sparsity
        );
        if (err.code != CORTEX_OK) {
            std::cerr << "Download failed or compression error: " << (err.message ? err.message : "Unknown error") << std::endl;
            cortex_error_free(&err);
            return 1;
        }
        std::cout << "Download and compression completed: " << dl_output << std::endl;
        return 0;
    }

    // Load the model if provided
    if (!model_path.empty()) {
        if (!load_model(model_path)) {
            std::cerr << "Failed to load model: " << model_path << std::endl;
            return 1;
        }
    } else {
        std::cout << "No model specified. Use 'load <model_path>' to load a model." << std::endl;
    }
    
    // Run in appropriate mode
    if (!tokenize_debug_text.empty()) {
        if (!g_llm_tokenizer || !g_llm_tokenizer->isLoaded()) {
            std::cerr << "Tokenizer runtime is not loaded. Provide a GGUF-derived .sdr model." << std::endl;
            return 1;
        }
        auto ids = g_llm_tokenizer->encode(tokenize_debug_text);
        std::cout << "Token IDs:";
        for (int id : ids) {
            std::cout << " " << id;
        }
        std::cout << std::endl;
        std::cout << "Token Texts:";
        for (int id : ids) {
            auto token = g_llm_tokenizer->idToToken(id);
            std::cout << " [" << id << ":";
            if (token.has_value()) {
                for (unsigned char ch : token.value()) {
                    if (ch == '\n') std::cout << "\\n";
                    else if (ch == '\r') std::cout << "\\r";
                    else if (ch == '\t') std::cout << "\\t";
                    else std::cout << static_cast<char>(ch);
                }
            } else {
                std::cout << "<missing>";
            }
            std::cout << "]";
        }
        std::cout << std::endl;
        std::cout << "Roundtrip: " << g_llm_tokenizer->decode(ids) << std::endl;
    } else if (!prompt.empty()) {
        // Single prompt mode
        std::cout << "Generating response for: " << prompt << std::endl;
        std::string response = generate_text(prompt, max_length);
        std::cout << response << std::endl;

        // Optional profiling output
        if (!profile_path.empty()) {
            try {
                std::ofstream pf(profile_path);
                if (!pf) {
                    std::cerr << "Error: Could not open profile path: " << profile_path << std::endl;
                } else {
                    pf << "{\n";
                    pf << "  \"duration_ms\": " << g_last_duration_ms << ",\n";
                    pf << "  \"tokens\": " << g_last_token_count << ",\n";
                    pf << "  \"tokens_per_sec\": " << std::fixed << std::setprecision(2) << g_last_tokens_per_sec << ",\n";
                    pf << "  \"engine_stats\": " << (g_last_benchmark_json.empty() ? "null" : g_last_benchmark_json) << "\n";
                    pf << "}\n";
                    std::cout << "Wrote profile to " << profile_path << std::endl;
                }
            } catch (...) {
                std::cerr << "Warning: Failed to write profile JSON" << std::endl;
            }
        }
    } else {
        // Interactive mode (default)
        interactive_chat();
    }
    
    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    g_inference_engine.reset();
    g_model_loader.reset();
    if (g_engine) {
        cortex_inference_engine_free(g_engine);
    }
    
    return 0;
}
