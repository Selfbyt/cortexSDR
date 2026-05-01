/**
 * @file cortex_sdk.cpp
 * @brief Implementation of the CortexSDR C++ SDK API
 * 
 * This file provides the C++ implementation of the CortexSDR software development kit,
 * offering high-level interfaces for neural network compression, decompression, and
 * inference operations with sparse distributed representations.
 * 
 * Key Features:
 * - Neural network model compression with various strategies
 * - On-demand layer loading for memory-efficient inference
 * - Multiple compression formats (SDR, RLE, Gzip, Quantization)
 * - C-compatible API for cross-language integration
 * - Comprehensive error handling and resource management
 */

#include "cortex_sdk.h"
#include "c_api.hpp"
#include "../SparseInferenceEngine.hpp"
#include "../LLMTokenizer.hpp"
#include "../core/AICompressor.hpp"
#include "../core/AIDecompressor.hpp"
#include <string>
#include <cstring>
#include "../parsers/ModelParserFactory.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h> // mkstemp, close, unlink
#include <fcntl.h>
#endif
 #include <sstream>
 #include <filesystem>

using namespace CortexAICompression;

// SDK version information
#define CORTEX_SDK_VERSION "1.0.0"

// Error code definitions for comprehensive error handling
#define CORTEX_SUCCESS 0                        ///< Operation completed successfully
#define CORTEX_ERROR_INVALID_ARGUMENT -1        ///< Invalid input parameter provided
#define CORTEX_ERROR_FILE_IO -2                 ///< File input/output operation failed
#define CORTEX_ERROR_MEMORY -3                  ///< Memory allocation or management error
#define CORTEX_ERROR_UNSUPPORTED_FORMAT -4      ///< Unsupported file or data format
#define CORTEX_ERROR_COMPRESSION -5             ///< Compression operation failed
#define CORTEX_ERROR_DECOMPRESSION -6           ///< Decompression operation failed
#define CORTEX_ERROR_INFERENCE -7               ///< Neural network inference failed
#define CORTEX_ERROR_UNKNOWN -99                ///< Unknown or unexpected error

/**
 * @brief Internal structure for inference engine handle management
 * 
 * Encapsulates the inference engine components with proper resource management
 * and supports both on-demand and legacy inference modes.
 */
struct CortexInferenceEngine {
    std::unique_ptr<SDRModelLoader> model_loader;
    std::unique_ptr<SDRInferenceEngine> inference_engine;
    std::unique_ptr<LLMTokenizer> tokenizer;
    std::string token_embedding_layer_name;
    std::vector<std::string> decoder_layer_order;
    std::unique_ptr<LayerInfo> cached_token_embedding;
    size_t native_hidden_dim = 0;
    bool native_decode_ready = false;
};

// Global storage for handles
static std::unordered_map<CortexInferenceEngineHandle, CortexInferenceEngine*> g_inferenceEngines;

// Helper functions
namespace {
    CortexError convert_exception(const std::exception& e) {
        char* msg_copy = new char[strlen(e.what()) + 1];
        strcpy(msg_copy, e.what());
        return {msg_copy, CORTEX_ERROR_UNKNOWN}; 
    }

    char* str_to_c(const std::string& str) {
        char* cstr = new char[str.length() + 1];
        strcpy(cstr, str.c_str());
        return cstr;
    }

    std::string to_lower_copy(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }

    bool is_token_embedding_name(const std::string& name_in) {
        const std::string n = to_lower_copy(name_in);
        return n.find("token_embd") != std::string::npos ||
               n.find("tok_embeddings") != std::string::npos ||
               n.find("embed_tokens") != std::string::npos;
    }

    bool is_decoder_excluded_name(const std::string& name_in) {
        const std::string n = to_lower_copy(name_in);
        if (n.find("gguf_") == 0 || n.find("tokenizer") != std::string::npos) {
            return true;
        }
        return is_token_embedding_name(n);
    }

    bool discover_native_llm_paths(CortexInferenceEngine& engine) {
        engine.token_embedding_layer_name.clear();
        engine.decoder_layer_order.clear();
        engine.native_hidden_dim = 0;

        const auto& segments = engine.model_loader->getSegmentIndex();
        for (const auto& seg : segments) {
            const std::string candidate = !seg.layer_name.empty() ? seg.layer_name : seg.name;
            if (engine.token_embedding_layer_name.empty() && is_token_embedding_name(candidate)) {
                engine.token_embedding_layer_name = candidate;
                if (seg.output_shape.size() >= 2) {
                    engine.native_hidden_dim = seg.output_shape.back();
                } else if (seg.input_shape.size() >= 2) {
                    engine.native_hidden_dim = seg.input_shape.back();
                }
            }
        }

        auto execution_order = engine.inference_engine->getExecutionOrder(segments);
        for (const auto& layer_name : execution_order) {
            if (is_decoder_excluded_name(layer_name)) {
                continue;
            }
            engine.decoder_layer_order.push_back(layer_name);
        }

        engine.native_decode_ready =
            engine.tokenizer && engine.tokenizer->isLoaded() &&
            !engine.token_embedding_layer_name.empty() &&
            !engine.decoder_layer_order.empty();
        return engine.native_decode_ready;
    }

    struct EmbeddingLayout {
        size_t vocab_size = 0;
        size_t embedding_dim = 0;
        bool valid() const { return vocab_size > 0 && embedding_dim > 0; }
    };

    EmbeddingLayout infer_embedding_layout(
        const CortexInferenceEngine& engine,
        const LayerInfo& token_embedding,
        int token_id_hint = -1) {
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
        if (engine.tokenizer && engine.tokenizer->isLoaded() && engine.native_hidden_dim > 0) {
            candidates.emplace_back(engine.tokenizer->vocabSize(), engine.native_hidden_dim);
        }
        if (engine.native_hidden_dim > 0) {
            candidates.emplace_back(total / engine.native_hidden_dim, engine.native_hidden_dim);
        }
        candidates.emplace_back(total / 3584, 3584);

        for (const auto& c : candidates) {
            const size_t vocab = c.first;
            const size_t dim = c.second;
            if (vocab == 0 || dim == 0) continue;
            if (vocab * dim > total) continue;
            if (token_id_hint >= 0 && static_cast<size_t>(token_id_hint) >= vocab) continue;
            if (dim < 64 || dim > 16384) continue;
            if (!best.valid() || vocab > best.vocab_size) {
                best = {vocab, dim};
            }
        }
        return best;
    }

    std::vector<float> embedding_lookup(
        const CortexInferenceEngine& engine,
        const LayerInfo& token_embedding,
        int token_id) {
        if (token_id < 0 || token_embedding.weights.empty()) {
            return {};
        }
        const EmbeddingLayout layout = infer_embedding_layout(engine, token_embedding, token_id);
        if (!layout.valid() || static_cast<size_t>(token_id) >= layout.vocab_size) {
            return {};
        }
        std::vector<float> out(layout.embedding_dim);
        const float* row = token_embedding.weights.data() + (static_cast<size_t>(token_id) * layout.embedding_dim);
        std::copy(row, row + layout.embedding_dim, out.begin());
        return out;
    }

    std::vector<float> build_prompt_context_hidden(
        const CortexInferenceEngine& engine,
        const LayerInfo& token_embedding,
        const std::vector<int>& prompt_tokens) {
        const EmbeddingLayout layout = infer_embedding_layout(engine, token_embedding);
        if (!layout.valid() || prompt_tokens.empty()) {
            return {};
        }

        std::vector<float> context(layout.embedding_dim, 0.0f);
        size_t used = 0;
        for (int token_id : prompt_tokens) {
            auto emb = embedding_lookup(engine, token_embedding, token_id);
            if (emb.empty()) {
                continue;
            }
            for (size_t i = 0; i < context.size(); ++i) {
                context[i] += emb[i];
            }
            ++used;
        }
        if (used == 0) {
            return {};
        }

        const float inv = 1.0f / static_cast<float>(used);
        for (float& value : context) {
            value *= inv;
        }
        return context;
    }

    bool token_is_repeated(const std::vector<int>& already_generated, int tok) {
        return std::find(already_generated.begin(), already_generated.end(), tok) != already_generated.end();
    }

    int select_next_token_from_embedding_transpose(
        const CortexInferenceEngine& engine,
        const std::vector<float>& hidden,
        const LayerInfo& token_embedding,
        const std::vector<int>& already_generated,
        float temperature = 0.8f) {
        const EmbeddingLayout layout = infer_embedding_layout(engine, token_embedding);
        if (!layout.valid() || hidden.size() != layout.embedding_dim) {
            return -1;
        }

        const float temp_scale = temperature > 0.0f ? (1.0f / temperature) : 1.0f;
        const float* emb = token_embedding.weights.data();
        struct BestCandidate {
            float score = -std::numeric_limits<float>::infinity();
            int token = -1;
        };

        const unsigned int hw_threads = (std::max)(1u, std::thread::hardware_concurrency());
        const size_t desired_workers = (std::max)(static_cast<size_t>(1), layout.vocab_size / 4096);
        const size_t hw_threads_size = static_cast<size_t>(hw_threads);
        const size_t worker_count = (std::min)(hw_threads_size, desired_workers);
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
            const size_t end = (std::min)(layout.vocab_size, begin + chunk);
            if (begin >= end) {
                best_by_worker[worker] = BestCandidate{};
                continue;
            }
            workers.emplace_back(score_range, worker, begin, end);
        }
        score_range(0, 0, (std::min)(layout.vocab_size, chunk));
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

    CortexError ensure_native_generation_ready(CortexInferenceEngine& engine) {
        if (engine.native_decode_ready) {
            return {nullptr, CORTEX_SUCCESS};
        }
        if (!engine.tokenizer) {
            engine.tokenizer = std::make_unique<LLMTokenizer>();
            try {
                engine.tokenizer->loadFromArchive(*engine.model_loader);
            } catch (const std::exception& e) {
                return {str_to_c(std::string("Tokenizer load failed: ") + e.what()), CORTEX_ERROR_INFERENCE};
            }
        }
        if (!discover_native_llm_paths(engine)) {
            return {"Native generation path is not available for this model", CORTEX_ERROR_UNSUPPORTED_FORMAT};
        }
        return {nullptr, CORTEX_SUCCESS};
    }
}

// Error handling and compression options initialization functions are defined in c_api.cpp
// We don't redefine them here to avoid duplicate symbol errors when building shared libraries

// Compressor functions - these are already implemented in c_api.cpp, so we'll just forward them
extern "C" {
    // These are defined in c_api.cpp
    extern CortexError cortex_compressor_create(const char* model_path, const char* format,
                                         const CortexCompressionOptions* options,
                                         CortexCompressorHandle* handle);
    
    extern CortexError cortex_compressor_compress(CortexCompressorHandle handle, const char* output_path);
    
    extern CortexError cortex_compressor_get_stats(CortexCompressorHandle handle,
                                            size_t* original_size,
                                            size_t* compressed_size,
                                            double* compression_ratio,
                                            double* compression_time_ms);
    
    extern CortexError cortex_compressor_free(CortexCompressorHandle handle);
    
    // Forward declaration - implementation is in c_api.cpp
CORTEXSDR_API CortexError cortex_decompressor_create(const char* compressed_path,
                                          CortexDecompressorHandle* handle,
                                          float sparsity);
    
    // Forward declaration - implementation is in c_api.cpp
CORTEXSDR_API CortexError cortex_decompressor_decompress(CortexDecompressorHandle handle,
                                               const char* compressed_path,
                                               const char* output_path);
    
    // Forward declaration - implementation is in c_api.cpp
CORTEXSDR_API CortexError cortex_decompressor_free(CortexDecompressorHandle handle);
}

// Inference Engine functions
CortexError cortex_inference_engine_create(
    const char* compressed_model_path,
    CortexInferenceEngineHandle* handle)
{
    try {
        if (!compressed_model_path || !handle) {
            return {"Invalid arguments (null pointers)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        // Create a new inference engine instance
        auto engine = new CortexInferenceEngine();
        
        // Create the model loader
        engine->model_loader = std::make_unique<SDRModelLoader>(compressed_model_path);
        
        // Create the inference engine
        engine->inference_engine = std::make_unique<SDRInferenceEngine>(*engine->model_loader);
        engine->inference_engine->enableAggressiveMemoryManagement(false);
        engine->inference_engine->enableLayerPrefetch(true);
        engine->inference_engine->setInferenceMode(false);
        engine->inference_engine->setBatchSize(1);

        try {
            engine->tokenizer = std::make_unique<LLMTokenizer>();
            engine->tokenizer->loadFromArchive(*engine->model_loader);
            discover_native_llm_paths(*engine);
        } catch (...) {
            engine->tokenizer.reset();
            engine->native_decode_ready = false;
        }
        
        // Store the handle
        *handle = static_cast<CortexInferenceEngineHandle>(engine);
        g_inferenceEngines[*handle] = engine;
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_set_batch_size(
    CortexInferenceEngineHandle handle,
    size_t batch_size)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->setBatchSize(batch_size);
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_enable_dropout(
    CortexInferenceEngineHandle handle,
    int enable)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->enableDropout(enable != 0);
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_set_mode(
    CortexInferenceEngineHandle handle,
    int training_mode)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->setInferenceMode(training_mode != 0);
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_run(
    CortexInferenceEngineHandle handle,
    const float* input_data,
    size_t input_size,
    float* output_data,
    size_t output_size,
    size_t* actual_output_size)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        if (!input_data || !output_data || !actual_output_size) {
            return {"Invalid arguments (null pointers)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        
        // Convert input data to vector
        std::vector<float> input(input_data, input_data + input_size);
        
        // Run inference
        std::vector<float> output = engine->inference_engine->run(input);
        
        // Check output buffer size
        if (output.size() > output_size) {
            *actual_output_size = output.size();
            return {"Output buffer too small", CORTEX_ERROR_MEMORY};
        }
        
        // Copy output data
        std::copy(output.begin(), output.end(), output_data);
        *actual_output_size = output.size();
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_run_layer(
    CortexInferenceEngineHandle handle,
    const char* layer_name,
    const float* input_data,
    size_t input_size,
    float* output_data,
    size_t output_size,
    size_t* actual_output_size)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        if (!layer_name || !input_data || !output_data || !actual_output_size) {
            return {"Invalid arguments (null pointers)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        
        // Convert input data to vector
        std::vector<float> input(input_data, input_data + input_size);
        
        // Load the layer
        LayerInfo layer = engine->model_loader->loadLayerByName(layer_name);
        
        // Run inference on the specific layer
        std::vector<float> output = engine->inference_engine->runLayer(layer, input);
        
        // Check output buffer size
        if (output.size() > output_size) {
            *actual_output_size = output.size();
            return {"Output buffer too small", CORTEX_ERROR_MEMORY};
        }
        
        // Copy output data
        std::copy(output.begin(), output.end(), output_data);
        *actual_output_size = output.size();
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_free(
    CortexInferenceEngineHandle handle)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        delete engine;
        g_inferenceEngines.erase(handle);
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_generate_text(
    CortexInferenceEngineHandle handle,
    const char* prompt,
    int max_new_tokens,
    char** out_text)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end() || !prompt || !out_text) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }

        auto engine = g_inferenceEngines[handle];
        CortexError ready = ensure_native_generation_ready(*engine);
        if (ready.code != CORTEX_SUCCESS) {
            return ready;
        }

        auto prompt_tokens = engine->tokenizer->encode(prompt);
        if (prompt_tokens.empty()) {
            return {"Prompt tokenization produced no tokens", CORTEX_ERROR_INFERENCE};
        }

        if (!engine->cached_token_embedding) {
            engine->cached_token_embedding = std::make_unique<LayerInfo>(
                engine->model_loader->loadLayerByName(engine->token_embedding_layer_name));
        }
        const LayerInfo& token_embedding = *engine->cached_token_embedding;
        if (token_embedding.weights.empty()) {
            return {"Token embedding layer has no weights", CORTEX_ERROR_INFERENCE};
        }

        const int step_limit = (std::max)(1, max_new_tokens);
        std::vector<int> generated;
        generated.reserve(static_cast<size_t>(step_limit));
        const std::vector<float> prompt_hidden =
            build_prompt_context_hidden(*engine, token_embedding, prompt_tokens);
        int cur_token = prompt_tokens.back();

        for (int step = 0; step < step_limit; ++step) {
            std::vector<float> decoder_out;
            if (step == 0) {
                if (prompt_hidden.empty()) {
                    return {"Prompt context embedding failed", CORTEX_ERROR_INFERENCE};
                }
                decoder_out = engine->inference_engine->runPrefill(prompt_hidden, engine->decoder_layer_order);
            } else {
                auto hidden = embedding_lookup(*engine, token_embedding, cur_token);
                if (hidden.empty()) {
                    return {"Token embedding lookup failed", CORTEX_ERROR_INFERENCE};
                }
                decoder_out = engine->inference_engine->runDecodeStep(hidden, engine->decoder_layer_order);
            }
            if (decoder_out.empty()) {
                return {"Decoder stack returned empty output", CORTEX_ERROR_INFERENCE};
            }

            cur_token = select_next_token_from_embedding_transpose(
                *engine, decoder_out, token_embedding, generated, 0.8f);
            if (cur_token < 0) {
                return {"Failed to sample next token", CORTEX_ERROR_INFERENCE};
            }
            generated.push_back(cur_token);
            if (engine->tokenizer->eosId() >= 0 && cur_token == engine->tokenizer->eosId()) {
                break;
            }
        }

        *out_text = str_to_c(engine->tokenizer->decode(generated));
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

const char* cortex_sdk_version() {
    return CORTEX_SDK_VERSION;
}
CortexError cortex_inference_engine_get_last_run_stats_json(
    CortexInferenceEngineHandle handle,
    char** out_json)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end() || !out_json) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        const auto& stats = engine->inference_engine->getLastRunStats();
        std::ostringstream os;
        os << "{\"total_ms\": " << stats.total_ms << ", \"layers\": [";
        for (size_t i = 0; i < stats.layers.size(); ++i) {
            const auto& l = stats.layers[i];
            os << "{\"name\":\"" << l.name << "\",\"load_ms\":" << l.load_ms
               << ",\"exec_ms\":" << l.exec_ms
               << ",\"output_size\":" << l.output_size
               << ",\"used_compressed\":" << (l.used_compressed ? "true" : "false")
               << "}";
            if (i + 1 < stats.layers.size()) os << ",";
        }
        os << "]}";
        std::string s = os.str();
        char* buf = (char*)malloc(s.size() + 1);
        if (!buf) return {"Allocation failure", CORTEX_ERROR_MEMORY};
        std::memcpy(buf, s.c_str(), s.size() + 1);
        *out_json = buf;
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

void cortex_free_string(char* s) {
    if (s) free(s);
}

// Inspect a compressed archive for tokenizer assets
CortexError cortex_archive_get_tokenizer_info(
    const char* archive_path,
    int* out_has_tokenizer,
    char** out_tokenizer_type)
{
    try {
        if (!archive_path || !out_has_tokenizer) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        *out_has_tokenizer = 0;
        if (out_tokenizer_type) *out_tokenizer_type = nullptr;

        // Create a temporary loader to read the archive index
        SDRModelLoader loader(archive_path);
        const auto& segments = loader.getSegmentIndex();
        bool has_vocab = false, has_merges = false, has_spm = false;
        for (const auto& seg : segments) {
            std::string n = seg.name;
            for (auto& ch : n) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
            if (n.find("tokenizer.model") != std::string::npos ||
                n.find("sentencepiece") != std::string::npos ||
                n.find(".spm") != std::string::npos ||
                n.find("gguf_tokenizer_model") != std::string::npos) {
                has_spm = true;
            }
            if (n.find("vocab.json") != std::string::npos ||
                n.find("vocab") != std::string::npos ||
                n.find("gguf_tokenizer_vocab") != std::string::npos) {
                has_vocab = true;
            }
            if (n.find("merges.txt") != std::string::npos || n.find("merges") != std::string::npos) {
                has_merges = true;
            }
        }
        if (has_spm || (has_vocab && has_merges)) {
            *out_has_tokenizer = 1;
            if (out_tokenizer_type) {
                std::string t = has_spm ? std::string("sentencepiece") : std::string("gpt2-bpe");
                *out_tokenizer_type = str_to_c(t);
            }
        }
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}
CortexError cortex_compress_from_url(
    const char* url_or_path,
    const char* format,
    const char* output_path,
    float sparsity)
{
    try {
        if (!url_or_path || !output_path) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }

        std::string src(url_or_path);
        std::string local_path = src;
        bool is_remote = (src.rfind("http://", 0) == 0) || (src.rfind("https://", 0) == 0);

        // Download to temp file if remote (or entire repo when applicable)
        std::string tmp_file;
        std::string tmp_dir; // used when downloading full Hugging Face repositories
        if (is_remote) {
            // Create a temporary file path
#ifdef _WIN32
            char tmpPath[MAX_PATH];
            DWORD len = GetTempPathA(MAX_PATH, tmpPath);
            if (len == 0 || len > MAX_PATH) {
                return {"Failed to get temp path", CORTEX_ERROR_FILE_IO};
            }
            char tmpFile[MAX_PATH];
            if (GetTempFileNameA(tmpPath, "cxsdr", 0, tmpFile) == 0) {
                return {"Failed to create temp file", CORTEX_ERROR_FILE_IO};
            }
            tmp_file = std::string(tmpFile);
#else
            char tmpname[] = "/tmp/cortexsdr_dl_XXXXXX";
            int fd = mkstemp(tmpname);
            if (fd == -1) {
                return {"Failed to create temp file", CORTEX_ERROR_FILE_IO};
            }
            close(fd);
            tmp_file = std::string(tmpname);
#endif

            // Build optional auth header for Hugging Face
            std::string authHeader;
            const char* env_hf1 = std::getenv("HUGGING_FACE_HUB_TOKEN");
            const char* env_hf2 = std::getenv("HUGGINGFACE_TOKEN");
            const char* env_hf3 = std::getenv("HF_TOKEN");
            const char* token = env_hf1 ? env_hf1 : (env_hf2 ? env_hf2 : env_hf3);
            bool is_hf = (src.find("huggingface.co/") != std::string::npos);
#ifndef _WIN32
            // Detect Hugging Face repository (folder) URL: no /resolve/ or /blob/
            auto strip_query = [](const std::string& u) {
                std::string p = u;
                auto qpos = p.find('?'); if (qpos != std::string::npos) p = p.substr(0, qpos);
                auto hpos = p.find('#'); if (hpos != std::string::npos) p = p.substr(0, hpos);
                return p;
            };
            std::string clean_src = strip_query(src);
            bool looks_like_repo = is_hf && (clean_src.find("/resolve/") == std::string::npos) && (clean_src.find("/blob/") == std::string::npos);
#else
            // On Windows, still attempt repo detection
            auto strip_query = [](const std::string& u) {
                std::string p = u;
                auto qpos = p.find('?'); if (qpos != std::string::npos) p = p.substr(0, qpos);
                auto hpos = p.find('#'); if (hpos != std::string::npos) p = p.substr(0, hpos);
                return p;
            };
            std::string clean_src = strip_query(src);
            bool looks_like_repo = is_hf && (clean_src.find("/resolve/") == std::string::npos) && (clean_src.find("/blob/") == std::string::npos);
#endif

            int rc = -1;
            bool handled_download = false;
            if (looks_like_repo) {
                // Create a temporary directory for the repository
#ifdef _WIN32
                char tmpPath[MAX_PATH];
                DWORD lenp = GetTempPathA(MAX_PATH, tmpPath);
                if (lenp == 0 || lenp > MAX_PATH) {
                    return {"Failed to get temp path", CORTEX_ERROR_FILE_IO};
                }
                char dirSeed[MAX_PATH];
                if (GetTempFileNameA(tmpPath, "cxsdr", 0, dirSeed) == 0) {
                    return {"Failed to create temp seed", CORTEX_ERROR_FILE_IO};
                }
                _unlink(dirSeed);
                std::filesystem::create_directory(dirSeed);
                tmp_dir = std::string(dirSeed);
#else
                char dirTemplate[] = "/tmp/cortexsdr_repo_XXXXXX";
                char* made = mkdtemp(dirTemplate);
                if (!made) {
                    return {"Failed to create temp directory", CORTEX_ERROR_FILE_IO};
                }
                tmp_dir = std::string(made);
#endif
                // Parse org/repo and optional revision from URL
                std::string after = clean_src;
                size_t hostPos = after.find("huggingface.co/");
                if (hostPos != std::string::npos) after = after.substr(hostPos + std::string("huggingface.co/").size());
                while (!after.empty() && after[0] == '/') after.erase(after.begin());
                std::string org, repo, revision;
                size_t s1 = after.find('/');
                if (s1 != std::string::npos) {
                    org = after.substr(0, s1);
                    std::string rest = after.substr(s1 + 1);
                    size_t s2 = rest.find('/');
                    if (s2 == std::string::npos) {
                        repo = rest;
                    } else {
                        repo = rest.substr(0, s2);
                        std::string t = rest.substr(s2 + 1);
                        if (t.rfind("tree/", 0) == 0) {
                            revision = t.substr(std::string("tree/").size());
                        }
                    }
                }
                if (!org.empty() && !repo.empty()) {
                    std::string fullrepo = org + "/" + repo;
#ifdef _WIN32
                    {
                        std::string cmd = std::string("huggingface-cli.exe download --repo-type model ") + fullrepo +
                            " --local-dir \"" + tmp_dir + "\"";
                        if (!revision.empty()) cmd += std::string(" --revision \"") + revision + "\"";
                        rc = system(cmd.c_str());
                    }
#else
                    {
                        std::string cmd = std::string("huggingface-cli download --repo-type model ") + fullrepo +
                            " --local-dir \"" + tmp_dir + "\"";
                        if (!revision.empty()) cmd += std::string(" --revision \"") + revision + "\"";
                        rc = system(cmd.c_str());
                    }
#endif
                    if (rc != 0) {
                        std::error_code ec; std::filesystem::remove_all(tmp_dir, ec);
                        return {"Failed to download Hugging Face repository (requires huggingface-cli)", CORTEX_ERROR_FILE_IO};
                    }
                    // Pick a primary model file by extension priority
                    std::vector<std::string> exts = {".gguf", ".onnx", ".pt", ".pth", ".pb"};
                    std::string chosen;
                    for (const auto& ext : exts) {
                        for (auto it = std::filesystem::recursive_directory_iterator(tmp_dir);
                             it != std::filesystem::recursive_directory_iterator(); ++it) {
                            if (!it->is_regular_file()) continue;
                            const auto& p = it->path();
                            std::string s = p.string();
                            if (s.size() >= ext.size() && s.rfind(ext) == s.size() - ext.size()) { chosen = s; break; }
                        }
                        if (!chosen.empty()) break;
                    }
                    if (chosen.empty()) {
                        std::error_code ec; std::filesystem::remove_all(tmp_dir, ec);
                        return {"Could not find a primary model file in repository", CORTEX_ERROR_FILE_IO};
                    }
                    local_path = chosen;
                    handled_download = true;
                } else {
                    // Fallback to single-file path if parsing failed
                    tmp_dir.clear();
                }
            }

            // Try curl first (curl.exe on Windows), then wget on Unix for single files
            if (!handled_download) {
#ifdef _WIN32
                {
                    std::string cmd = std::string("curl.exe -L -f -s --retry 3 --connect-timeout 10 ");
                    if (is_hf && token) {
                        cmd += std::string("-H \"Authorization: Bearer ") + token + "\" ";
                    }
                    cmd += std::string("-o \"") + tmp_file + "\" \"" + src + "\"";
                    rc = system(cmd.c_str());
                }
#else
                {
                    std::string cmd = std::string("curl -L -f -s --retry 3 --connect-timeout 10 ");
                    if (is_hf && token) {
                        cmd += std::string("-H \"Authorization: Bearer ") + token + "\" ";
                    }
                    cmd += std::string("-o ") + tmp_file + " " + src;
                    rc = system(cmd.c_str());
                    if (rc != 0) {
                        cmd = std::string("wget -q ");
                        if (is_hf && token) {
                            cmd += std::string("--header=\"Authorization: Bearer ") + token + "\" ";
                        }
                        cmd += std::string("-O ") + tmp_file + " " + src;
                        rc = system(cmd.c_str());
                    }
                }
#endif
                if (rc != 0) {
#ifdef _WIN32
                    _unlink(tmp_file.c_str());
#else
                    unlink(tmp_file.c_str());
#endif
                    return {"Failed to download remote model", CORTEX_ERROR_FILE_IO};
                }
                local_path = tmp_file;
            }
#ifdef _WIN32
            {
                std::string cmd = std::string("curl.exe -L -f -s --retry 3 --connect-timeout 10 ");
                if (is_hf && token) {
                    cmd += std::string("-H \"Authorization: Bearer ") + token + "\" ";
                }
                cmd += std::string("-o \"") + tmp_file + "\" \"" + src + "\"";
                rc = system(cmd.c_str());
            }
#else
            {
                std::string cmd = std::string("curl -L -f -s --retry 3 --connect-timeout 10 ");
                if (is_hf && token) {
                    cmd += std::string("-H \"Authorization: Bearer ") + token + "\" ";
                }
                cmd += std::string("-o ") + tmp_file + " " + src;
                rc = system(cmd.c_str());
                if (rc != 0) {
                    cmd = std::string("wget -q ");
                    if (is_hf && token) {
                        // wget supports --header to pass Authorization
                        cmd += std::string("--header=\"Authorization: Bearer ") + token + "\" ";
                    }
                    cmd += std::string("-O ") + tmp_file + " " + src;
                    rc = system(cmd.c_str());
                }
            }
#endif
            if (rc != 0) {
                #ifdef _WIN32
                _unlink(tmp_file.c_str());
                #else
                unlink(tmp_file.c_str());
                #endif
                return {"Failed to download remote model", CORTEX_ERROR_FILE_IO};
            }
            local_path = tmp_file;
        }

        // If local path is a directory (e.g., ~/.llama), try to detect a primary model file inside
        if (!is_remote) {
            std::error_code ec;
            if (std::filesystem::is_directory(local_path, ec)) {
                std::vector<std::string> exts = {".gguf", ".onnx", ".pt", ".pth", ".pb"};
                std::string chosen;
                for (const auto& ext : exts) {
                    for (auto it = std::filesystem::recursive_directory_iterator(local_path, ec);
                         it != std::filesystem::recursive_directory_iterator(); ++it) {
                        if (ec) break;
                        if (!it->is_regular_file()) continue;
                        const auto& p = it->path();
                        std::string s = p.string();
                        if (s.size() >= ext.size() && s.rfind(ext) == s.size() - ext.size()) { chosen = s; break; }
                    }
                    if (!chosen.empty()) break;
                }
                if (chosen.empty()) {
                    return {"Local directory provided but no primary model file (.gguf/.onnx/.pt/.pth/.pb) found", CORTEX_ERROR_FILE_IO};
                }
                local_path = chosen;
            }
        }

        // Prepare compression options
        CortexCompressionOptions options;
        CortexError err = cortex_compression_options_init(&options);
        if (err.code != CORTEX_SUCCESS) return err;
        options.sparsity = sparsity;
        options.verbose = 1;
        options.show_stats = 1;

        // Determine format: accept explicit, or auto-detect if empty/"auto"
        std::string format_str = (format ? std::string(format) : std::string());
        if (format_str.empty() || format_str == "auto") {
            try {
                format_str = CortexAICompression::ModelParserFactory::detectFormat(local_path);
            } catch (const std::exception& e) {
                // As a fallback, try extension-based guess or default to onnx
                std::cerr << "[SDK] Format auto-detection failed: " << e.what() << ". Assuming ONNX." << std::endl;
                format_str = "onnx";
            }
        }

        CortexCompressorHandle compressor;
        err = cortex_compressor_create(local_path.c_str(), format_str.c_str(), &options, &compressor);
        if (err.code != CORTEX_SUCCESS) {
            if (is_remote) {
#ifdef _WIN32
                if (tmp_dir.empty()) _unlink(local_path.c_str());
#else
                if (tmp_dir.empty()) unlink(local_path.c_str());
#endif
                if (!tmp_dir.empty()) { std::error_code ec; std::filesystem::remove_all(tmp_dir, ec); }
            }
            return err;
        }

        err = cortex_compressor_compress(compressor, output_path);
        cortex_compressor_free(compressor);

        if (is_remote) {
#ifdef _WIN32
            if (tmp_dir.empty()) _unlink(local_path.c_str());
#else
            if (tmp_dir.empty()) unlink(local_path.c_str());
#endif
            if (!tmp_dir.empty()) { std::error_code ec; std::filesystem::remove_all(tmp_dir, ec); }
        }
        return err;
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

// Memory management helpers exposed via SDK
CortexError cortex_inference_engine_init_memory_pool(
    CortexInferenceEngineHandle handle,
    size_t max_memory_mb)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->initializeMemoryPool(max_memory_mb);
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_enable_aggressive_memory(
    CortexInferenceEngineHandle handle,
    int enable)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->enableAggressiveMemoryManagement(enable != 0);
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_enable_layer_prefetch(
    CortexInferenceEngineHandle handle,
    int enable)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->enableLayerPrefetch(enable != 0);
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_get_memory_usage(
    CortexInferenceEngineHandle handle,
    size_t* bytes)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end() || !bytes) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        *bytes = engine->inference_engine->getCurrentMemoryUsage();
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}
