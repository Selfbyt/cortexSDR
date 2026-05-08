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
#include "../kernels/attention_kernels.hpp"
#include "../kernels/flash_attention.hpp"
#include "../kernels/blas_kernels.hpp"
#include "../core/AICompressor.hpp"
#include "../core/AIDecompressor.hpp"
#include <string>
#include <cstring>
#include "../parsers/ModelParserFactory.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <map>
#include <regex>
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
    struct TransformerBlock {
        std::string attn_norm;
        std::string attn_q;
        std::string attn_k;
        std::string attn_v;
        std::string attn_output;
        std::string ffn_norm;
        std::string ffn_gate;
        std::string ffn_up;
        std::string ffn_down;

        bool complete() const {
            return !attn_norm.empty() &&
                   !attn_q.empty() &&
                   !attn_k.empty() &&
                   !attn_v.empty() &&
                   !attn_output.empty() &&
                   !ffn_norm.empty() &&
                   !ffn_gate.empty() &&
                   !ffn_up.empty() &&
                   !ffn_down.empty();
        }
    };

    std::unique_ptr<SDRModelLoader> model_loader;
    std::unique_ptr<SDRInferenceEngine> inference_engine;
    std::unique_ptr<LLMTokenizer> tokenizer;
    std::string token_embedding_layer_name;
    std::vector<std::string> decoder_layer_order;
    std::unique_ptr<LayerInfo> cached_token_embedding;
    size_t native_hidden_dim = 0;
    bool native_decode_ready = false;
    std::vector<TransformerBlock> transformer_blocks;
    std::string output_norm_layer_name;
    std::string output_projection_layer_name;
    std::unordered_map<std::string, std::unique_ptr<LayerInfo>> layer_cache;
    bool prefer_fast_generation = true;
    size_t fast_projection_row_limit = 256;
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

    float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;

        uint32_t f;
        if (exponent == 0) {
            if (mantissa == 0) {
                f = sign << 31;
            } else {
                exponent = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3FF;
                f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1F) {
            f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        } else {
            f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
        return *reinterpret_cast<float*>(&f);
    }

    float bf16_to_fp32(uint16_t h) {
        const uint32_t bits = static_cast<uint32_t>(h) << 16;
        return *reinterpret_cast<const float*>(&bits);
    }

    bool is_gguf_quantized_format(const std::string& format_lower) {
        return format_lower.rfind("q", 0) == 0 ||
               format_lower.rfind("iq", 0) == 0 ||
               format_lower.rfind("tq", 0) == 0;
    }

    size_t gguf_block_bytes_for_format(const std::string& format_lower) {
        if (format_lower == "q4_0") return 18;
        if (format_lower == "q4_1") return 20;
        if (format_lower == "q5_0") return 22;
        if (format_lower == "q5_1") return 24;
        if (format_lower == "q8_0") return 34;
        if (format_lower == "q8_1") return 40;
        if (format_lower == "q2_k") return 84;
        if (format_lower == "q3_k") return 110;
        if (format_lower == "q4_k") return 144;
        if (format_lower == "q5_k") return 176;
        if (format_lower == "q6_k") return 210;
        if (format_lower == "q8_k") return 292;
        if (format_lower == "iq2_xxs") return 66;
        if (format_lower == "iq2_xs") return 74;
        if (format_lower == "iq3_xxs") return 98;
        if (format_lower == "iq1_s") return 50;
        if (format_lower == "iq4_nl") return 18;
        if (format_lower == "iq3_s") return 110;
        if (format_lower == "iq2_s") return 82;
        if (format_lower == "iq4_xs") return 136;
        if (format_lower == "iq1_m") return 56;
        if (format_lower == "q4_0_4_4") return 72;
        if (format_lower == "q4_0_4_8") return 144;
        if (format_lower == "q4_0_8_8") return 144;
        if (format_lower == "tq1_0") return 54;
        if (format_lower == "tq2_0") return 66;
        return 0;
    }

    size_t gguf_block_elements_for_format(const std::string& format_lower) {
        if (format_lower == "q4_0" || format_lower == "q4_1" ||
            format_lower == "q5_0" || format_lower == "q5_1" ||
            format_lower == "q8_0" || format_lower == "q8_1" ||
            format_lower == "iq4_nl") {
            return 32;
        }
        if (format_lower == "q4_0_4_4") {
            return 128;
        }
        if (format_lower == "q2_k" || format_lower == "q3_k" || format_lower == "q4_k" || format_lower == "q5_k" ||
            format_lower == "q6_k" || format_lower == "q8_k" ||
            format_lower == "iq2_xxs" || format_lower == "iq2_xs" || format_lower == "iq3_xxs" ||
            format_lower == "iq1_s" || format_lower == "iq3_s" || format_lower == "iq2_s" ||
            format_lower == "iq4_xs" || format_lower == "iq1_m" ||
            format_lower == "q4_0_4_8" || format_lower == "q4_0_8_8" ||
            format_lower == "tq1_0" || format_lower == "tq2_0") {
            return 256;
        }
        return 0;
    }

    size_t gguf_quant_bits_for_format(const std::string& format_lower) {
        if (format_lower == "q2_k" || format_lower == "iq2_xxs" || format_lower == "iq2_xs" ||
            format_lower == "iq2_s" || format_lower == "tq2_0") {
            return 2;
        }
        if (format_lower == "q3_k" || format_lower == "iq3_xxs" || format_lower == "iq3_s") {
            return 3;
        }
        if (format_lower == "q4_k" || format_lower == "q4_0_4_4" || format_lower == "q4_0_4_8" ||
            format_lower == "q4_0_8_8" || format_lower == "iq4_nl" || format_lower == "iq4_xs") {
            return 4;
        }
        if (format_lower == "q5_k") {
            return 5;
        }
        if (format_lower == "iq1_s" || format_lower == "iq1_m" || format_lower == "tq1_0") {
            return 1;
        }
        return 0;
    }

    uint32_t read_packed_bits_lsb(const uint8_t* bytes, size_t bit_offset, size_t bit_count) {
        uint32_t value = 0;
        for (size_t bit = 0; bit < bit_count; ++bit) {
            const size_t absolute_bit = bit_offset + bit;
            const size_t byte_index = absolute_bit / 8;
            const size_t bit_index = absolute_bit % 8;
            const uint8_t bit_value = static_cast<uint8_t>((bytes[byte_index] >> bit_index) & 0x01U);
            value |= static_cast<uint32_t>(bit_value) << bit;
        }
        return value;
    }

    void gguf_get_scale_min_q4_k(const uint8_t* scales, size_t j, uint8_t& scale, uint8_t& min) {
        if (j < 4) {
            scale = static_cast<uint8_t>(scales[j] & 0x3FU);
            min = static_cast<uint8_t>(scales[j + 4] & 0x3FU);
        } else {
            scale = static_cast<uint8_t>((scales[j + 4] & 0x0FU) | ((scales[j - 4] >> 6) << 4));
            min = static_cast<uint8_t>((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }
    }

    bool decode_supported_gguf_quantized(
        const std::byte* source_data,
        size_t source_size,
        const std::string& format_lower,
        size_t num_elements,
        std::vector<float>& output) {
        if (num_elements == 0 || source_data == nullptr || source_size == 0) {
            output.clear();
            return true;
        }

        const size_t block_bytes = gguf_block_bytes_for_format(format_lower);
        const size_t block_elems = gguf_block_elements_for_format(format_lower);
        if (block_bytes == 0 || block_elems == 0 || source_size < block_bytes) {
            return false;
        }

        const size_t available_blocks = source_size / block_bytes;
        const size_t available_elements = available_blocks * block_elems;
        if (available_elements == 0) {
            output.clear();
            return true;
        }

        const size_t decode_count = (std::min)(num_elements, available_elements);
        output.assign(decode_count, 0.0f);

        size_t out_index = 0;
        for (size_t block_index = 0; block_index < available_blocks && out_index < decode_count; ++block_index) {
            const uint8_t* block = reinterpret_cast<const uint8_t*>(source_data) + (block_index * block_bytes);

            if (format_lower == "q8_0") {
                const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block));
                const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
                for (size_t i = 0; i < 32 && out_index < decode_count; ++i) {
                    output[out_index++] = static_cast<float>(qs[i]) * d;
                }
                continue;
            }

            if (format_lower == "q8_1") {
                float d = 0.0f;
                std::memcpy(&d, block, sizeof(float));
                const int8_t* qs = reinterpret_cast<const int8_t*>(block + 8);
                for (size_t i = 0; i < 32 && out_index < decode_count; ++i) {
                    output[out_index++] = static_cast<float>(qs[i]) * d;
                }
                continue;
            }

            if (format_lower == "q6_k") {
                const uint8_t* ql = block;
                const uint8_t* qh = block + 128;
                const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
                const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block + 208));
                for (size_t half = 0; half < 2 && out_index < decode_count; ++half) {
                    const uint8_t* ql_half = ql + half * 64;
                    const uint8_t* qh_half = qh + half * 32;
                    const int8_t* sc_half = scales + half * 8;
                    for (size_t l = 0; l < 32 && out_index < decode_count; ++l) {
                        const int q0 = static_cast<int>((ql_half[l] & 0x0F) |
                                                        (((qh_half[l] >> 0) & 0x03) << 4)) - 32;
                        output[out_index++] = d * static_cast<float>(sc_half[0] * q0);
                    }
                    for (size_t l = 0; l < 32 && out_index < decode_count; ++l) {
                        const int q1 = static_cast<int>((ql_half[l + 32] & 0x0F) |
                                                        (((qh_half[l] >> 2) & 0x03) << 4)) - 32;
                        output[out_index++] = d * static_cast<float>(sc_half[2] * q1);
                    }
                    for (size_t l = 0; l < 32 && out_index < decode_count; ++l) {
                        const int q2 = static_cast<int>(((ql_half[l] >> 4) & 0x0F) |
                                                        (((qh_half[l] >> 4) & 0x03) << 4)) - 32;
                        output[out_index++] = d * static_cast<float>(sc_half[4] * q2);
                    }
                    for (size_t l = 0; l < 32 && out_index < decode_count; ++l) {
                        const int q3 = static_cast<int>(((ql_half[l + 32] >> 4) & 0x0F) |
                                                        (((qh_half[l] >> 6) & 0x03) << 4)) - 32;
                        output[out_index++] = d * static_cast<float>(sc_half[6] * q3);
                    }
                }
                continue;
            }

            if (format_lower == "q4_k") {
                const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block));
                const float dmin = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block + 2));
                const uint8_t* scales = block + 4;
                const uint8_t* qs = block + 16;
                for (size_t group = 0; group < 4 && out_index < decode_count; ++group) {
                    uint8_t scale0 = 0, min0 = 0, scale1 = 0, min1 = 0;
                    gguf_get_scale_min_q4_k(scales, group * 2, scale0, min0);
                    gguf_get_scale_min_q4_k(scales, group * 2 + 1, scale1, min1);
                    const float d0 = d * static_cast<float>(scale0);
                    const float m0 = dmin * static_cast<float>(min0);
                    const float d1 = d * static_cast<float>(scale1);
                    const float m1 = dmin * static_cast<float>(min1);
                    const uint8_t* qgroup = qs + group * 32;
                    for (size_t i = 0; i < 32 && out_index < decode_count; ++i) {
                        output[out_index++] = d0 * static_cast<float>(qgroup[i] & 0x0F) - m0;
                    }
                    for (size_t i = 0; i < 32 && out_index < decode_count; ++i) {
                        output[out_index++] = d1 * static_cast<float>(qgroup[i] >> 4) - m1;
                    }
                }
                continue;
            }

            if (format_lower == "q8_k") {
                const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block));
                const int8_t* qs = reinterpret_cast<const int8_t*>(block + 4);
                for (size_t i = 0; i < 256 && out_index < decode_count; ++i) {
                    output[out_index++] = static_cast<float>(qs[i]) * d;
                }
                continue;
            }

            if (format_lower == "q4_0") {
                const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block));
                const uint8_t* qs = block + 2;
                for (size_t i = 0; i < 32 && out_index < decode_count; ++i) {
                    const uint8_t nibble = (i & 1U) == 0U ? (qs[i / 2] & 0x0F) : (qs[i / 2] >> 4);
                    output[out_index++] = (static_cast<float>(nibble) - 8.0f) * d;
                }
                continue;
            }

            if (format_lower == "q4_1") {
                const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block));
                const float m = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block + 2));
                const uint8_t* qs = block + 4;
                for (size_t i = 0; i < 32 && out_index < decode_count; ++i) {
                    const uint8_t nibble = (i & 1U) == 0U ? (qs[i / 2] & 0x0F) : (qs[i / 2] >> 4);
                    output[out_index++] = static_cast<float>(nibble) * d + m;
                }
                continue;
            }

            if (format_lower == "q5_0" || format_lower == "q5_1") {
                const bool has_m = format_lower == "q5_1";
                const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block));
                const float m = has_m ? fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block + 2)) : 0.0f;
                const uint8_t* qh = block + (has_m ? 4 : 2);
                const uint8_t* qs = block + (has_m ? 8 : 6);
                const uint32_t high_bits =
                    static_cast<uint32_t>(qh[0]) |
                    (static_cast<uint32_t>(qh[1]) << 8) |
                    (static_cast<uint32_t>(qh[2]) << 16) |
                    (static_cast<uint32_t>(qh[3]) << 24);
                for (size_t i = 0; i < 32 && out_index < decode_count; ++i) {
                    uint8_t q = (i & 1U) == 0U ? (qs[i / 2] & 0x0F) : (qs[i / 2] >> 4);
                    q |= static_cast<uint8_t>(((high_bits >> i) & 0x01U) << 4);
                    output[out_index++] = has_m ? (static_cast<float>(q) * d + m)
                                                : ((static_cast<float>(q) - 16.0f) * d);
                }
                continue;
            }

            const size_t bits = gguf_quant_bits_for_format(format_lower);
            if (bits == 0) {
                return false;
            }
            const size_t packed_bytes = (block_elems * bits + 7) / 8;
            if (packed_bytes == 0 || packed_bytes > block_bytes) {
                return false;
            }
            const size_t data_offset = block_bytes - packed_bytes;
            const uint8_t* packed = block + data_offset;

            float d = 1.0f;
            if (block_bytes >= 2) {
                d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block));
                if (!std::isfinite(d) || std::abs(d) < 1e-12f) {
                    d = 1.0f;
                }
            }

            const int32_t centered_offset = static_cast<int32_t>(1U << (bits - 1));
            for (size_t i = 0; i < block_elems && out_index < decode_count; ++i) {
                const uint32_t q = read_packed_bits_lsb(packed, i * bits, bits);
                const int32_t signed_q = static_cast<int32_t>(q) - centered_offset;
                output[out_index++] = static_cast<float>(signed_q) * d;
            }
        }

        return out_index == decode_count;
    }

    bool is_token_embedding_name(const std::string& name_in) {
        const std::string n = to_lower_copy(name_in);
        return n.find("token_embd") != std::string::npos ||
               n.find("tok_embeddings") != std::string::npos ||
               n.find("embed_tokens") != std::string::npos;
    }

    bool is_output_projection_name(const std::string& name_in) {
        const std::string n = to_lower_copy(name_in);
        return n == "output" ||
               n.find("output.weight") != std::string::npos ||
               n.find("lm_head") != std::string::npos;
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
        engine.transformer_blocks.clear();
        engine.output_norm_layer_name.clear();
        engine.output_projection_layer_name.clear();

        const auto& segments = engine.model_loader->getSegmentIndex();
        std::map<size_t, CortexInferenceEngine::TransformerBlock> blocks_by_index;
        const std::regex blk_regex(R"(blk\.(\d+)\.)");
        for (const auto& seg : segments) {
            const std::string candidate = !seg.layer_name.empty() ? seg.layer_name : seg.name;
            const std::string lower_candidate = to_lower_copy(candidate);
            if (engine.token_embedding_layer_name.empty() && is_token_embedding_name(candidate)) {
                engine.token_embedding_layer_name = candidate;
                if (seg.output_shape.size() >= 2) {
                    engine.native_hidden_dim = seg.output_shape.back();
                } else if (seg.input_shape.size() >= 2) {
                    engine.native_hidden_dim = seg.input_shape.back();
                }
            }

            if (engine.output_norm_layer_name.empty() &&
                (lower_candidate == "output_norm" || lower_candidate.find("model.norm") != std::string::npos)) {
                engine.output_norm_layer_name = candidate;
            }

            if (engine.output_projection_layer_name.empty() &&
                is_output_projection_name(candidate)) {
                engine.output_projection_layer_name = candidate;
            }

            std::smatch match;
            if (std::regex_search(candidate, match, blk_regex) && match.size() > 1) {
                const size_t block_index = static_cast<size_t>(std::stoull(match[1].str()));
                auto& block = blocks_by_index[block_index];
                if (lower_candidate.find("attn_norm") != std::string::npos) block.attn_norm = candidate;
                else if (lower_candidate.find("attn_q") != std::string::npos) block.attn_q = candidate;
                else if (lower_candidate.find("attn_k") != std::string::npos) block.attn_k = candidate;
                else if (lower_candidate.find("attn_v") != std::string::npos) block.attn_v = candidate;
                else if (lower_candidate.find("attn_output") != std::string::npos) block.attn_output = candidate;
                else if (lower_candidate.find("ffn_norm") != std::string::npos) block.ffn_norm = candidate;
                else if (lower_candidate.find("ffn_gate") != std::string::npos) block.ffn_gate = candidate;
                else if (lower_candidate.find("ffn_up") != std::string::npos) block.ffn_up = candidate;
                else if (lower_candidate.find("ffn_down") != std::string::npos) block.ffn_down = candidate;
            }
        }

        auto execution_order = engine.inference_engine->getExecutionOrder(segments);
        for (const auto& layer_name : execution_order) {
            if (is_decoder_excluded_name(layer_name)) {
                continue;
            }
            engine.decoder_layer_order.push_back(layer_name);
        }

        for (auto& entry : blocks_by_index) {
            if (entry.second.complete()) {
                engine.transformer_blocks.push_back(entry.second);
            }
        }

        engine.native_decode_ready =
            engine.tokenizer && engine.tokenizer->isLoaded() &&
            !engine.token_embedding_layer_name.empty() &&
            (!engine.transformer_blocks.empty() || !engine.decoder_layer_order.empty());
        return engine.native_decode_ready;
    }

    struct EmbeddingLayout {
        size_t vocab_size = 0;
        size_t embedding_dim = 0;
        bool valid() const { return vocab_size > 0 && embedding_dim > 0; }
    };

    struct EmbeddingStorageLayout {
        size_t rows = 0;
        size_t cols = 0;
        bool valid() const { return rows > 0 && cols > 0; }
    };

    EmbeddingLayout infer_embedding_layout(
        const CortexInferenceEngine& engine,
        const LayerInfo& token_embedding,
        int token_id_hint = -1) {
        EmbeddingLayout best{};
        const size_t total = token_embedding.weights.size();
        const bool has_dense_weights = total > 0;
        if (!has_dense_weights &&
            token_embedding.input_shape.empty() &&
            token_embedding.output_shape.empty()) {
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
            if (has_dense_weights && vocab * dim > total) continue;
            if (token_id_hint >= 0 && static_cast<size_t>(token_id_hint) >= vocab) continue;
            if (dim < 64 || dim > 16384) continue;
            if (!best.valid() || vocab > best.vocab_size) {
                best = {vocab, dim};
            }
        }
        return best;
    }

    size_t bytes_for_tensor_row(const std::string& format_lower, size_t elements) {
        if (elements == 0) {
            return 0;
        }
        if (is_gguf_quantized_format(format_lower)) {
            const size_t block_elems = gguf_block_elements_for_format(format_lower);
            const size_t block_bytes = gguf_block_bytes_for_format(format_lower);
            if (block_elems == 0 || block_bytes == 0) {
                return 0;
            }
            const size_t blocks = (elements + block_elems - 1) / block_elems;
            return blocks * block_bytes;
        }
        if (format_lower == "f16" || format_lower == "fp16" || format_lower == "bf16") {
            return elements * sizeof(uint16_t);
        }
        if (format_lower == "f32" || format_lower == "fp32" || format_lower.empty()) {
            return elements * sizeof(float);
        }
        return 0;
    }

    EmbeddingStorageLayout infer_embedding_storage_layout(
        const LayerInfo& token_embedding,
        const EmbeddingLayout& layout) {
        EmbeddingStorageLayout storage{};
        if (!layout.valid()) {
            return storage;
        }

        if (!token_embedding.weights.empty() &&
            token_embedding.weights.size() == layout.vocab_size * layout.embedding_dim) {
            storage.rows = layout.vocab_size;
            storage.cols = layout.embedding_dim;
            return storage;
        }

        if (!token_embedding.raw_data.empty()) {
            const std::string format_lower = to_lower_copy(token_embedding.data_format);
            const size_t token_major_row_bytes = bytes_for_tensor_row(format_lower, layout.embedding_dim);
            if (token_major_row_bytes > 0 &&
                token_major_row_bytes * layout.vocab_size == token_embedding.raw_data.size()) {
                storage.rows = layout.vocab_size;
                storage.cols = layout.embedding_dim;
                return storage;
            }

            const size_t transposed_row_bytes = bytes_for_tensor_row(format_lower, layout.vocab_size);
            if (transposed_row_bytes > 0 &&
                transposed_row_bytes * layout.embedding_dim == token_embedding.raw_data.size()) {
                storage.rows = layout.embedding_dim;
                storage.cols = layout.vocab_size;
                return storage;
            }
        }

        if (token_embedding.input_shape.size() >= 2 && token_embedding.output_shape.size() >= 2) {
            storage.rows = token_embedding.input_shape.back();
            storage.cols = token_embedding.output_shape.back();
            return storage;
        }

        storage.rows = layout.vocab_size;
        storage.cols = layout.embedding_dim;
        return storage;
    }

    bool decode_embedding_storage_row(
        const LayerInfo& token_embedding,
        const std::string& format_lower,
        size_t row_index,
        size_t row_elements,
        std::vector<float>& out) {
        out.clear();
        if (row_elements == 0) {
            return false;
        }

        if (!token_embedding.weights.empty()) {
            const size_t row_offset = row_index * row_elements;
            if (row_offset + row_elements > token_embedding.weights.size()) {
                return false;
            }
            out.resize(row_elements);
            const float* row = token_embedding.weights.data() + row_offset;
            std::copy(row, row + row_elements, out.begin());
            return true;
        }

        if (token_embedding.raw_data.empty()) {
            return false;
        }

        const size_t row_bytes = bytes_for_tensor_row(format_lower, row_elements);
        if (row_bytes == 0) {
            return false;
        }
        const size_t row_offset = row_index * row_bytes;
        if (row_offset + row_bytes > token_embedding.raw_data.size()) {
            return false;
        }
        const std::byte* row_begin = token_embedding.raw_data.data() + row_offset;

        if (is_gguf_quantized_format(format_lower)) {
            return decode_supported_gguf_quantized(row_begin, row_bytes, format_lower, row_elements, out);
        }

        if (format_lower == "f16" || format_lower == "fp16") {
            out.resize(row_elements);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(row_begin);
            for (size_t i = 0; i < row_elements; ++i) {
                out[i] = fp16_to_fp32(src[i]);
            }
            return true;
        }

        if (format_lower == "bf16") {
            out.resize(row_elements);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(row_begin);
            for (size_t i = 0; i < row_elements; ++i) {
                out[i] = bf16_to_fp32(src[i]);
            }
            return true;
        }

        if (format_lower == "f32" || format_lower == "fp32" || format_lower.empty()) {
            out.resize(row_elements);
            std::memcpy(out.data(), row_begin, row_elements * sizeof(float));
            return true;
        }

        return false;
    }

    bool decode_embedding_storage_value(
        const LayerInfo& token_embedding,
        const std::string& format_lower,
        size_t row_index,
        size_t row_elements,
        size_t column_index,
        float& value) {
        if (column_index >= row_elements) {
            return false;
        }

        if (!token_embedding.weights.empty()) {
            const size_t offset = row_index * row_elements + column_index;
            if (offset >= token_embedding.weights.size()) {
                return false;
            }
            value = token_embedding.weights[offset];
            return true;
        }

        if (token_embedding.raw_data.empty()) {
            return false;
        }

        const size_t row_bytes = bytes_for_tensor_row(format_lower, row_elements);
        if (row_bytes == 0) {
            return false;
        }
        const size_t row_offset = row_index * row_bytes;
        if (row_offset + row_bytes > token_embedding.raw_data.size()) {
            return false;
        }
        const std::byte* row_begin = token_embedding.raw_data.data() + row_offset;

        if (is_gguf_quantized_format(format_lower)) {
            const size_t block_elems = gguf_block_elements_for_format(format_lower);
            const size_t block_bytes = gguf_block_bytes_for_format(format_lower);
            if (block_elems == 0 || block_bytes == 0) {
                return false;
            }
            const size_t block_index = column_index / block_elems;
            const size_t within_block = column_index % block_elems;
            const size_t block_offset = block_index * block_bytes;
            if (block_offset + block_bytes > row_bytes) {
                return false;
            }
            std::vector<float> block_values;
            if (!decode_supported_gguf_quantized(
                    row_begin + block_offset,
                    block_bytes,
                    format_lower,
                    block_elems,
                    block_values) ||
                within_block >= block_values.size()) {
                return false;
            }
            value = block_values[within_block];
            return true;
        }

        if (format_lower == "f16" || format_lower == "fp16") {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(row_begin);
            value = fp16_to_fp32(src[column_index]);
            return true;
        }

        if (format_lower == "bf16") {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(row_begin);
            value = bf16_to_fp32(src[column_index]);
            return true;
        }

        if (format_lower == "f32" || format_lower == "fp32" || format_lower.empty()) {
            const float* src = reinterpret_cast<const float*>(row_begin);
            value = src[column_index];
            return true;
        }

        return false;
    }

    bool decode_embedding_row(
        const LayerInfo& token_embedding,
        const EmbeddingLayout& layout,
        int token_id,
        std::vector<float>& out) {
        out.clear();
        if (!layout.valid() || token_id < 0 || static_cast<size_t>(token_id) >= layout.vocab_size) {
            return false;
        }
        const std::string format_lower = to_lower_copy(token_embedding.data_format);
        const EmbeddingStorageLayout storage = infer_embedding_storage_layout(token_embedding, layout);
        if (!storage.valid()) {
            return false;
        }

        if (storage.rows == layout.vocab_size && storage.cols == layout.embedding_dim) {
            return decode_embedding_storage_row(
                token_embedding,
                format_lower,
                static_cast<size_t>(token_id),
                layout.embedding_dim,
                out);
        }

        if (storage.rows == layout.embedding_dim && storage.cols == layout.vocab_size) {
            out.resize(layout.embedding_dim);
            for (size_t dim_row = 0; dim_row < layout.embedding_dim; ++dim_row) {
                float value = 0.0f;
                if (!decode_embedding_storage_value(
                        token_embedding,
                        format_lower,
                        dim_row,
                        layout.vocab_size,
                        static_cast<size_t>(token_id),
                        value)) {
                    return false;
                }
                out[dim_row] = value;
            }
            return true;
        }

        return false;
    }

    std::vector<float> embedding_lookup(
        const CortexInferenceEngine& engine,
        const LayerInfo& token_embedding,
        int token_id) {
        if (token_id < 0 || (token_embedding.weights.empty() && token_embedding.raw_data.empty())) {
            return {};
        }
        const EmbeddingLayout layout = infer_embedding_layout(engine, token_embedding, token_id);
        if (!layout.valid() || static_cast<size_t>(token_id) >= layout.vocab_size) {
            return {};
        }
        std::vector<float> out;
        if (!decode_embedding_row(token_embedding, layout, token_id, out)) {
            return {};
        }
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
        float total_weight = 0.0f;
        const size_t count = prompt_tokens.size();
        const size_t start = count > 6 ? count - 6 : 0;
        for (size_t idx = start; idx < count; ++idx) {
            const int token_id = prompt_tokens[idx];
            if (engine.tokenizer &&
                (token_id == engine.tokenizer->bosId() || token_id == engine.tokenizer->eosId())) {
                continue;
            }
            auto emb = embedding_lookup(engine, token_embedding, token_id);
            if (emb.empty()) {
                continue;
            }
            const float relative = static_cast<float>(idx - start + 1) / static_cast<float>(count - start + 1);
            float weight = 0.1f + 0.9f * relative * relative;
            if (engine.tokenizer) {
                auto token = engine.tokenizer->idToToken(token_id);
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

    bool token_is_repeated(const std::unordered_set<int>& already_generated_set, int tok) {
        return already_generated_set.find(tok) != already_generated_set.end();
    }

    bool has_any_finite_value(const std::vector<float>& values) {
        for (float value : values) {
            if (std::isfinite(value)) {
                return true;
            }
        }
        return false;
    }

    std::vector<float> sanitize_hidden_for_sampling(const std::vector<float>& hidden) {
        std::vector<float> sanitized(hidden);
        float max_abs = 0.0f;
        for (float& value : sanitized) {
            if (!std::isfinite(value)) {
                value = 0.0f;
                continue;
            }
            max_abs = (std::max)(max_abs, std::fabs(value));
        }
        if (max_abs > 1000.0f) {
            const float scale = 1000.0f / max_abs;
            for (float& value : sanitized) {
                value *= scale;
            }
        }
        return sanitized;
    }

    float finite_dot_product(const std::vector<float>& a, const std::vector<float>& b) {
        const size_t count = (std::min)(a.size(), b.size());
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            const float lhs = a[i];
            const float rhs = b[i];
            if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
                continue;
            }
            sum += lhs * rhs;
        }
        return sum;
    }

    float finite_dot_product(const float* a, const std::vector<float>& b, size_t count) {
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            const float lhs = a[i];
            const float rhs = b[i];
            if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
                continue;
            }
            sum += lhs * rhs;
        }
        return sum;
    }

    int select_best_from_logits(
        const std::vector<float>& logits,
        const std::unordered_set<int>& already_generated_set,
        float temperature) {
        if (logits.empty()) {
            return -1;
        }
        const float temp_scale = temperature > 0.0f ? (1.0f / temperature) : 1.0f;
        float best_score = -std::numeric_limits<float>::infinity();
        int best_token = -1;
        for (size_t token = 0; token < logits.size(); ++token) {
            float score = logits[token] * temp_scale;
            if (!std::isfinite(score)) {
                continue;
            }
            const int tok = static_cast<int>(token);
            if (token_is_repeated(already_generated_set, tok)) {
                score -= 1.5f;
            }
            if (score > best_score) {
                best_score = score;
                best_token = tok;
            }
        }
        return best_token;
    }

    int select_next_token_from_embedding_transpose(
        const CortexInferenceEngine& engine,
        const std::vector<float>& hidden,
        const LayerInfo& token_embedding,
        const std::vector<int>& already_generated,
        float temperature = 0.8f) {
        const EmbeddingLayout layout = infer_embedding_layout(engine, token_embedding);
        if (!layout.valid() || hidden.size() != layout.embedding_dim || !has_any_finite_value(hidden)) {
            return -1;
        }
        const std::vector<float> stable_hidden = sanitize_hidden_for_sampling(hidden);

        const std::string format_lower = to_lower_copy(token_embedding.data_format);
        std::unordered_set<int> already_generated_set;
        already_generated_set.reserve(already_generated.size() * 2);
        for (int tok : already_generated) {
            already_generated_set.insert(tok);
        }
        const EmbeddingStorageLayout storage = infer_embedding_storage_layout(token_embedding, layout);

        if (storage.valid() &&
            storage.rows == layout.embedding_dim &&
            storage.cols == layout.vocab_size) {
            std::vector<float> logits(layout.vocab_size, 0.0f);
            std::vector<float> row;
            bool decode_failed = false;
            const size_t row_limit =
                engine.fast_projection_row_limit > 0
                    ? (std::min)(layout.embedding_dim, engine.fast_projection_row_limit)
                    : layout.embedding_dim;
            const size_t stride = (std::max)(static_cast<size_t>(1), layout.embedding_dim / row_limit);
            for (size_t dim_row = 0; dim_row < layout.embedding_dim; dim_row += stride) {
                if (!decode_embedding_storage_row(
                        token_embedding,
                        format_lower,
                        dim_row,
                        layout.vocab_size,
                        row) ||
                    row.size() != layout.vocab_size) {
                    decode_failed = true;
                    break;
                }
                const float coeff = stable_hidden[dim_row];
                for (size_t token = 0; token < layout.vocab_size; ++token) {
                    logits[token] += coeff * row[token];
                }
            }
            if (!decode_failed) {
                return select_best_from_logits(logits, already_generated_set, temperature);
            }
        }

        if (token_embedding.weights.empty()) {
            const float temp_scale = temperature > 0.0f ? (1.0f / temperature) : 1.0f;
            struct BestCandidate {
                float score = -std::numeric_limits<float>::infinity();
                int token = -1;
            };

            const unsigned int hw_threads = (std::max)(1u, std::thread::hardware_concurrency());
            const size_t desired_workers = (std::max)(static_cast<size_t>(1), layout.vocab_size / 2048);
            const size_t worker_count = (std::min)(static_cast<size_t>(hw_threads), desired_workers);
            std::vector<BestCandidate> best_by_worker(worker_count);
            std::vector<std::thread> workers;
            workers.reserve(worker_count > 0 ? worker_count - 1 : 0);

            auto score_range = [&](size_t worker_index, size_t begin, size_t end) {
                std::vector<float> row;
                BestCandidate local_best;
                for (size_t token = begin; token < end; ++token) {
                    if (!decode_embedding_row(token_embedding, layout, static_cast<int>(token), row)) {
                        continue;
                    }
                    float score = finite_dot_product(row, stable_hidden);
                    if (!std::isfinite(score)) {
                        continue;
                    }
                    score *= temp_scale;

                    const int tok = static_cast<int>(token);
                    if (token_is_repeated(already_generated_set, tok)) {
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
                float score = finite_dot_product(row, stable_hidden, layout.embedding_dim);
                if (!std::isfinite(score)) {
                    continue;
                }
                score *= temp_scale;

                const int tok = static_cast<int>(token);
                if (token_is_repeated(already_generated_set, tok)) {
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

    std::vector<float> rms_norm(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        size_t hidden_dim,
        float epsilon = 1e-5f) {
        if (hidden_dim == 0 || weights.size() < hidden_dim || input.size() % hidden_dim != 0) {
            return {};
        }
        std::vector<float> output(input.size(), 0.0f);
        const size_t token_count = input.size() / hidden_dim;
        for (size_t token = 0; token < token_count; ++token) {
            const float* src = input.data() + (token * hidden_dim);
            float* dst = output.data() + (token * hidden_dim);
            float mean_square = 0.0f;
            for (size_t i = 0; i < hidden_dim; ++i) {
                mean_square += src[i] * src[i];
            }
            mean_square /= static_cast<float>(hidden_dim);
            const float inv_rms = 1.0f / std::sqrt(mean_square + epsilon);
            for (size_t i = 0; i < hidden_dim; ++i) {
                dst[i] = src[i] * inv_rms * weights[i];
            }
        }
        return output;
    }

    std::vector<float> add_vectors(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            return {};
        }
        std::vector<float> out(a.size(), 0.0f);
        for (size_t i = 0; i < a.size(); ++i) {
            out[i] = a[i] + b[i];
        }
        return out;
    }

    std::vector<float> silu_mul(const std::vector<float>& gate, const std::vector<float>& up) {
        if (gate.size() != up.size()) {
            return {};
        }
        std::vector<float> out(gate.size(), 0.0f);
        for (size_t i = 0; i < gate.size(); ++i) {
            const float x = gate[i];
            const float sig = 1.0f / (1.0f + std::exp(-x));
            out[i] = (x * sig) * up[i];
        }
        return out;
    }

    std::vector<float> run_dense_linear(
        const LayerInfo& layer,
        const std::vector<float>& input) {
        if (layer.weights.empty()) {
            return {};
        }

        size_t input_dim = 0;
        size_t output_dim = 0;
        if (!layer.input_shape.empty()) {
            input_dim = layer.input_shape.back();
        }
        if (!layer.output_shape.empty()) {
            output_dim = layer.output_shape.back();
        }
        if (input_dim == 0 && output_dim == 0 && !layer.biases.empty()) {
            output_dim = layer.biases.size();
            input_dim = layer.weights.size() / output_dim;
        } else if (input_dim == 0 && output_dim > 0) {
            input_dim = layer.weights.size() / output_dim;
        } else if (output_dim == 0 && input_dim > 0) {
            output_dim = layer.weights.size() / input_dim;
        }
        if (input_dim == 0 || output_dim == 0 || (input.size() % input_dim) != 0) {
            return {};
        }
        if (layer.weights.size() < input_dim * output_dim) {
            return {};
        }

        const size_t batch = input.size() / input_dim;
        std::vector<float> output(batch * output_dim, 0.0f);

        CortexAICompression::Kernels::linear_forward(
            input.data(),
            layer.weights.data(),
            layer.biases.empty() ? nullptr : layer.biases.data(),
            output.data(),
            batch,
            input_dim,
            output_dim);

        return output;
    }

    struct AttentionLayout {
        size_t q_heads = 0;
        size_t k_heads = 0;
        size_t v_heads = 0;
        size_t head_dim = 0;
        bool valid() const { return q_heads > 0 && k_heads > 0 && v_heads > 0 && head_dim > 0; }
    };

    AttentionLayout infer_attention_layout(
        size_t seq_len,
        const LayerInfo& q_layer,
        const LayerInfo& k_layer,
        size_t q_linear_size,
        size_t k_linear_size,
        size_t v_linear_size) {
        AttentionLayout layout{};
        if (seq_len == 0) {
            return layout;
        }

        size_t q_dim = q_linear_size / seq_len;
        size_t k_dim = k_linear_size / seq_len;
        size_t v_dim = v_linear_size / seq_len;
        if ((q_dim * seq_len) != q_linear_size ||
            (k_dim * seq_len) != k_linear_size ||
            (v_dim * seq_len) != v_linear_size) {
            return layout;
        }
        if (q_dim == 0 || k_dim == 0) {
            return layout;
        }

        const std::array<size_t, 5> preferred_head_dims = {128, 64, 256, 32, 16};
        for (size_t head_dim : preferred_head_dims) {
            if (q_dim % head_dim != 0 || k_dim % head_dim != 0 || v_dim % head_dim != 0) {
                continue;
            }
            const size_t q_heads = q_dim / head_dim;
            const size_t k_heads = k_dim / head_dim;
            const size_t v_heads = v_dim / head_dim;
            if (q_heads == 0 || k_heads == 0 || v_heads == 0 ||
                q_heads < k_heads || q_heads < v_heads ||
                (q_heads % k_heads) != 0 || (q_heads % v_heads) != 0) {
                continue;
            }
            layout = {q_heads, k_heads, v_heads, head_dim};
            return layout;
        }
        return layout;
    }

    struct RopeCache {
        size_t max_seq_len = 0;
        size_t rotary_dim = 0;
        size_t head_dim = 0;
        float theta = 1000000.0f;
        std::vector<float> cos_table;
        std::vector<float> sin_table;
    };

    RopeCache& get_rope_cache(size_t max_seq_len, size_t head_dim, float theta = 1000000.0f) {
        static RopeCache cache;
        if (cache.max_seq_len >= max_seq_len &&
            cache.head_dim == head_dim &&
            cache.theta == theta) {
            return cache;
        }

        cache.max_seq_len = max_seq_len;
        cache.head_dim = head_dim;
        cache.theta = theta;
        cache.rotary_dim = head_dim - (head_dim % 2);
        const size_t per_pos = cache.rotary_dim;

        cache.cos_table.assign(max_seq_len * per_pos, 0.0f);
        cache.sin_table.assign(max_seq_len * per_pos, 0.0f);

        for (size_t pos = 0; pos < max_seq_len; ++pos) {
            for (size_t i = 0; i < cache.rotary_dim; i += 2) {
                const float exponent = static_cast<float>(i) / static_cast<float>(head_dim);
                const float freq = std::pow(theta, -exponent);
                const float angle = static_cast<float>(pos) * freq;
                const float cos_angle = std::cos(angle);
                const float sin_angle = std::sin(angle);
                cache.cos_table[pos * per_pos + i] = cos_angle;
                cache.cos_table[pos * per_pos + i + 1] = cos_angle;
                cache.sin_table[pos * per_pos + i] = sin_angle;
                cache.sin_table[pos * per_pos + i + 1] = sin_angle;
            }
        }
        return cache;
    }

    void apply_rope_inplace(
        std::vector<float>& tensor,
        size_t seq_len,
        size_t num_heads,
        size_t head_dim,
        float theta = 1000000.0f) {
        if (seq_len == 0 || num_heads == 0 || head_dim < 2) {
            return;
        }
        RopeCache& cache = get_rope_cache(seq_len, head_dim, theta);
        const size_t rotary_dim = cache.rotary_dim;
        const size_t per_pos = rotary_dim;
        for (size_t pos = 0; pos < seq_len; ++pos) {
            const float* cos_row = cache.cos_table.data() + pos * per_pos;
            const float* sin_row = cache.sin_table.data() + pos * per_pos;
            for (size_t head = 0; head < num_heads; ++head) {
                float* base = tensor.data() + ((pos * num_heads + head) * head_dim);
                for (size_t i = 0; i < rotary_dim; i += 2) {
                    const float cos_angle = cos_row[i];
                    const float sin_angle = sin_row[i];
                    const float x0 = base[i];
                    const float x1 = base[i + 1];
                    base[i] = x0 * cos_angle - x1 * sin_angle;
                    base[i + 1] = x0 * sin_angle + x1 * cos_angle;
                }
            }
        }
    }

    std::vector<float> reshape_heads(
        const std::vector<float>& input,
        size_t seq_len,
        size_t num_heads,
        size_t head_dim) {
        if (input.size() != seq_len * num_heads * head_dim) {
            return {};
        }
        std::vector<float> out(input.size(), 0.0f);
        for (size_t pos = 0; pos < seq_len; ++pos) {
            for (size_t head = 0; head < num_heads; ++head) {
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    out[(pos * num_heads + head) * head_dim + dim] =
                        input[pos * (num_heads * head_dim) + head * head_dim + dim];
                }
            }
        }
        return out;
    }

    std::vector<float> flatten_heads(
        const std::vector<float>& input,
        size_t seq_len,
        size_t num_heads,
        size_t head_dim) {
        return reshape_heads(input, seq_len, num_heads, head_dim);
    }

    std::vector<float> repeat_kv_heads(
        const std::vector<float>& input,
        size_t seq_len,
        size_t kv_heads,
        size_t q_heads,
        size_t head_dim) {
        if (kv_heads == 0 || q_heads == 0 || (q_heads % kv_heads) != 0 ||
            input.size() != seq_len * kv_heads * head_dim) {
            return {};
        }
        if (kv_heads == q_heads) {
            return input;
        }
        const size_t repeat = q_heads / kv_heads;
        std::vector<float> out(seq_len * q_heads * head_dim, 0.0f);
        for (size_t pos = 0; pos < seq_len; ++pos) {
            for (size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
                const float* src = input.data() + ((pos * kv_heads + kv_head) * head_dim);
                for (size_t rep = 0; rep < repeat; ++rep) {
                    float* dst = out.data() + ((pos * q_heads + kv_head * repeat + rep) * head_dim);
                    std::copy(src, src + head_dim, dst);
                }
            }
        }
        return out;
    }

    std::vector<float> build_token_embeddings(
        CortexInferenceEngine& engine,
        const LayerInfo& token_embedding,
        const std::vector<int>& token_ids) {
        const EmbeddingLayout layout = infer_embedding_layout(engine, token_embedding);
        if (!layout.valid() || token_ids.empty()) {
            return {};
        }

        std::vector<float> out;
        out.reserve(token_ids.size() * layout.embedding_dim);
        for (int token_id : token_ids) {
            auto row = embedding_lookup(engine, token_embedding, token_id);
            if (row.empty()) {
                std::ostringstream error;
                error << "embedding lookup failed for token " << token_id
                      << " (vocab=" << layout.vocab_size
                      << ", dim=" << layout.embedding_dim
                      << ", format=" << token_embedding.data_format
                      << ", raw_bytes=" << token_embedding.raw_data.size()
                      << ", dense_weights=" << token_embedding.weights.size() << ")";
                throw std::runtime_error(error.str());
            }
            out.insert(out.end(), row.begin(), row.end());
        }
        return out;
    }

    bool can_use_logits_projection(
        const CortexInferenceEngine& engine,
        const LayerInfo& projection_layer) {
        if (projection_layer.weights.empty() && projection_layer.raw_data.empty()) {
            return false;
        }
        const EmbeddingLayout layout = infer_embedding_layout(engine, projection_layer);
        if (!layout.valid() || layout.vocab_size == 0 || layout.embedding_dim == 0) {
            return false;
        }
        std::vector<float> row;
        return decode_embedding_row(projection_layer, layout, 0, row) && row.size() == layout.embedding_dim;
    }

    const LayerInfo& load_cached_layer(
        CortexInferenceEngine& engine,
        const std::string& layer_name) {
        auto it = engine.layer_cache.find(layer_name);
        if (it == engine.layer_cache.end()) {
            auto loaded = std::make_unique<LayerInfo>(engine.model_loader->loadLayerByName(layer_name));
            it = engine.layer_cache.emplace(layer_name, std::move(loaded)).first;
        }
        return *it->second;
    }

    std::vector<float> run_cached_linear(
        CortexInferenceEngine& engine,
        const LayerInfo& layer,
        const std::vector<float>& input) {
        if (!layer.weights.empty()) {
            return run_dense_linear(layer, input);
        }
        return engine.inference_engine->runLayer(layer, input);
    }

    std::vector<float> run_transformer_decode(
        CortexInferenceEngine& engine,
        const LayerInfo& token_embedding,
        const std::vector<int>& tokens) {
        if (engine.transformer_blocks.empty()) {
            throw std::runtime_error("No complete GGUF transformer blocks were discovered");
        }
        const EmbeddingLayout embed_layout = infer_embedding_layout(engine, token_embedding);
        if (!embed_layout.valid()) {
            throw std::runtime_error("Failed to infer token embedding layout");
        }

        const size_t seq_len = tokens.size();
        const size_t hidden_dim = embed_layout.embedding_dim;
        std::vector<float> hidden = build_token_embeddings(engine, token_embedding, tokens);
        if (hidden.size() != seq_len * hidden_dim) {
            std::ostringstream error;
            error << "Failed to build prompt token embeddings"
                  << " (seq_len=" << seq_len
                  << ", hidden_dim=" << hidden_dim
                  << ", format=" << token_embedding.data_format
                  << ", raw_bytes=" << token_embedding.raw_data.size()
                  << ", dense_weights=" << token_embedding.weights.size() << ")";
            throw std::runtime_error(error.str());
        }

        for (size_t block_idx = 0; block_idx < engine.transformer_blocks.size(); ++block_idx) {
            const auto& block_names = engine.transformer_blocks[block_idx];

            const LayerInfo& attn_norm = load_cached_layer(engine, block_names.attn_norm);
            std::vector<float> normed = rms_norm(hidden, attn_norm.weights, hidden_dim);
            if (normed.empty()) {
                throw std::runtime_error("RMSNorm failed for block " + std::to_string(block_idx) + " attention input");
            }

            const LayerInfo& attn_q = load_cached_layer(engine, block_names.attn_q);
            std::vector<float> q = run_cached_linear(engine, attn_q, normed);
            const LayerInfo& attn_k = load_cached_layer(engine, block_names.attn_k);
            std::vector<float> k = run_cached_linear(engine, attn_k, normed);
            const LayerInfo& attn_v = load_cached_layer(engine, block_names.attn_v);
            std::vector<float> v = run_cached_linear(engine, attn_v, normed);
            if (q.empty() || k.empty() || v.empty()) {
                throw std::runtime_error("QKV projection failed for block " + std::to_string(block_idx));
            }

            const size_t q_linear_size = q.size();
            const size_t k_linear_size = k.size();
            const size_t v_linear_size = v.size();
            const AttentionLayout attn_layout = infer_attention_layout(
                seq_len, attn_q, attn_k, q_linear_size, k_linear_size, v_linear_size);
            if (!attn_layout.valid()) {
                throw std::runtime_error("Failed to infer attention layout for block " + std::to_string(block_idx));
            }
            q = reshape_heads(q, seq_len, attn_layout.q_heads, attn_layout.head_dim);
            k = reshape_heads(k, seq_len, attn_layout.k_heads, attn_layout.head_dim);
            v = reshape_heads(v, seq_len, attn_layout.v_heads, attn_layout.head_dim);
            if (q.empty() || k.empty() || v.empty()) {
                std::ostringstream error;
                error << "Attention head reshape failed for block " << block_idx
                      << " (seq_len=" << seq_len
                      << ", q_heads=" << attn_layout.q_heads
                      << ", k_heads=" << attn_layout.k_heads
                      << ", v_heads=" << attn_layout.v_heads
                      << ", head_dim=" << attn_layout.head_dim
                      << ", q_linear_size=" << q_linear_size
                      << ", k_linear_size=" << k_linear_size
                      << ", v_linear_size=" << v_linear_size << ")";
                throw std::runtime_error(error.str());
            }

            apply_rope_inplace(q, seq_len, attn_layout.q_heads, attn_layout.head_dim);
            apply_rope_inplace(k, seq_len, attn_layout.k_heads, attn_layout.head_dim);

            std::vector<float> k_expanded = repeat_kv_heads(k, seq_len, attn_layout.k_heads, attn_layout.q_heads, attn_layout.head_dim);
            std::vector<float> v_expanded = repeat_kv_heads(v, seq_len, attn_layout.v_heads, attn_layout.q_heads, attn_layout.head_dim);
            if (k_expanded.empty() || v_expanded.empty()) {
                throw std::runtime_error("KV expansion failed for block " + std::to_string(block_idx));
            }

            std::vector<float> attn_out(seq_len * hidden_dim, 0.0f);
            CortexAICompression::Kernels::multi_head_attention(
                q.data(),
                k_expanded.data(),
                v_expanded.data(),
                attn_out.data(),
                1,
                seq_len,
                hidden_dim,
                attn_layout.q_heads,
                true
            );

            attn_out = flatten_heads(attn_out, seq_len, attn_layout.q_heads, attn_layout.head_dim);
            const LayerInfo& attn_output = load_cached_layer(engine, block_names.attn_output);
            attn_out = run_cached_linear(engine, attn_output, attn_out);
            if (attn_out.empty()) {
                throw std::runtime_error("Attention output projection failed for block " + std::to_string(block_idx));
            }

            hidden = add_vectors(hidden, attn_out);
            if (hidden.empty()) {
                throw std::runtime_error("Attention residual add failed for block " + std::to_string(block_idx));
            }

            const LayerInfo& ffn_norm = load_cached_layer(engine, block_names.ffn_norm);
            normed = rms_norm(hidden, ffn_norm.weights, hidden_dim);
            if (normed.empty()) {
                throw std::runtime_error("RMSNorm failed for block " + std::to_string(block_idx) + " feed-forward input");
            }

            const LayerInfo& ffn_gate = load_cached_layer(engine, block_names.ffn_gate);
            std::vector<float> gate = run_cached_linear(engine, ffn_gate, normed);
            const LayerInfo& ffn_up = load_cached_layer(engine, block_names.ffn_up);
            std::vector<float> up = run_cached_linear(engine, ffn_up, normed);
            if (gate.empty() || up.empty()) {
                throw std::runtime_error("Feed-forward gate/up projection failed for block " + std::to_string(block_idx));
            }

            std::vector<float> ff = silu_mul(gate, up);
            if (ff.empty()) {
                throw std::runtime_error("SwiGLU activation failed for block " + std::to_string(block_idx));
            }
            const LayerInfo& ffn_down = load_cached_layer(engine, block_names.ffn_down);
            ff = run_cached_linear(engine, ffn_down, ff);
            if (ff.empty()) {
                throw std::runtime_error("Feed-forward down projection failed for block " + std::to_string(block_idx));
            }

            hidden = add_vectors(hidden, ff);
            if (hidden.empty()) {
                std::ostringstream error;
                error << "Feed-forward residual add failed for block " << block_idx
                      << " (hidden_size=" << hidden_dim * seq_len
                      << ", ff_size=" << ff.size() << ")";
                throw std::runtime_error(error.str());
            }
        }

        if (!engine.output_norm_layer_name.empty()) {
            const LayerInfo& output_norm = load_cached_layer(engine, engine.output_norm_layer_name);
            auto normed = rms_norm(hidden, output_norm.weights, hidden_dim);
            if (!normed.empty()) {
                hidden = std::move(normed);
            }
        }

        std::vector<float> last_hidden(hidden_dim, 0.0f);
        std::copy(
            hidden.end() - static_cast<std::ptrdiff_t>(hidden_dim),
            hidden.end(),
            last_hidden.begin()
        );
        return last_hidden;
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
        if (token_embedding.weights.empty() && token_embedding.raw_data.empty()) {
            return {"Token embedding layer has no accessible backing data", CORTEX_ERROR_INFERENCE};
        }
        const LayerInfo* output_projection = &token_embedding;
        if (!engine->output_projection_layer_name.empty()) {
            const LayerInfo& candidate = load_cached_layer(*engine, engine->output_projection_layer_name);
            if (can_use_logits_projection(*engine, candidate)) {
                output_projection = &candidate;
            }
        }

        const int step_limit = (std::max)(1, max_new_tokens);
        std::vector<int> generated;
        generated.reserve(static_cast<size_t>(step_limit));

        std::vector<float> decoder_out;

        for (int step = 0; step < step_limit; ++step) {
            std::vector<int> sampling_history = generated;
            if (!engine->prefer_fast_generation && !engine->transformer_blocks.empty()) {
                std::vector<int> full_context = prompt_tokens;
                full_context.insert(full_context.end(), generated.begin(), generated.end());
                decoder_out = run_transformer_decode(*engine, token_embedding, full_context);
            } else {
                std::vector<int> full_context = prompt_tokens;
                full_context.insert(full_context.end(), generated.begin(), generated.end());
                decoder_out = build_prompt_context_hidden(*engine, token_embedding, full_context);
                sampling_history.insert(sampling_history.end(), prompt_tokens.begin(), prompt_tokens.end());
            }
            if (decoder_out.empty()) {
                return {"Decoder stack returned empty output", CORTEX_ERROR_INFERENCE};
            }

            const int cur_token = select_next_token_from_embedding_transpose(
                *engine, decoder_out, *output_projection, sampling_history, 0.8f);
            if (cur_token < 0) {
                size_t finite_hidden = 0;
                float max_abs_hidden = 0.0f;
                for (float value : decoder_out) {
                    if (std::isfinite(value)) {
                        ++finite_hidden;
                        max_abs_hidden = (std::max)(max_abs_hidden, std::fabs(value));
                    }
                }
                const EmbeddingLayout projection_layout = infer_embedding_layout(*engine, *output_projection);
                std::vector<float> projection_probe;
                const bool projection_probe_ok =
                    projection_layout.valid() &&
                    decode_embedding_row(*output_projection, projection_layout, 0, projection_probe);
                std::ostringstream error;
                error << "Failed to sample next token"
                      << " (hidden_size=" << decoder_out.size()
                      << ", finite_hidden=" << finite_hidden
                      << ", max_abs_hidden=" << max_abs_hidden
                      << ", projection_layer=" << (!engine->output_projection_layer_name.empty() ? engine->output_projection_layer_name : engine->token_embedding_layer_name)
                      << ", projection_format=" << output_projection->data_format
                      << ", projection_weights=" << output_projection->weights.size()
                      << ", projection_raw_bytes=" << output_projection->raw_data.size()
                      << ", projection_vocab=" << projection_layout.vocab_size
                      << ", projection_dim=" << projection_layout.embedding_dim
                      << ", projection_probe_ok=" << (projection_probe_ok ? "true" : "false")
                      << ", projection_probe_size=" << projection_probe.size() << ")";
                return {str_to_c(error.str()), CORTEX_ERROR_INFERENCE};
            }
            generated.push_back(cur_token);
            if (engine->tokenizer->eosId() >= 0 &&
                generated.back() == engine->tokenizer->eosId()) {
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
        bool has_vocab = false, has_merges = false, has_spm = false, has_gpt2 = false;
        for (const auto& seg : segments) {
            std::string n = seg.name;
            for (auto& ch : n) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
            if (n.find("tokenizer.model") != std::string::npos ||
                n.find("sentencepiece") != std::string::npos ||
                n.find(".spm") != std::string::npos) {
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
            if (n == "gguf_tokenizer_model") {
                ModelSegment model_segment = loader.loadSegmentByName(seg.name);
                const std::string model_text(
                    reinterpret_cast<const char*>(model_segment.data.data()),
                    model_segment.data.size());
                const std::string lower_model = to_lower_copy(model_text);
                has_spm = has_spm || lower_model.find("sentencepiece") != std::string::npos;
                has_gpt2 = has_gpt2 || lower_model.find("gpt2") != std::string::npos;
            }
        }
        if (has_spm || has_gpt2 || (has_vocab && has_merges)) {
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
