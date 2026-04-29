/**
 * @file SparseInferenceEngine.cpp
 * @brief Implementation of sparse neural network inference engine with on-demand layer loading
 * 
 * This file implements the SparseInferenceEngine and SDRModelLoader classes that provide
 * efficient neural network inference with on-demand layer loading capabilities, similar
 * to modern LLM inference engines like Ollama.
 * 
 * Key features:
 * - On-demand layer loading to minimize memory usage
 * - Support for various neural network layer types
 * - Compressed model format support (.sdr files)
 * - Dynamic layer execution with fallback mechanisms
 * - BLAS-accelerated linear algebra operations
 * - SIMD-optimized activations and element-wise ops
 * - Im2col + GEMM convolution for CNNs
 */

#include "SparseInferenceEngine.hpp"
#include "kernels/blas_kernels.hpp"
#include "kernels/simd_kernels.hpp"
#include "kernels/conv_kernels.hpp"
#include "kernels/fused_kernels.hpp"
#include "kernels/quantized_kernels.hpp"
#include "kernels/attention_kernels.hpp"
#include "kernels/flash_attention.hpp"
#include "kernels/sparse_kernels.hpp"
#include "utils/execution_graph.hpp"
#include "utils/fp16_convert.hpp"
#include "utils/kv_cache.hpp"
#include "utils/tensor_validator.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <unordered_map>
#include <random>
#include "core/ArchiveConstants.hpp"
#include <functional>
#include <future>
#include <thread>
#include <regex>
#include <chrono>
#include <mutex>
#include <memory>

// Compression strategy implementations
#include "strategies/AdaptiveSDRStrategy.hpp"
#include "strategies/GzipStrategy.hpp"
#include "strategies/NumericalRLE.hpp"
#include "strategies/QuantizedTensorStrategy.hpp"
#include "strategies/SDRIndexStorage.hpp"

#ifdef ENABLE_ONNX_PROTOBUF
#include <../onnx_proto/onnx.pb.h>
#endif

namespace CortexAICompression {

// Helper function for FP16 to FP32 conversion
static inline float fp16_to_fp32(uint16_t h) {
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

static bool isWeightLikeSegmentType(SegmentType type) {
    return type == SegmentType::WEIGHTS_FP32 ||
           type == SegmentType::WEIGHTS_FP16 ||
           type == SegmentType::WEIGHTS_INT8 ||
           type == SegmentType::WEIGHTS_INT4 ||
           type == SegmentType::ATTENTION_WEIGHTS ||
           type == SegmentType::FEED_FORWARD_WEIGHTS ||
           type == SegmentType::EMBEDDING_WEIGHTS ||
           type == SegmentType::LAYER_NORM_WEIGHTS;
}

static bool isExecutableOpLayerType(const std::string& layer_type) {
    if (layer_type.empty()) {
        return false;
    }
    std::string lower = layer_type;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lower == "add" ||
           lower == "sub" ||
           lower == "mul" ||
           lower == "div" ||
           lower == "relu" ||
           lower == "sigmoid" ||
           lower == "tanh" ||
           lower == "softmax" ||
           lower == "gelu" ||
           lower == "leakyrelu" ||
           lower == "elu" ||
           lower == "silu" ||
           lower == "swish" ||
           lower == "reshape" ||
           lower == "transpose" ||
           lower == "flatten" ||
           lower == "concat" ||
           lower == "slice" ||
           lower == "gather" ||
           lower == "matmul" ||
           lower == "gemm" ||
           lower == "conv" ||
           lower == "convtranspose" ||
           lower == "batchnormalization" ||
           lower == "layernormalization" ||
           lower == "maxpool" ||
           lower == "averagepool" ||
           lower == "avgpool" ||
           lower == "globalaveragepool" ||
           lower == "attention";
}

static bool isBiasLikeSegmentName(const std::string& segment_name) {
    return segment_name.find(".bias") != std::string::npos ||
           segment_name.find("/bias") != std::string::npos ||
           segment_name.find("_bias") != std::string::npos;
}

enum class TensorDecodeKind {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4
};

static std::string toLowerCopy(std::string value);

static bool startsWith(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

static inline float bf16_to_fp32(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    return *reinterpret_cast<const float*>(&bits);
}

static bool isGGUFQuantizedFormat(const std::string& format_lower) {
    return startsWith(format_lower, "q") || startsWith(format_lower, "iq") || startsWith(format_lower, "tq");
}

static size_t ggufBlockBytesForFormat(const std::string& format_lower) {
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

static size_t ggufBlockElementsForFormat(const std::string& format_lower) {
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

static size_t ggufQuantBitsForFormat(const std::string& format_lower) {
    if (format_lower == "q2_k" || format_lower == "iq2_xxs" || format_lower == "iq2_xs" || format_lower == "iq2_s" ||
        format_lower == "tq2_0") {
        return 2;
    }
    if (format_lower == "q3_k" || format_lower == "iq3_xxs" || format_lower == "iq3_s") {
        return 3;
    }
    if (format_lower == "q4_k" || format_lower == "q4_0_4_4" || format_lower == "q4_0_4_8" || format_lower == "q4_0_8_8" ||
        format_lower == "iq4_nl" || format_lower == "iq4_xs") {
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

static uint32_t readPackedBitsLSB(const uint8_t* bytes, size_t bit_offset, size_t bit_count) {
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

static bool decodeSupportedGGUFQuantized(
    const std::vector<std::byte>& source_data,
    const std::string& format_lower,
    size_t num_elements,
    std::vector<float>& output
) {
    if (num_elements == 0 || source_data.empty()) {
        output.clear();
        return true;
    }

    const size_t block_bytes = ggufBlockBytesForFormat(format_lower);
    if (block_bytes == 0 || source_data.size() < block_bytes) {
        return false;
    }

    const size_t block_elems = ggufBlockElementsForFormat(format_lower);
    if (block_elems == 0) {
        return false;
    }

    const size_t available_blocks = source_data.size() / block_bytes;
    const size_t available_elements = available_blocks * block_elems;
    if (available_elements == 0) {
        output.clear();
        return true;
    }
    const size_t decode_count = std::min(num_elements, available_elements);
    output.assign(decode_count, 0.0f);

    size_t out_index = 0;
    for (size_t block_index = 0; block_index < available_blocks && out_index < decode_count; ++block_index) {
        const uint8_t* block = reinterpret_cast<const uint8_t*>(source_data.data()) + (block_index * block_bytes);

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
            // q6_k block layout (210 bytes): ql[128], qh[64], scales[16], d(fp16)
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
            const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block + 208));
            for (size_t i = 0; i < 256 && out_index < decode_count; ++i) {
                const uint8_t low4 = (i & 1U) == 0U ? (ql[i / 2] & 0x0F) : (ql[i / 2] >> 4);
                const uint8_t high2 = static_cast<uint8_t>((qh[i / 4] >> (2 * (i % 4))) & 0x03U);
                const int q = static_cast<int>(low4 | (high2 << 4)) - 32;
                const int scale = static_cast<int>(scales[i / 16]);
                output[out_index++] = static_cast<float>(q * scale) * d;
            }
            continue;
        }

        if (format_lower == "q8_k") {
            // q8_k block layout (292 bytes): d(fp16), dmin(fp16), qs[256], bsums[16]int16
            // For inference decoding we decode the primary qs stream scaled by d.
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

        // Generic fallback for remaining GGUF block formats where exact
        // sub-block scale/min tables differ per variant.
        const size_t bits = ggufQuantBitsForFormat(format_lower);
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
            const uint32_t q = readPackedBitsLSB(packed, i * bits, bits);
            const int32_t signed_q = static_cast<int32_t>(q) - centered_offset;
            output[out_index++] = static_cast<float>(signed_q) * d;
        }
        continue;
    }

    return out_index == decode_count;
}

static TensorDecodeKind detectTensorDecodeKind(const ModelSegment& model_segment) {
    const std::string format = toLowerCopy(model_segment.data_format);
    if (format == "f16" || format == "fp16") {
        return TensorDecodeKind::FP16;
    }
    if (format == "bf16") {
        return TensorDecodeKind::BF16;
    }
    if (isGGUFQuantizedFormat(format)) {
        return TensorDecodeKind::FP32; // Decode via GGUF block path.
    }
    if (format == "i8" || format == "int8") {
        return TensorDecodeKind::INT8;
    }
    if (startsWith(format, "int4")) {
        return TensorDecodeKind::INT4;
    }

    if (model_segment.type == SegmentType::WEIGHTS_FP16) {
        return TensorDecodeKind::FP16;
    }
    if (model_segment.type == SegmentType::WEIGHTS_INT8) {
        return TensorDecodeKind::INT8;
    }
    if (model_segment.type == SegmentType::WEIGHTS_INT4) {
        return TensorDecodeKind::INT4;
    }

    return TensorDecodeKind::FP32;
}

static std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static size_t productFromShape(const std::vector<size_t>& shape, bool drop_batch_dim_if_present) {
    if (shape.empty()) {
        return 0;
    }
    size_t start = 0;
    if (drop_batch_dim_if_present && shape.size() >= 2) {
        start = 1;
    }

    size_t product = 1;
    for (size_t index = start; index < shape.size(); ++index) {
        const size_t dim = shape[index];
        if (dim == 0 || product > (std::numeric_limits<size_t>::max() / dim)) {
            return 0;
        }
        product *= dim;
    }
    return product;
}

static std::string deriveExecutionName(const CompressedSegmentHeader& seg) {
    if (!seg.layer_name.empty()) {
        return seg.layer_name;
    }

    std::string name = seg.name;
    static const std::array<const char*, 8> suffixes = {
        ".weight", ".weights", ".bias", "/weight", "/weights", "/bias", "_weight", "_bias"
    };
    for (const char* suffix : suffixes) {
        const std::string suffix_str(suffix);
        if (name.size() > suffix_str.size() &&
            name.compare(name.size() - suffix_str.size(), suffix_str.size(), suffix_str) == 0) {
            name.erase(name.size() - suffix_str.size());
            break;
        }
    }
    return name;
}

static bool supportsSparseStreamingCompute(const std::vector<std::byte>& compressed_bytes) {
    if (compressed_bytes.empty()) {
        return false;
    }
    const uint8_t flag = static_cast<uint8_t>(compressed_bytes[0]);
    // SDRIndexStorageStrategy::forEachIndexValue currently supports these formats.
    return flag == 0x95 || flag == 0x96;
}

/**
 * @brief Decode varint-encoded indices from SDR compressed data.
 * @param data Raw compressed byte buffer containing varint-encoded indices.
 * @return Vector of decoded sparse indices (0-based positions).
 *
 * @details Decodes a sequence of unsigned varints packed consecutively.
 * Each varint uses the MSB as a continuation bit. This routine is used to
 * rebuild sparse position lists for compressed tensors.
 *
 * @complexity O(N) where N is the number of decoded indices.
 */
std::vector<size_t> SDRModelLoader::decode_varint_indices(const std::vector<std::byte>& data) {
    std::vector<size_t> indices;
    size_t pos = 0;
    while (pos < data.size()) {
        uint32_t index = 0;
        uint32_t shift = 0;
        while (pos < data.size()) {
            uint8_t byte = static_cast<uint8_t>(data[pos++]);
            index |= (byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) break;
            shift += 7;
        }
        indices.push_back(index);
    }
    return indices;
}

/**
 * @brief Parse layer metadata from string format into LayerInfo structure.
 * @param metadata Space-delimited key/value pairs (e.g., "type conv strides 1,1").
 * @param layer LayerInfo to populate with parsed attributes.
 *
 * @details Recognized keys: type, kernel_shape, strides, padding, activation,
 * dropout, batch_norm. Unknown keys are ignored. Numeric lists are comma-separated.
 *
 * @note This parser is tolerant and best-effort; validation occurs later during
 * execution. Invalid numeric conversions may throw std::invalid_argument.
 */
static void parseLayerMetadataPayload(const std::string& metadata, LayerInfo& layer) {
    std::istringstream iss(metadata);
    std::string key, value;
    
    while (iss >> key >> value) {
        if (key == "type") {
            std::string lower_value = toLowerCopy(value);
            if (lower_value == "conv")
                layer.layer_type = "CONV2D";
            else if (lower_value == "linear" || lower_value == "gemm" || lower_value == "matmul" || lower_value == "dense")
                layer.layer_type = "LINEAR";
            else if (lower_value == "attention")
                layer.layer_type = "ATTENTION";
            else if (lower_value == "norm" || lower_value == "layer_norm")
                layer.layer_type = "NORM";
            else if (lower_value == "batchnormalization")
                layer.layer_type = "BatchNormalization";
            else if (lower_value == "layernormalization")
                layer.layer_type = "LayerNormalization";
            else if (lower_value == "pool")
                layer.layer_type = "POOLING";
            else if (lower_value == "activation")
                layer.layer_type = "ACTIVATION";
            else if (lower_value == "add")
                layer.layer_type = "Add";
            else if (lower_value == "relu")
                layer.layer_type = "Relu";
            else if (lower_value == "sigmoid")
                layer.layer_type = "Sigmoid";
            else if (lower_value == "tanh")
                layer.layer_type = "Tanh";
            else if (lower_value == "softmax")
                layer.layer_type = "Softmax";
            else if (lower_value == "gelu")
                layer.layer_type = "Gelu";
            else if (lower_value == "leakyrelu" || lower_value == "leaky_relu")
                layer.layer_type = "LeakyRelu";
            else if (lower_value == "elu")
                layer.layer_type = "Elu";
            else if (lower_value == "silu" || lower_value == "swish")
                layer.layer_type = "Silu";
            else if (lower_value == "reshape")
                layer.layer_type = "Reshape";
            else if (lower_value == "transpose")
                layer.layer_type = "Transpose";
            else if (lower_value == "flatten")
                layer.layer_type = "Flatten";
            else if (lower_value == "concat")
                layer.layer_type = "Concat";
            else if (lower_value == "slice")
                layer.layer_type = "Slice";
            else if (lower_value == "gather")
                layer.layer_type = "Gather";
            else if (lower_value == "sub")
                layer.layer_type = "Sub";
            else if (lower_value == "mul")
                layer.layer_type = "Mul";
            else if (lower_value == "div")
                layer.layer_type = "Div";
            else if (lower_value == "maxpool")
                layer.layer_type = "MaxPool";
            else if (lower_value == "averagepool" || lower_value == "avgpool")
                layer.layer_type = "AveragePool";
            else if (lower_value == "globalaveragepool")
                layer.layer_type = "GlobalAveragePool";
            else
                layer.layer_type = value;
        } else if (key == "kernel_shape") {
            std::istringstream value_stream(value);
            std::string dim;
            while (std::getline(value_stream, dim, ',')) {
                layer.properties.kernel_shape.push_back(std::stoull(dim));
            }
        } else if (key == "strides") {
            std::istringstream value_stream(value);
            std::string dim;
            while (std::getline(value_stream, dim, ',')) {
                layer.properties.strides.push_back(std::stoull(dim));
            }
        } else if (key == "padding") {
            std::istringstream value_stream(value);
            std::string dim;
            while (std::getline(value_stream, dim, ',')) {
                layer.properties.padding.push_back(std::stoull(dim));
            }
        } else if (key == "activation") {
            layer.properties.activation_type = value;
        } else if (key == "dropout") {
            layer.properties.dropout_rate = std::stof(value);
        } else if (key == "batch_norm") {
            layer.properties.use_batch_norm = (value == "true");
        } else if (key == "op_type") {
            if (layer.layer_type.empty()) {
                layer.layer_type = value;
            }
            const std::string lower_value = toLowerCopy(value);
            if (lower_value == "matmul") {
                layer.layer_type = "MatMul";
            } else if (lower_value == "gemm") {
                layer.layer_type = "Gemm";
            } else if (lower_value == "conv") {
                layer.layer_type = "Conv";
            } else if (lower_value == "convtranspose") {
                layer.layer_type = "ConvTranspose";
            } else if (lower_value == "batchnormalization") {
                layer.layer_type = "BatchNormalization";
            } else if (lower_value == "layernormalization") {
                layer.layer_type = "LayerNormalization";
            } else if (lower_value == "globalaveragepool") {
                layer.layer_type = "GlobalAveragePool";
            }
        }
    }
}

void SDRModelLoader::parseLayerMetadata(const std::string& metadata, LayerInfo& layer) {
    parseLayerMetadataPayload(metadata, layer);
}

/**
 * @brief Construct an SDRModelLoader with on-demand loading support.
 * @param archive_path Path to the compressed model archive (.sdr file).
 *
 * @details Registers decompression strategies and indexes the archive for
 * on-demand access. Also installs compatibility strategies for legacy assets.
 *
 * @throws std::runtime_error If the archive cannot be opened or parsed.
 */
SDRModelLoader::SDRModelLoader(const std::string& archive_path) : archive_path_(archive_path) {
    // Initialize decompressor with registered compression strategies
    decompressor_ = std::make_unique<AIDecompressor>();
    
    // Strategy IDs must match those used during compression
    const uint8_t SDR_STRATEGY_ID = 1;
    const uint8_t RLE_STRATEGY_ID = 2;
    const uint8_t GZIP_STRATEGY_ID = 3;
    const uint8_t QUANT_STRATEGY_ID = 4;
    
    // Register compression strategies
    auto adaptiveStrategy = std::make_shared<AdaptiveSDRStrategy>(0.02f);
    decompressor_->registerStrategy(SDR_STRATEGY_ID, adaptiveStrategy);
    decompressor_->registerStrategy(RLE_STRATEGY_ID, std::make_shared<NumericalRLEStrategy>());
    decompressor_->registerStrategy(GZIP_STRATEGY_ID, std::make_shared<GzipStrategy>());
#ifdef ENABLE_QUANTIZATION
    decompressor_->registerStrategy(QUANT_STRATEGY_ID, std::make_shared<QuantizedTensorStrategy>());
#endif

    // Legacy compatibility support
    auto legacySdrStrategy = std::make_shared<SDRIndexStorageStrategy>();
    decompressor_->registerStrategy(SDR_STRATEGY_ID + 10, legacySdrStrategy);

    loadFromArchive(archive_path);
}

/**
 * @brief Populate a LayerInfo from a decompressed ModelSegment.
 * @param model_segment Source model segment containing tensor bytes and meta.
 * @param layer Target LayerInfo to populate in-place.
 *
 * @details Determines whether the segment contains weights or biases and
 * performs the appropriate copy, handling element size by segment type.
 * Tensor shape metadata is propagated when available.
 */
static void fillLayerInfoFromSegment(const ModelSegment& model_segment, LayerInfo& layer) {
    // Set layer type from segment metadata
    if (layer.layer_type.empty()) {
        layer.layer_type = model_segment.layer_type;
    }

    // Extract tensor metadata if available (but don't use tensor dimensions as layer shapes)
    // The actual layer input/output shapes come from model_segment.input_shape/output_shape
    // which are set at the end of this function

    if (layer.raw_data.empty()) {
        layer.raw_data = model_segment.data;
    }
    const auto& source_data = model_segment.data;
    const std::string lower_layer_type = toLowerCopy(layer.layer_type);
    const bool norm_like = lower_layer_type.find("norm") != std::string::npos;
    const bool weight_like_segment = isWeightLikeSegmentType(model_segment.type);

    if (!weight_like_segment) {
        if (!source_data.empty() &&
            (model_segment.type == SegmentType::METADATA_JSON ||
             model_segment.type == SegmentType::CONFIG ||
             model_segment.type == SegmentType::METADATA_TOML)) {
            const std::string metadata_text(
                reinterpret_cast<const char*>(source_data.data()),
                source_data.size()
            );
            parseLayerMetadataPayload(metadata_text, layer);
        }
        return;
    }

    const TensorDecodeKind decode_kind = detectTensorDecodeKind(model_segment);
    const std::string format_lower = toLowerCopy(model_segment.data_format);
    const bool is_gguf_quantized = isGGUFQuantizedFormat(format_lower);
    const bool supports_direct_gguf_decode = ggufBlockBytesForFormat(format_lower) > 0;

    size_t num_elements = 0;
    if (model_segment.tensor_metadata && !model_segment.tensor_metadata->dimensions.empty()) {
        num_elements = 1;
        for (size_t dim : model_segment.tensor_metadata->dimensions) {
            if (dim == 0 || num_elements > (std::numeric_limits<size_t>::max() / dim)) {
                num_elements = 0;
                break;
            }
            num_elements *= dim;
        }
    }
    if (num_elements == 0) {
        size_t element_size = sizeof(float);
        if (decode_kind == TensorDecodeKind::FP16) {
            element_size = sizeof(uint16_t);
        } else if (decode_kind == TensorDecodeKind::INT8) {
            element_size = sizeof(int8_t);
        } else if (decode_kind == TensorDecodeKind::INT4) {
            element_size = 0; // Packed 4-bit path handled separately below.
        }
        if (element_size > 0 && model_segment.original_size > 0) {
            num_elements = model_segment.original_size / element_size;
        } else if (decode_kind == TensorDecodeKind::INT4) {
            num_elements = model_segment.original_size * 2;
        }
        if (num_elements == 0 && supports_direct_gguf_decode) {
            const size_t block_bytes = ggufBlockBytesForFormat(format_lower);
            if (block_bytes > 0) {
                num_elements = (source_data.size() / block_bytes) * 32;
            }
        }
    }

    // Distinguish between weights and biases by name convention
    if (isBiasLikeSegmentName(model_segment.name)) {
        if (!source_data.empty()) {
            layer.biases.resize(num_elements);
             if (is_gguf_quantized && supports_direct_gguf_decode) {
                std::vector<float> decoded;
                if (decodeSupportedGGUFQuantized(source_data, format_lower, num_elements, decoded)) {
                    layer.biases = std::move(decoded);
                } else {
                    std::cerr << "[SDRInferenceEngine] WARNING: Failed decoding GGUF quantized biases format '"
                              << model_segment.data_format << "' for segment " << model_segment.name << std::endl;
                    layer.biases.clear();
                }
            } else if (is_gguf_quantized) {
                std::cerr << "[SDRInferenceEngine] WARNING: Unsupported GGUF quantized format '" << model_segment.data_format
                          << "' for layer segment " << model_segment.name
                          << ". Clearing decoded bias tensor; layer may fall back to pass-through." << std::endl;
                layer.biases.clear();
            } else if (decode_kind == TensorDecodeKind::FP16) {
                // Convert FP16 to FP32
                const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(source_data.data());
                const size_t max_elements = std::min(num_elements, source_data.size() / sizeof(uint16_t));
                for (size_t i = 0; i < max_elements; ++i) {
                    layer.biases[i] = fp16_to_fp32(fp16_data[i]);
                }
            } else if (decode_kind == TensorDecodeKind::BF16) {
                const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(source_data.data());
                const size_t max_elements = std::min(num_elements, source_data.size() / sizeof(uint16_t));
                for (size_t i = 0; i < max_elements; ++i) {
                    layer.biases[i] = bf16_to_fp32(bf16_data[i]);
                }
            } else if (decode_kind == TensorDecodeKind::INT8) {
                const int8_t* int8_data = reinterpret_cast<const int8_t*>(source_data.data());
                const size_t max_elements = std::min(num_elements, source_data.size());
                for (size_t i = 0; i < max_elements; ++i) {
                    layer.biases[i] = static_cast<float>(int8_data[i]);
                }
            } else if (decode_kind == TensorDecodeKind::INT4) {
                size_t out_index = 0;
                for (size_t i = 0; i < source_data.size() && out_index < num_elements; ++i) {
                    const uint8_t packed = static_cast<uint8_t>(source_data[i]);
                    const int8_t low = static_cast<int8_t>((packed & 0x0F) - 8);
                    layer.biases[out_index++] = static_cast<float>(low);
                    if (out_index < num_elements) {
                        const int8_t high = static_cast<int8_t>(((packed >> 4) & 0x0F) - 8);
                        layer.biases[out_index++] = static_cast<float>(high);
                    }
                }
            } else {
                const size_t byte_count = std::min(source_data.size(), layer.biases.size() * sizeof(float));
                std::memcpy(layer.biases.data(), source_data.data(), byte_count);
            }
            if (norm_like) {
                layer.properties.bn_biases = layer.biases;
            }
        }
    } else {
        if (!source_data.empty()) {
            layer.weights.resize(num_elements);
            if (is_gguf_quantized && supports_direct_gguf_decode) {
                std::vector<float> decoded;
                if (decodeSupportedGGUFQuantized(source_data, format_lower, num_elements, decoded)) {
                    layer.weights = std::move(decoded);
                } else {
                    std::cerr << "[SDRInferenceEngine] WARNING: Failed decoding GGUF quantized weights format '"
                              << model_segment.data_format << "' for segment " << model_segment.name << std::endl;
                    layer.weights.clear();
                }
            } else if (is_gguf_quantized) {
                std::cerr << "[SDRInferenceEngine] WARNING: Unsupported GGUF quantized format '" << model_segment.data_format
                          << "' for layer segment " << model_segment.name
                          << ". Clearing decoded weight tensor; layer may fall back to pass-through." << std::endl;
                layer.weights.clear();
            } else if (decode_kind == TensorDecodeKind::FP16) {
                // Convert FP16 to FP32
                const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(source_data.data());
                const size_t max_elements = std::min(num_elements, source_data.size() / sizeof(uint16_t));
                for (size_t i = 0; i < max_elements; ++i) {
                    layer.weights[i] = fp16_to_fp32(fp16_data[i]);
                }
            } else if (decode_kind == TensorDecodeKind::BF16) {
                const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(source_data.data());
                const size_t max_elements = std::min(num_elements, source_data.size() / sizeof(uint16_t));
                for (size_t i = 0; i < max_elements; ++i) {
                    layer.weights[i] = bf16_to_fp32(bf16_data[i]);
                }
            } else if (decode_kind == TensorDecodeKind::INT8) {
                const int8_t* int8_data = reinterpret_cast<const int8_t*>(source_data.data());
                const size_t max_elements = std::min(num_elements, source_data.size());
                for (size_t i = 0; i < max_elements; ++i) {
                    layer.weights[i] = static_cast<float>(int8_data[i]);
                }
            } else if (decode_kind == TensorDecodeKind::INT4) {
                size_t out_index = 0;
                for (size_t i = 0; i < source_data.size() && out_index < num_elements; ++i) {
                    const uint8_t packed = static_cast<uint8_t>(source_data[i]);
                    const int8_t low = static_cast<int8_t>((packed & 0x0F) - 8);
                    layer.weights[out_index++] = static_cast<float>(low);
                    if (out_index < num_elements) {
                        const int8_t high = static_cast<int8_t>(((packed >> 4) & 0x0F) - 8);
                        layer.weights[out_index++] = static_cast<float>(high);
                    }
                }
            } else {
                const size_t byte_count = std::min(source_data.size(), layer.weights.size() * sizeof(float));
                std::memcpy(layer.weights.data(), source_data.data(), byte_count);
            }
            if (norm_like) {
                layer.properties.bn_weights = layer.weights;
            }
        }
    }

    // Set input/output shapes from segment if available
    if (!model_segment.input_shape.empty()) {
        layer.input_shape = model_segment.input_shape;
    }
    if (!model_segment.output_shape.empty()) {
        layer.output_shape = model_segment.output_shape;
    }
}

/**
 * @brief Load model archive headers for on-demand access.
 * @param archive_path Path to the compressed model archive.
 *
 * @details Reads only archive headers, not payloads, to build a segment index
 * for subsequent on-demand segment decompression.
 *
 * @throws std::runtime_error Propagated if header parsing fails.
 */
void SDRModelLoader::loadFromArchive(const std::string& archive_path) {
    try {
        std::ifstream infile(archive_path, std::ios::binary);
        if (!infile) {
            throw std::runtime_error("[SDRModelLoader] Failed to open archive: " + archive_path);
        }

        // Read only headers for on-demand loading efficiency
        segments_ = decompressor_->readArchiveHeaders(infile);
#ifdef ENABLE_ONNX_PROTOBUF
        loaded_model_proto_.reset();
        const auto graph_it = std::find_if(segments_.begin(), segments_.end(), [](const CompressedSegmentHeader& seg) {
            return seg.original_type == SegmentType::GRAPH_STRUCTURE_PROTO;
        });
        if (graph_it != segments_.end()) {
            try {
                ModelSegment graph_segment = decompressor_->decompressSegment(
                    archive_path, *graph_it, graph_it->data_offset
                );
                if (!graph_segment.data.empty()) {
                    onnx::GraphProto graph_proto;
                    if (graph_proto.ParseFromArray(graph_segment.data.data(), static_cast<int>(graph_segment.data.size()))) {
                        onnx::ModelProto model_proto;
                        model_proto.mutable_graph()->CopyFrom(graph_proto);
                        loaded_model_proto_ = std::move(model_proto);
                    } else {
                        std::cerr << "[SDRModelLoader] WARNING: Failed to parse GRAPH_STRUCTURE_PROTO payload as ONNX GraphProto" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[SDRModelLoader] WARNING: Failed to load ONNX graph structure from archive: " << e.what() << std::endl;
            }
        }
#endif

    } catch (const std::exception& e) {
        std::cerr << "Error loading compressed model headers: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief Load a single layer by name synchronously.
 * @param name Layer name to load.
 * @return LayerInfo containing the loaded layer data.
 */
LayerInfo SDRModelLoader::loadLayerByName(const std::string& name) const {
    return loadLayerByNameAsync(name).get();
}

/**
 * @brief Load a single layer by name asynchronously.
 * @param name Layer name to load.
 * @return Future containing the LayerInfo when loading completes.
 *
 * @details Implements on-demand loading with an async cache. If a load is
 * already in progress or completed, returns the cached future.
 */
std::shared_future<LayerInfo> SDRModelLoader::loadLayerByNameAsync(const std::string& name) const {
    {
        std::lock_guard<std::mutex> lock(layer_cache_mutex_);
        auto it = layer_cache_.find(name);
        if (it != layer_cache_.end()) {
            return it->second;
        }
    }

    auto future = std::async(std::launch::async, [this, name]() {
        std::vector<const CompressedSegmentHeader*> matched_segments;
        matched_segments.reserve(8);

        for (const auto& seg : segments_) {
            if ((!seg.layer_name.empty() && seg.layer_name == name) ||
                (seg.layer_name.empty() && seg.name == name)) {
                matched_segments.push_back(&seg);
            }
        }

        if (matched_segments.empty()) {
            auto seg_it = std::find_if(segments_.begin(), segments_.end(),
                [&](const CompressedSegmentHeader& seg) { return seg.name == name; });
            if (seg_it != segments_.end()) {
                matched_segments.push_back(&(*seg_it));
            }
        }

        if (matched_segments.empty()) {
            throw std::runtime_error("Segment info not found for layer: " + name);
        }

        LayerInfo layer;
        layer.name = name;
        bool have_primary_compressed_weights = false;
        constexpr uint8_t SDR_STRATEGY_ID = 1;

        for (const auto* seg_ptr : matched_segments) {
            const auto& seg_info = *seg_ptr;
            const bool weight_like =
                seg_info.original_type == SegmentType::WEIGHTS_FP32 ||
                seg_info.original_type == SegmentType::WEIGHTS_FP16 ||
                seg_info.original_type == SegmentType::WEIGHTS_INT8 ||
                seg_info.original_type == SegmentType::WEIGHTS_INT4 ||
                seg_info.original_type == SegmentType::ATTENTION_WEIGHTS ||
                seg_info.original_type == SegmentType::FEED_FORWARD_WEIGHTS ||
                seg_info.original_type == SegmentType::EMBEDDING_WEIGHTS ||
                seg_info.original_type == SegmentType::LAYER_NORM_WEIGHTS;
            const bool bias_segment = isBiasLikeSegmentName(seg_info.name);

            if (layer.layer_type.empty() && !seg_info.layer_type.empty()) {
                layer.layer_type = seg_info.layer_type;
            }
            if (seg_info.tensor_metadata) {
                const auto& meta = seg_info.tensor_metadata.value();
                if (layer.input_shape.empty() && !meta.dimensions.empty()) {
                    layer.input_shape = meta.dimensions;
                }
                if (layer.output_shape.empty() && !meta.dimensions.empty()) {
                    layer.output_shape = meta.dimensions;
                }
            }
            if (!seg_info.input_shape.empty()) {
                layer.input_shape = seg_info.input_shape;
            }
            if (!seg_info.output_shape.empty()) {
                layer.output_shape = seg_info.output_shape;
            }

            if (weight_like &&
                !bias_segment &&
                !have_primary_compressed_weights &&
                seg_info.compression_strategy_id == SDR_STRATEGY_ID) {
                std::vector<std::byte> compressed_bytes =
                    decompressor_->readCompressedBytes(archive_path_, seg_info, seg_info.data_offset);
                if (supportsSparseStreamingCompute(compressed_bytes)) {
                    layer.raw_data = std::move(compressed_bytes);
                    have_primary_compressed_weights = true;
                    continue;
                }
            }

            ModelSegment model_segment = decompressor_->decompressSegment(
                archive_path_, seg_info, seg_info.data_offset
            );
            fillLayerInfoFromSegment(model_segment, layer);
        }
        
        return layer;
    }).share();

    {
        std::lock_guard<std::mutex> lock(layer_cache_mutex_);
        auto [it, inserted] = layer_cache_.emplace(name, future);
        if (!inserted) {
            return it->second;
        }
    }

    return future;
}

/**
 * @brief Clear a layer from cache to free memory.
 * @param name Layer name to remove from cache.
 */
void SDRModelLoader::clearLayerFromCache(const std::string& name) const {
    std::lock_guard<std::mutex> lock(layer_cache_mutex_);
    auto it = layer_cache_.find(name);
    if (it != layer_cache_.end()) {
        layer_cache_.erase(it);
    }
}

/**
 * @brief Deprecated SDR decompression method.
 * @deprecated Use AIDecompressor framework instead.
 * @throws std::logic_error Always thrown.
 */
std::vector<std::byte> SDRModelLoader::decompressSDR(const std::vector<std::byte>& compressed_data, size_t original_size) const {
    throw std::logic_error("SDRModelLoader::decompressSDR is deprecated.");
}

/**
 * @brief Get all loaded layers (legacy method).
 * @return Vector of LayerInfo structures for pre-loaded layers.
 * @deprecated Use on-demand loading methods instead.
 */
const std::vector<LayerInfo>& SDRModelLoader::getLayers() const {
    return layers;
}

/**
 * @brief Get segment index for on-demand loading.
 * @return Vector of compressed segment headers.
 */
const std::vector<CompressedSegmentHeader>& SDRModelLoader::getSegmentIndex() const {
    return segments_;
}

/**
 * @brief Construct SDRInferenceEngine with on-demand model access.
 * @param model_loader Reference to SDRModelLoader for layer access.
 *
 * @details Initializes the dispatch table for common ops and the default
 * fallback handler for unknown layer types.
 */
SDRInferenceEngine::SDRInferenceEngine(SDRModelLoader& model_loader)
    : loader_(model_loader), batch_size(1), dropout_enabled(false), training_mode(false),
      memory_pool_offset_(0), max_memory_usage_(8ULL * 1024 * 1024 * 1024), // 8GB default
      aggressive_memory_management_(true), current_buffer_idx_(0), enable_prefetch_(false) {
    
    // Register operation handlers for common layer types
    op_dispatch_["Conv"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyConvolutionalLayer(l, in); 
    };
    op_dispatch_["CONV2D"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return applyConvolutionalLayer(l, in);
    };
    op_dispatch_["ConvTranspose"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return applyConvolutionalLayer(l, in);
    };
    op_dispatch_["MatMul"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyLinearLayer(l, in); 
    };
    op_dispatch_["Gemm"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyLinearLayer(l, in); 
    };
    op_dispatch_["LINEAR"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyLinearLayer(l, in); 
    };
    op_dispatch_["ATTENTION"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeAttentionOperation(l, in);
    };
    op_dispatch_["FEED_FORWARD"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return applyLinearLayer(l, in);
    };
    op_dispatch_["NORM"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeNormalizationOperation(l, in);
    };
    op_dispatch_["EMBEDDING"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return applyLinearLayer(l, in);
    };
    op_dispatch_["TOKEN_EMBEDDING"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return applyLinearLayer(l, in);
    };
    op_dispatch_["BATCH_NORM"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyBatchNorm(l, in); 
    };
    op_dispatch_["BatchNormalization"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeNormalizationOperation(l, in);
    };
    op_dispatch_["LayerNormalization"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeNormalizationOperation(l, in);
    };
    op_dispatch_["ACTIVATION"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyActivation(l.properties.activation_type, in); 
    };
    op_dispatch_["Relu"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["Sigmoid"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["Tanh"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["Softmax"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["Gelu"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["LeakyRelu"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["Elu"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["Silu"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["Swish"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeActivationOperation(l, in);
    };
    op_dispatch_["MaxPool"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executePoolingOperation(l, in);
    };
    op_dispatch_["AveragePool"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executePoolingOperation(l, in);
    };
    op_dispatch_["GlobalAveragePool"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executePoolingOperation(l, in);
    };
    op_dispatch_["Add"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Sub"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Mul"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Div"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Concat"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Reshape"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Transpose"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Flatten"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Slice"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    op_dispatch_["Gather"] = [this](const LayerInfo& l, const std::vector<float>& in) {
        return executeElementwiseOperation(l, in);
    };
    
    // Default handler for unknown layer types
    default_handler_ = [this](const LayerInfo& layer, const std::vector<float>& input) {
        return executeDynamicLayer(layer, input);
    };
    
    // Log performance optimizations
    std::cout << "[SDRInferenceEngine] Performance optimizations:" << std::endl;
    std::cout << "  - BLAS: " << CortexAICompression::Kernels::get_blas_implementation() << std::endl;
    std::cout << "  - SIMD: " << CortexAICompression::Kernels::get_simd_level() << std::endl;
}

/**
 * @brief Set batch size for inference.
 * @param size Number of samples to process simultaneously.
 */
void SDRInferenceEngine::setBatchSize(size_t size) {
    if (size == 0) {
        std::cerr << "[SDRInferenceEngine] WARNING: batch size cannot be 0, using 1" << std::endl;
        batch_size = 1;
        return;
    }
    batch_size = size;
}

/**
 * @brief Enable or disable dropout during inference.
 * @param enable True to enable, false to disable.
 */
void SDRInferenceEngine::enableDropout(bool enable) {
    dropout_enabled = enable;
}

/**
 * @brief Set training/inference mode.
 * @param training True for training mode, false for inference mode.
 */
void SDRInferenceEngine::setInferenceMode(bool training) {
    training_mode = training;
}

void SDRInferenceEngine::setForceCompressedCompute(bool enable) {
    force_compressed_compute_ = enable;
}

/**
 * @brief Apply linear (fully connected) layer operation with BLAS acceleration
 * @param layer LayerInfo containing weights, biases, and shape information
 * @param input Input tensor data
 * @return Output tensor after linear transformation
 */
std::vector<float> SDRInferenceEngine::applyLinearLayer(const LayerInfo& layer, const std::vector<float>& input) {
    // Debug: print shapes for first few layers BEFORE calculating
    static int debug_count = 0;
    if (debug_count < 5) {
        std::cerr << "[DEBUG] Layer: " << layer.name << std::endl;
        std::cerr << "  input_shape: [";
        for (size_t i = 0; i < layer.input_shape.size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << layer.input_shape[i];
        }
        std::cerr << "]" << std::endl;
        std::cerr << "  output_shape: [";
        for (size_t i = 0; i < layer.output_shape.size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << layer.output_shape[i];
        }
        std::cerr << "]" << std::endl;
        std::cerr << "  biases.size()=" << layer.biases.size() << ", weights.size()=" << layer.weights.size() << std::endl;
        debug_count++;
    }
    
    size_t input_size = productFromShape(layer.input_shape, true);
    size_t output_size = productFromShape(layer.output_shape, true);
    
    if (debug_count <= 5) {
        std::cerr << "  Calculated: input_size=" << input_size << ", output_size=" << output_size << std::endl;
    }

    auto infer_from_weights = [&]() -> bool {
        // If we have valid shapes from metadata, trust them (especially for compressed models)
        if (input_size > 0 && output_size > 0) {
            return true;
        }
        
        if (layer.weights.empty()) {
            return false;
        }

        if (!layer.biases.empty() && layer.weights.size() % layer.biases.size() == 0) {
            output_size = layer.biases.size();
            input_size = layer.weights.size() / output_size;
            return input_size > 0 && output_size > 0;
        }

        if (!input.empty() && layer.weights.size() % input.size() == 0) {
            input_size = input.size();
            output_size = layer.weights.size() / input_size;
            return input_size > 0 && output_size > 0;
        }

        if (input_size > 0 && layer.weights.size() % input_size == 0) {
            output_size = layer.weights.size() / input_size;
            return output_size > 0;
        }

        if (output_size > 0 && layer.weights.size() % output_size == 0) {
            input_size = layer.weights.size() / output_size;
            return input_size > 0;
        }

        return false;
    };

    if (input_size == 0 || output_size == 0) {
        if (!infer_from_weights()) {
            std::cerr << "[SDRInferenceEngine] ERROR: Missing/invalid linear dimensions for layer: " << layer.name << std::endl;
            return {};
        }
        if (debug_count <= 5) {
            std::cerr << "  After infer_from_weights: input_size=" << input_size << ", output_size=" << output_size << std::endl;
        }
    }

    if (input.size() % input_size != 0) {
        if (debug_count <= 5) {
            std::cerr << "  Input size mismatch: input.size()=" << input.size() << ", input_size=" << input_size << std::endl;
        }
        // For compressed models (empty weights), trust the stored dimensions
        if (layer.weights.empty() && !layer.raw_data.empty()) {
            std::cerr << "[SDRInferenceEngine] WARNING: Input size mismatch for compressed layer " << layer.name 
                      << ", but trusting stored dimensions (input_size=" << input_size << ", output_size=" << output_size << ")" << std::endl;
            // Don't recalculate - trust the metadata
        } else if (!input.empty() && !layer.weights.empty() && layer.weights.size() % input.size() == 0) {
            input_size = input.size();
            output_size = layer.weights.size() / input_size;
            if (debug_count <= 5) {
                std::cerr << "  Recalculated from input: input_size=" << input_size << ", output_size=" << output_size << std::endl;
            }
        } else {
            std::cerr << "[SDRInferenceEngine] ERROR: Input size " << input.size()
                      << " is not divisible by expected per-sample size " << input_size
                      << " for linear layer: " << layer.name << std::endl;
            return {};
        }
    }
    
    if (debug_count <= 5) {
        std::cerr << "  Before effective_batch calc: input_size=" << input_size << ", output_size=" << output_size << std::endl;
    }

    const size_t effective_batch = input.size() / input_size;

    try {
        if (!layer.weights.empty()) {
            Utils::TensorValidator::validate_linear_weights(
                layer.weights.size(), input_size, output_size, layer.name
            );
        }
    } catch (const Utils::TensorValidationError& e) {
        std::cerr << "[SDRInferenceEngine] Validation error: " << e.what() << std::endl;
        return {};
    }

    if (force_compressed_compute_ || layer.weights.size() != input_size * output_size) {
        // OPTIMIZED: Use sparse kernels for zero-decompression inference
        std::vector<float> output(effective_batch * output_size, 0.0f);
        try {
            CortexAICompression::SDRIndexStorageStrategy s;
            
            // Collect sparse indices and values for optimized computation
            std::vector<size_t> indices;
            std::vector<float> values;
            indices.reserve(input_size * output_size * 0.05f); // Reserve for ~5% sparsity
            values.reserve(input_size * output_size * 0.05f);
            
            s.forEachIndexValue(layer.raw_data, input_size * output_size, [&](size_t flatIndex, float w) {
                indices.push_back(flatIndex);
                values.push_back(w);
            });
            
            // Sparse kernel currently operates on one sample at a time.
            for (size_t b = 0; b < effective_batch; ++b) {
                const float* input_sample = input.data() + (b * input_size);
                float* output_sample = output.data() + (b * output_size);
                CortexAICompression::SparseKernels::sparse_linear_forward(
                    indices,
                    values,
                    input_sample,
                    layer.biases.empty() ? nullptr : layer.biases.data(),
                    output_sample,
                    input_size,
                    output_size
                );
            }
            
            last_layer_used_compressed_ = true;
            last_layer_retained_ratio_ = (input_size * output_size) ? 
                (static_cast<double>(indices.size()) / static_cast<double>(input_size * output_size)) : 0.0;
            
            return output;
        } catch (const std::exception& e) {
            std::cerr << "[SDRInferenceEngine] Compressed linear path failed: " << e.what() << std::endl;
            // Fall through to error path
        }
        std::cerr << "[SDRInferenceEngine] ERROR: Weights size " << layer.weights.size() << " does not match input*output " << input_size * output_size << " for linear layer: " << layer.name << std::endl;
        return {};
    }
    if (!layer.biases.empty() && layer.biases.size() != output_size) {
        if (debug_count <= 5) {
            std::cerr << "[DEBUG] Bias check: biases.size()=" << layer.biases.size() << ", output_size=" << output_size << ", input_size=" << input_size << std::endl;
        }
        std::cerr << "[SDRInferenceEngine] ERROR: Biases size " << layer.biases.size() << " does not match output " << output_size << " for linear layer: " << layer.name << std::endl;
        return {};
    }
    
    // Use ping-pong buffer to reduce allocations
    std::vector<float>& output = getNextPingPongBuffer(effective_batch * output_size);
    
    // Use BLAS-accelerated linear forward: output = input * weights^T + bias
    CortexAICompression::Kernels::linear_forward(
        input.data(),
        layer.weights.data(),
        layer.biases.empty() ? nullptr : layer.biases.data(),
        output.data(),
        effective_batch,
        input_size,
        output_size
    );
    
    return output;
}

/**
 * @brief Apply 2D convolutional layer operation using im2col + GEMM
 * @param layer LayerInfo containing convolution parameters and weights
 * @param input Input tensor in NCHW format
 * @return Output tensor after convolution operation
 */
std::vector<float> SDRInferenceEngine::applyConvolutionalLayer(const LayerInfo& layer, const std::vector<float>& input) {
    // Validate convolution parameters
    try {
        if (layer.input_shape.size() != 4 || layer.output_shape.size() != 4 ||
            layer.properties.kernel_shape.size() != 2 || layer.properties.strides.size() != 2 || 
            layer.properties.padding.size() != 2) {
            throw Utils::TensorValidationError("Invalid shape/params for convolutional layer: " + layer.name);
        }
        
        Utils::TensorValidator::validate_conv_params(
            layer.input_shape, 
            layer.properties.kernel_shape,
            layer.properties.strides,
            layer.properties.padding,
            layer.name
        );
    } catch (const Utils::TensorValidationError& e) {
        std::cerr << "[SDRInferenceEngine] Validation error: " << e.what() << std::endl;
        return {};
    }
    int batch = static_cast<int>(layer.input_shape[0]);
    int in_channels = layer.input_shape[1];
    int in_height = layer.input_shape[2];
    int in_width = layer.input_shape[3];
    int out_channels = layer.output_shape[1];
    int kernel_h = layer.properties.kernel_shape[0];
    int kernel_w = layer.properties.kernel_shape[1];
    int stride_h = layer.properties.strides[0];
    int stride_w = layer.properties.strides[1];
    int pad_h = layer.properties.padding[0];
    int pad_w = layer.properties.padding[1];
    int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    const size_t per_sample_size = static_cast<size_t>(in_channels) * static_cast<size_t>(in_height) * static_cast<size_t>(in_width);
    if (per_sample_size == 0 || input.size() % per_sample_size != 0) {
        std::cerr << "[SDRInferenceEngine] ERROR: Input size " << input.size()
                  << " is incompatible with convolution input shape for layer: " << layer.name << std::endl;
        return {};
    }
    const size_t runtime_batch = input.size() / per_sample_size;
    if (static_cast<size_t>(batch) != runtime_batch) {
        std::cerr << "[SDRInferenceEngine] INFO: Using runtime batch " << runtime_batch
                  << " (metadata batch " << batch << ") for layer: " << layer.name << std::endl;
    }
    const size_t expected_w = out_channels * in_channels * kernel_h * kernel_w;
    if (force_compressed_compute_ || layer.weights.size() != expected_w) {
        // Streaming sparse convolution using compressed weights
        std::vector<float> output(runtime_batch * out_channels * out_height * out_width, 0.0f);
        try {
            CortexAICompression::SDRIndexStorageStrategy s;
            s.forEachIndexValue(layer.raw_data, expected_w, [&](size_t flatIndex, float w) {
                // Map flatIndex -> (oc, ic, kh, kw)
                size_t oc = flatIndex / (in_channels * kernel_h * kernel_w);
                if (oc >= out_channels) return;
                size_t rem = flatIndex % (in_channels * kernel_h * kernel_w);
                size_t ic = rem / (kernel_h * kernel_w);
                rem = rem % (kernel_h * kernel_w);
                size_t kh = rem / kernel_w;
                size_t kw = rem % kernel_w;

                // Accumulate across batch and spatial positions
                for (size_t b = 0; b < runtime_batch; ++b) {
                    for (size_t oh = 0; oh < out_height; ++oh) {
                        int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
                        if (ih < 0 || ih >= static_cast<int>(in_height)) continue;
                        for (size_t ow = 0; ow < out_width; ++ow) {
                            int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
                            if (iw < 0 || iw >= static_cast<int>(in_width)) continue;
                            size_t in_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + static_cast<size_t>(ih) * in_width + static_cast<size_t>(iw);
                            size_t out_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow;
                            output[out_idx] += input[in_idx] * w;
                        }
                    }
                }
            });
            if (!layer.biases.empty()) {
                for (size_t b = 0; b < runtime_batch; ++b) {
                    for (size_t oc = 0; oc < out_channels; ++oc) {
                        for (size_t oh = 0; oh < out_height; ++oh) {
                            for (size_t ow = 0; ow < out_width; ++ow) {
                                size_t out_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow;
                                output[out_idx] += layer.biases[oc];
                            }
                        }
                    }
                }
            }
            last_layer_used_compressed_ = true;
            // Estimate retained ratio from compressed stream by counting decoded pairs
            size_t pair_count = 0;
            s.forEachIndexValue(layer.raw_data, expected_w, [&](size_t, float){ ++pair_count; });
            last_layer_retained_ratio_ = expected_w ? (static_cast<double>(pair_count) / static_cast<double>(expected_w)) : 0.0;
            return output;
        } catch (const std::exception& e) {
            std::cerr << "[SDRInferenceEngine] Compressed conv path failed: " << e.what() << std::endl;
            // Fall through to dense path error if weights also mismatch
            if (layer.weights.size() != expected_w) return {};
        }
    }
    if (layer.weights.size() != expected_w) {
        std::cerr << "[SDRInferenceEngine] ERROR: Weights size " << layer.weights.size() << " does not match expected " << expected_w << " for convolutional layer: " << layer.name << std::endl;
        return {};
    }
    if (!layer.biases.empty() && layer.biases.size() != static_cast<size_t>(out_channels)) {
        std::cerr << "[SDRInferenceEngine] ERROR: Biases size " << layer.biases.size() << " does not match out_channels " << out_channels << " for convolutional layer: " << layer.name << std::endl;
        return {};
    }
    
    // Use optimized im2col + GEMM convolution
    std::vector<float> output(runtime_batch * out_channels * out_height * out_width, 0.0f);
    
    CortexAICompression::Kernels::conv2d_im2col(
        input.data(),
        layer.weights.data(),
        layer.biases.empty() ? nullptr : layer.biases.data(),
        output.data(),
        static_cast<int>(runtime_batch),
        in_channels, in_height, in_width,
        out_channels,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );
    
    return output;
}

/**
 * @brief Apply batch normalization to input tensor.
 * @param layer LayerInfo containing batch norm parameters.
 * @param input Input tensor to normalize.
 * @return Normalized output tensor.
 */
std::vector<float> SDRInferenceEngine::applyBatchNorm(const LayerInfo& layer, const std::vector<float>& input) {
    if (!layer.properties.use_batch_norm) return input;
    if (layer.input_shape.size() < 2) {
        std::cerr << "[SDRInferenceEngine] WARNING: Missing channel dimension for batch norm layer: " << layer.name << std::endl;
        return input;
    }

    std::vector<float> output = input;
    size_t channels = layer.input_shape[1];
    if (channels == 0) return input;

    size_t effective_batch = batch_size > 0 ? batch_size : 1;
    size_t elems_per_batch = channels;
    if (layer.input_shape.size() > 2) {
        for (size_t i = 2; i < layer.input_shape.size(); ++i) {
            elems_per_batch *= layer.input_shape[i];
        }
    }
    if (elems_per_batch > 0 && input.size() % elems_per_batch == 0) {
        effective_batch = input.size() / elems_per_batch;
    }
    if (effective_batch == 0 || input.size() % (effective_batch * channels) != 0) {
        std::cerr << "[SDRInferenceEngine] ERROR: Invalid batch norm input size for layer: " << layer.name << std::endl;
        return {};
    }
    size_t spatial_size = input.size() / (effective_batch * channels);
    if (layer.properties.bn_running_mean.size() < channels ||
        layer.properties.bn_running_var.size() < channels ||
        layer.properties.bn_weights.size() < channels ||
        layer.properties.bn_biases.size() < channels) {
        std::cerr << "[SDRInferenceEngine] WARNING: Incomplete batch norm statistics, using pass-through for layer: " << layer.name << std::endl;
        return input;
    }

    for (size_t b = 0; b < effective_batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float mean = layer.properties.bn_running_mean[c];
            float var = layer.properties.bn_running_var[c];
            float gamma = layer.properties.bn_weights[c];
            float beta = layer.properties.bn_biases[c];

            for (size_t i = 0; i < spatial_size; ++i) {
                size_t idx = b * channels * spatial_size + c * spatial_size + i;
                output[idx] = gamma * (input[idx] - mean) / std::sqrt(var + 1e-5f) + beta;
            }
        }
    }

    if (training_mode) {
        updateBatchNormStats(layer, input);
    }

    return output;
}

/**
 * @brief Apply activation function to input tensor using SIMD-optimized kernels
 * @param type Activation function type (relu, gelu, swish, sigmoid, tanh, etc.)
 * @param input Input tensor
 * @return Output tensor with activation applied element-wise
 */
std::vector<float> SDRInferenceEngine::applyActivation(const std::string& type, const std::vector<float>& input) {
    // Use ping-pong buffer for output to reduce allocations
    std::vector<float>& output = getNextPingPongBuffer(input.size());
    
    // Use SIMD-optimized activation functions
    if (type == "relu") {
        CortexAICompression::Kernels::relu(input.data(), output.data(), input.size());
    } else if (type == "leaky_relu") {
        CortexAICompression::Kernels::leaky_relu(input.data(), output.data(), input.size(), 0.01f);
    } else if (type == "gelu") {
        CortexAICompression::Kernels::gelu(input.data(), output.data(), input.size());
    } else if (type == "swish" || type == "silu") {
        CortexAICompression::Kernels::swish(input.data(), output.data(), input.size());
    } else if (type == "sigmoid") {
        CortexAICompression::Kernels::sigmoid(input.data(), output.data(), input.size());
    } else if (type == "tanh") {
        CortexAICompression::Kernels::tanh_activation(input.data(), output.data(), input.size());
    } else if (type == "elu") {
        for (size_t index = 0; index < input.size(); ++index) {
            const float value = input[index];
            output[index] = value >= 0.0f ? value : (std::exp(value) - 1.0f);
        }
    } else if (type == "softmax") {
        CortexAICompression::Kernels::softmax(input.data(), output.data(), input.size());
    } else {
        // Unknown activation, pass through
        output = input;
    }

    return output;
}

/**
 * @brief Apply dropout regularization during training.
 * @param layer LayerInfo containing dropout rate.
 * @param input Input tensor.
 * @return Output tensor with dropout applied (or unchanged if disabled).
 */
std::vector<float> SDRInferenceEngine::applyDropout(const LayerInfo& layer, const std::vector<float>& input) {
    if (!dropout_enabled || layer.properties.dropout_rate <= 0.0f) return input;

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::vector<float> output = input;
    float scale = 1.0f / (1.0f - layer.properties.dropout_rate);

    for (float& val : output) {
        if (dis(gen) < layer.properties.dropout_rate) {
            val = 0.0f;
        } else {
            val *= scale;
    }
    }

    return output;
}

/**
 * @brief Apply 2x2 max pooling operation.
 * @param input Input tensor in NCHW format.
 * @param input_shape Shape of input tensor [N, C, H, W].
 * @return Output tensor after max pooling.
 */
std::vector<float> SDRInferenceEngine::applyMaxPool(const std::vector<float>& input, const std::vector<size_t>& input_shape) {
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t in_height = input_shape[2];
    size_t in_width = input_shape[3];
    size_t out_height = in_height / 2;
    size_t out_width = in_width / 2;
    
    std::vector<float> output(batch_size * channels * out_height * out_width);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (size_t kh = 0; kh < 2; ++kh) {
                        for (size_t kw = 0; kw < 2; ++kw) {
                            size_t in_h = oh * 2 + kh;
                            size_t in_w = ow * 2 + kw;
                            
                            if (in_h < in_height && in_w < in_width) {
                                size_t input_idx = b * channels * in_height * in_width +
                                                 c * in_height * in_width +
                                                 in_h * in_width + in_w;
                                max_val = std::max(max_val, input[input_idx]);
                            }
                        }
                    }
                    
                    size_t output_idx = b * channels * out_height * out_width +
                                      c * out_height * out_width +
                                      oh * out_width + ow;
                    output[output_idx] = max_val;
                }
            }
        }
    }
    
    return output;
}

/**
 * @brief Apply 2x2 average pooling operation
 * @param input Input tensor in NCHW format
 * @param input_shape Shape of input tensor [N, C, H, W]
 * @return Output tensor after average pooling
 */
std::vector<float> SDRInferenceEngine::applyAvgPool(const std::vector<float>& input, const std::vector<size_t>& input_shape) {
    size_t batch_size = input_shape[0];
    size_t channels = input_shape[1];
    size_t in_height = input_shape[2];
    size_t in_width = input_shape[3];
    size_t out_height = in_height / 2;
    size_t out_width = in_width / 2;
    
    std::vector<float> output(batch_size * channels * out_height * out_width);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    int count = 0;
                    
                    for (size_t kh = 0; kh < 2; ++kh) {
                        for (size_t kw = 0; kw < 2; ++kw) {
                            size_t in_h = oh * 2 + kh;
                            size_t in_w = ow * 2 + kw;
                            
                            if (in_h < in_height && in_w < in_width) {
                                size_t input_idx = b * channels * in_height * in_width +
                                                 c * in_height * in_width +
                                                 in_h * in_width + in_w;
                                sum += input[input_idx];
                                count++;
                            }
                        }
                    }
                    
                    size_t output_idx = b * channels * out_height * out_width +
                                      c * out_height * out_width +
                                      oh * out_width + ow;
                    output[output_idx] = sum / count;
                }
            }
        }
    }
    
    return output;
}

/**
 * @brief Update batch normalization running statistics during training
 * @param layer LayerInfo containing batch norm parameters
 * @param input Input tensor used for statistics calculation
 */
void SDRInferenceEngine::updateBatchNormStats(const LayerInfo& layer, const std::vector<float>& input) {
    if (!layer.properties.use_batch_norm) return;
    if (layer.input_shape.size() < 2) return;

    size_t channels = layer.input_shape[1];
    if (channels == 0) return;
    if (layer.properties.bn_running_mean.size() < channels ||
        layer.properties.bn_running_var.size() < channels) {
        return;
    }

    size_t effective_batch = batch_size > 0 ? batch_size : 1;
    size_t elems_per_batch = channels;
    if (layer.input_shape.size() > 2) {
        for (size_t i = 2; i < layer.input_shape.size(); ++i) {
            elems_per_batch *= layer.input_shape[i];
        }
    }
    if (elems_per_batch > 0 && input.size() % elems_per_batch == 0) {
        effective_batch = input.size() / elems_per_batch;
    }
    if (effective_batch == 0 || input.size() % (effective_batch * channels) != 0) {
        return;
    }
    size_t spatial_size = input.size() / (effective_batch * channels);
    float momentum = 0.1f;

    for (size_t c = 0; c < channels; ++c) {
        float mean = 0.0f;
        float var = 0.0f;

        // Calculate batch mean
        for (size_t b = 0; b < effective_batch; ++b) {
            for (size_t i = 0; i < spatial_size; ++i) {
                size_t idx = b * channels * spatial_size + c * spatial_size + i;
                mean += input[idx];
            }
        }
        mean /= (effective_batch * spatial_size);

        // Calculate batch variance
        for (size_t b = 0; b < effective_batch; ++b) {
            for (size_t i = 0; i < spatial_size; ++i) {
                size_t idx = b * channels * spatial_size + c * spatial_size + i;
                float diff = input[idx] - mean;
                var += diff * diff;
            }
        }
        var /= (effective_batch * spatial_size);

        // Update exponential moving averages
        layer.properties.bn_running_mean[c] = (1 - momentum) * layer.properties.bn_running_mean[c] + momentum * mean;
        layer.properties.bn_running_var[c] = (1 - momentum) * layer.properties.bn_running_var[c] + momentum * var;
    }
}

/**
 * @brief Reshape tensor to new dimensions
 * @param input Input tensor
 * @param shape Target shape dimensions
 * @return Reshaped tensor
 */
std::vector<float> SDRInferenceEngine::reshapeTensor(const std::vector<float>& input, const std::vector<size_t>& shape) {
    if (shape.empty()) {
        std::cerr << "[SDRInferenceEngine] ERROR: Empty shape provided for reshape" << std::endl;
        return input;
    }
    
    // Calculate total elements in target shape
    size_t total_elements = 1;
    for (size_t dim : shape) {
        total_elements *= dim;
    }
    
    if (total_elements != input.size()) {
        std::cerr << "[SDRInferenceEngine] Shape mismatch in reshape. Input size: "
                  << input.size() << ", Target size: " << total_elements << std::endl;
        return input;
    }
    
    // For now, return a copy of the input (reshape is just a view change)
    // In a more sophisticated implementation, we'd return a view with different indexing
    std::vector<float> result = input;
    return result;
}

/**
 * @brief Flatten multi-dimensional tensor to 1D
 * @param input Input tensor
 * @return Flattened tensor
 */
std::vector<float> SDRInferenceEngine::flattenTensor(const std::vector<float>& input) {
    // For tensors already in 1D memory layout, flatten is a no-op on data
    // However, we validate the operation is semantically valid
    if (!Utils::TensorValidator::is_finite(input)) {
        std::cerr << "[SDRInferenceEngine] WARNING: Non-finite values detected in flatten operation" << std::endl;
    }
    return input;
}

/**
 * @brief Run complete neural network inference with on-demand layer loading.
 * @param input_tensor Input tensor data for inference.
 * @return Output tensor after processing through all network layers.
 *
 * @details Pipeline:
 * 1) Determine execution order from segment metadata
 * 2) Execute layers using on-demand loading
 * 3) Collect and log statistics about execution
 */
std::vector<float> SDRInferenceEngine::run(const std::vector<float>& input_tensor) {
    auto run_start = std::chrono::high_resolution_clock::now();
    last_run_stats_ = RunStats{};
    intermediate_tensors_.clear();
    execution_graph_.reset();
    const auto& segments = loader_.getSegmentIndex();
    if (segments.empty()) {
        std::cerr << "[SDRInferenceEngine] No segments found in the model loader!" << std::endl;
        return input_tensor;
    }


    std::vector<std::string> layer_names = getExecutionOrder(segments);
    
    if (layer_names.empty()) {
        std::cerr << "[SDRInferenceEngine] No executable layers found!" << std::endl;
        return input_tensor;
    }

    if (!execution_graph_) {
        try {
            execution_graph_ = std::make_unique<Utils::ExecutionGraph>();
            for (const auto& layer_name : layer_names) {
                execution_graph_->add_node(Utils::GraphNode(layer_name, "layer"));
            }
            for (size_t index = 1; index < layer_names.size(); ++index) {
                execution_graph_->add_edge(layer_names[index - 1], layer_names[index]);
            }
        } catch (...) {
            execution_graph_.reset();
        }
    }


    auto result = runLayersOnDemand(layer_names, input_tensor);
    auto run_end = std::chrono::high_resolution_clock::now();
    last_run_stats_.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(run_end - run_start).count();


    return result;
}

// --- Memory Management for Large Models ---

/**
 * @brief Initialize memory pool for efficient tensor allocation
 * @param max_memory_mb Maximum memory pool size in MB
 */
void SDRInferenceEngine::initializeMemoryPool(size_t max_memory_mb) {
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    
    max_memory_usage_ = max_memory_mb * 1024ULL * 1024ULL;
    size_t pool_size = max_memory_usage_ / sizeof(float);
    
    memory_pool_.resize(pool_size, 0.0f);
    memory_pool_offset_ = 0;
}

/**
 * @brief Clean up memory pool and reset offset
 */
void SDRInferenceEngine::cleanupMemoryPool() {
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    
    std::fill(memory_pool_.begin(), memory_pool_.end(), 0.0f);
    memory_pool_offset_ = 0;
    free_list_.clear();
}

/**
 * @brief Deallocate memory from pool by marking block as free
 * @param ptr Pointer to memory block to deallocate
 * 
 * Note: Pass the pointer returned by allocateFromPool, not the size
 */
void SDRInferenceEngine::deallocateFromPool(float* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    
    // Find the block corresponding to this pointer
    size_t offset = ptr - memory_pool_.data();
    
    for (auto& block : free_list_) {
        if (block.offset == offset && !block.is_free) {
            block.is_free = true;
            
            // Try to coalesce with adjacent free blocks
            coalesceFreeBlocks();
            return;
        }
    }
    
    std::cerr << "[SDRInferenceEngine] WARNING: Attempted to deallocate unknown block" << std::endl;
}

/**
 * @brief Legacy interface for deallocateFromPool (deprecated)
 */
void SDRInferenceEngine::deallocateFromPool(size_t size) {
    // Legacy interface - search by size (less efficient)
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    
    for (auto& block : free_list_) {
        if (block.size == size && !block.is_free) {
            block.is_free = true;
            coalesceFreeBlocks();
            return;
        }
    }
}

/**
 * @brief Enable or disable aggressive memory management
 * @param enable True to enable aggressive cleanup
 */
void SDRInferenceEngine::enableAggressiveMemoryManagement(bool enable) {
    aggressive_memory_management_ = enable;
    std::cout << "[SDRInferenceEngine] Aggressive memory management "
              << (enable ? "enabled" : "disabled") << std::endl;
}

/**
 * @brief Get current memory usage in bytes
 * @return Current memory usage
 */
size_t SDRInferenceEngine::getCurrentMemoryUsage() const {
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    return memory_pool_offset_ * sizeof(float);
}

// --- Critical Operations for Large Models ---

/**
 * @brief Execute multi-head self-attention for transformer models
 * @param layer LayerInfo containing attention parameters
 * @param input Input tensor (batch_size x seq_len x hidden_dim)
 * @return Output tensor after attention computation
 */
std::vector<float> SDRInferenceEngine::executeAttentionOperation(const LayerInfo& layer, const std::vector<float>& input) {
    if (layer.weights.empty()) {
        std::cerr << "[SDRInferenceEngine] ERROR: No weights found for attention layer" << std::endl;
        return input;
    }
    
    // Detect dimensions from shapes
    size_t batch_size = 1;
    size_t seq_len = 1;
    size_t hidden_dim = input.size();
    
    if (!layer.input_shape.empty()) {
        if (layer.input_shape.size() == 2) {
            seq_len = layer.input_shape[0];
            hidden_dim = layer.input_shape[1];
        } else if (layer.input_shape.size() == 3) {
            batch_size = layer.input_shape[0];
            seq_len = layer.input_shape[1];
            hidden_dim = layer.input_shape[2];
        }
    }
    
    // Typical attention has num_heads=8 or 12
    size_t num_heads = 8;
    if (hidden_dim % num_heads != 0) {
        // Try common head counts
        for (size_t nh : {12, 16, 4}) {
            if (hidden_dim % nh == 0) {
                num_heads = nh;
                break;
            }
        }
    }
    
    // Full multi-head attention implementation with proper Q, K, V projections
    
    size_t qkv_dim = hidden_dim;  // Assuming same dimension for Q, K, V
    size_t head_dim = hidden_dim / num_heads;
    
    std::vector<float>& output = getNextPingPongBuffer(input.size());
    
    // Check if we have separate Q, K, V weight matrices in the layer
    // Expected weight layout: [Wq | Wk | Wv] concatenated (3 * hidden_dim x hidden_dim)
    if (!layer.weights.empty() && layer.weights.size() >= 3 * hidden_dim * hidden_dim) {
        // Project input to Q, K, V
        std::vector<float> query(input.size());
        std::vector<float> key(input.size());
        std::vector<float> value(input.size());
        
        // Extract weight matrices
        const float* Wq = layer.weights.data();
        const float* Wk = layer.weights.data() + hidden_dim * hidden_dim;
        const float* Wv = layer.weights.data() + 2 * hidden_dim * hidden_dim;
        
        // Project to Q, K, V spaces
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t t = 0; t < seq_len; ++t) {
                size_t offset = (b * seq_len + t) * hidden_dim;
                const float* x = input.data() + offset;
                
                // Q = x @ Wq^T
                CortexAICompression::Kernels::gemv(Wq, x, query.data() + offset, 
                                                    hidden_dim, hidden_dim);
                // K = x @ Wk^T
                CortexAICompression::Kernels::gemv(Wk, x, key.data() + offset,
                                                    hidden_dim, hidden_dim);
                // V = x @ Wv^T
                CortexAICompression::Kernels::gemv(Wv, x, value.data() + offset,
                                                    hidden_dim, hidden_dim);
            }
        }
        
        // Use Flash Attention if available for memory efficiency
        CortexAICompression::Kernels::FlashAttentionConfig config;
        config.use_causal_mask = (layer.properties.activation_type == "causal");
        
        CortexAICompression::Kernels::flash_attention_forward(
            query.data(),
            key.data(),
            value.data(),
            output.data(),
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
            config
        );
        
        // Apply output projection if available
        if (layer.weights.size() >= 4 * hidden_dim * hidden_dim) {
            const float* Wo = layer.weights.data() + 3 * hidden_dim * hidden_dim;
            std::vector<float> temp = output;
            
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t t = 0; t < seq_len; ++t) {
                    size_t offset = (b * seq_len + t) * hidden_dim;
                    CortexAICompression::Kernels::gemv(Wo, temp.data() + offset,
                                                        output.data() + offset,
                                                        hidden_dim, hidden_dim);
                }
            }
        }
    } else {
        // Fallback: use input as Q, K, V (self-attention without learned projections)
        std::cerr << "[SDRInferenceEngine] WARNING: No QKV weights found, using input directly" << std::endl;
        
        CortexAICompression::Kernels::multi_head_attention(
            input.data(),  // Query
            input.data(),  // Key  
            input.data(),  // Value
            output.data(),
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
            false  // causal mask
        );
    }
    
    return output;
}

/**
 * @brief Execute normalization operation (LayerNorm, BatchNorm, etc.)
 * @param layer LayerInfo containing normalization parameters
 * @param input Input tensor
 * @return Output tensor after normalization
 */
std::vector<float> SDRInferenceEngine::executeNormalizationOperation(const LayerInfo& layer, const std::vector<float>& input) {
    // Check if this is LayerNorm vs BatchNorm
    std::string lower_type = layer.layer_type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    
    if (lower_type.find("layer_norm") != std::string::npos ||
        lower_type.find("layernorm") != std::string::npos ||
        lower_type == "norm") {
        // Use SIMD-optimized LayerNorm
        std::vector<float>& output = getNextPingPongBuffer(input.size());

        const std::vector<float>& norm_weights =
            !layer.properties.bn_weights.empty() ? layer.properties.bn_weights : layer.weights;
        const std::vector<float>& norm_biases =
            !layer.properties.bn_biases.empty() ? layer.properties.bn_biases : layer.biases;

        if (!norm_weights.empty()) {
            std::vector<float> zero_biases;
            const float* bias_ptr = nullptr;
            if (!norm_biases.empty()) {
                bias_ptr = norm_biases.data();
            } else {
                zero_biases.assign(norm_weights.size(), 0.0f);
                bias_ptr = zero_biases.data();
            }

            CortexAICompression::Kernels::layer_norm(
                input.data(),
                output.data(),
                norm_weights.data(),
                bias_ptr,
                input.size(),
                1e-5f
            );
            return output;
        }
    }
    
    // Fall back to batch normalization
    return applyBatchNorm(layer, input);
}

/**
 * @brief Execute activation operation
 * @param layer LayerInfo containing activation parameters
 * @param input Input tensor
 * @return Output tensor after activation
 */
std::vector<float> SDRInferenceEngine::executeActivationOperation(const LayerInfo& layer, const std::vector<float>& input) {
    
    std::string activation_type = layer.properties.activation_type;
    if (activation_type.empty()) {
        std::string lower_type = toLowerCopy(layer.layer_type);
        if (lower_type.find("sigmoid") != std::string::npos) {
            activation_type = "sigmoid";
        } else if (lower_type.find("tanh") != std::string::npos) {
            activation_type = "tanh";
        } else if (lower_type.find("softmax") != std::string::npos) {
            activation_type = "softmax";
        } else if (lower_type.find("gelu") != std::string::npos) {
            activation_type = "gelu";
        } else if (lower_type.find("leakyrelu") != std::string::npos || lower_type.find("leaky_relu") != std::string::npos) {
            activation_type = "leaky_relu";
        } else if (lower_type.find("elu") != std::string::npos) {
            activation_type = "elu";
        } else if (lower_type.find("silu") != std::string::npos || lower_type.find("swish") != std::string::npos) {
            activation_type = "silu";
        } else {
            activation_type = "relu";
        }
    }
    
    return applyActivation(activation_type, input);
}

/**
 * @brief Execute pooling operation
 * @param layer LayerInfo containing pooling parameters
 * @param input Input tensor
 * @return Output tensor after pooling
 */
std::vector<float> SDRInferenceEngine::executePoolingOperation(const LayerInfo& layer, const std::vector<float>& input) {
    
    if (layer.input_shape.size() < 4) {
        return input;
    }

    const std::string lower_type = toLowerCopy(layer.layer_type);
    const std::string lower_activation = toLowerCopy(layer.properties.activation_type);
    if (lower_type.find("maxpool") != std::string::npos || lower_activation == "max") {
        return applyMaxPool(input, layer.input_shape);
    } else {
        return applyAvgPool(input, layer.input_shape);
    }
}

/**
 * @brief Execute elementwise operation (Add, Mul, etc.)
 * @param layer LayerInfo containing elementwise parameters
 * @param input Input tensor
 * @return Output tensor after elementwise operation
 */
std::vector<float> SDRInferenceEngine::executeElementwiseOperation(const LayerInfo& layer, const std::vector<float>& input) {
    std::string lower_type = layer.layer_type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
    
    // Route to specific elementwise operation
    if (lower_type.find("concat") != std::string::npos) {
        return executeConcatOperation(layer, input);
    } else if (lower_type.find("add") != std::string::npos || lower_type.find("sum") != std::string::npos) {
        return executeAddOperation(layer, input);
    } else if (lower_type.find("sub") != std::string::npos || lower_type.find("subtract") != std::string::npos) {
        return executeSubOperation(layer, input);
    } else if (lower_type.find("mul") != std::string::npos || lower_type.find("multiply") != std::string::npos) {
        return executeMulOperation(layer, input);
    } else if (lower_type.find("div") != std::string::npos || lower_type.find("divide") != std::string::npos) {
        return executeDivOperation(layer, input);
    } else if (lower_type.find("slice") != std::string::npos) {
        return executeSliceOperation(layer, input);
    } else if (lower_type.find("reshape") != std::string::npos) {
        return executeReshapeOperation(layer, input);
    } else if (lower_type.find("transpose") != std::string::npos) {
        return executeTransposeOperation(layer, input);
    } else if (lower_type.find("flatten") != std::string::npos) {
        return executeFlattenOperation(layer, input);
    }
    
    // Unknown elementwise operation - log and pass through
    std::cerr << "[SDRInferenceEngine] WARNING: Unknown elementwise operation: " << layer.layer_type << std::endl;
    return input;
}

/**
 * @brief Adaptive fallback for unknown operations
 * @param layer LayerInfo containing layer parameters
 * @param input Input tensor
 * @return Output tensor (pass-through for unknown operations)
 */
std::vector<float> SDRInferenceEngine::executeAdaptiveFallback(const LayerInfo& layer, const std::vector<float>& input) {
    // Adaptive fallback with intelligent guessing based on layer properties
    
    // Log the unknown operation for debugging
    std::cerr << "[SDRInferenceEngine] INFO: Using adaptive fallback for layer: " 
              << layer.name << " (type: " << layer.layer_type << ")" << std::endl;
    
    // Try to infer operation from available data
    bool has_weights = !layer.weights.empty();
    bool has_biases = !layer.biases.empty();
    bool has_valid_shapes = !layer.input_shape.empty() && !layer.output_shape.empty();
    
    // Attempt learned transformation if weights are available
    if (has_weights && has_valid_shapes) {
        size_t input_size = 1;
        for (size_t d : layer.input_shape) input_size *= d;
        size_t output_size = 1;
        for (size_t d : layer.output_shape) output_size *= d;
        
        // If dimensions suggest linear transformation
        if (layer.weights.size() == input_size * output_size && input.size() == input_size) {
            std::cerr << "[SDRInferenceEngine] Attempting linear transformation fallback" << std::endl;
            try {
                return applyLinearLayer(layer, input);
            } catch (...) {
                std::cerr << "[SDRInferenceEngine] Linear fallback failed" << std::endl;
            }
        }
    }
    
    // Attempt bias-only transformation
    if (has_biases && layer.biases.size() == input.size()) {
        std::cerr << "[SDRInferenceEngine] Applying bias-only transformation" << std::endl;
        std::vector<float> output = input;
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += layer.biases[i];
        }
        return output;
    }
    
    // Last resort: identity transformation with validation
    if (!Utils::TensorValidator::is_finite(input)) {
        std::cerr << "[SDRInferenceEngine] ERROR: Non-finite values in fallback layer" << std::endl;
    }
    
    std::cerr << "[SDRInferenceEngine] Using identity transformation (pass-through)" << std::endl;
    return input;
}

#ifdef ENABLE_ONNX_PROTOBUF
const std::optional<onnx::ModelProto>& SDRModelLoader::getLoadedModelProto() const {
    return loaded_model_proto_;
}
#endif

/**
 * @brief Execute a single neural network layer.
 * @param layer LayerInfo containing layer parameters and weights.
 * @param input Input tensor for the layer.
 * @return Output tensor after layer processing.
 *
 * @details Uses a dispatch table for known types, otherwise falls back to
 * dynamic layer detection via executeDynamicLayer.
 */
std::vector<float> SDRInferenceEngine::runLayer(const LayerInfo& layer, const std::vector<float>& input) {
    encountered_layer_types_.insert(layer.layer_type);
    auto it = op_dispatch_.find(layer.layer_type);
    if (it != op_dispatch_.end()) {
        return it->second(layer, input);
    } else {
        unhandled_layer_types_.insert(layer.layer_type);
        return default_handler_(layer, input);
    }
}

/**
 * @brief Determine an execution order for neural network layers.
 * @param segments Vector of compressed segment headers from model.
 * @return Ordered list of layer names for execution.
 *
 * @details Groups weight segments by base name and performs numeric-aware
 * sorting to handle common naming conventions (e.g., transformer blocks).
 */
std::vector<std::string> SDRInferenceEngine::getExecutionOrder(const std::vector<CompressedSegmentHeader>& segments) {
#ifdef ENABLE_ONNX_PROTOBUF
    const auto& model_proto_opt = loader_.getLoadedModelProto();
    if (model_proto_opt && model_proto_opt->has_graph()) {
        const onnx::GraphProto& graph_proto = model_proto_opt->graph();
        std::unordered_set<std::string> available_layers;
        for (const auto& seg : segments) {
            if (isWeightLikeSegmentType(seg.original_type) || isExecutableOpLayerType(seg.layer_type)) {
                const std::string exec_name = deriveExecutionName(seg);
                if (!exec_name.empty()) {
                    available_layers.insert(exec_name);
                }
            }
        }

        if (!available_layers.empty()) {
            try {
                auto graph = std::make_unique<Utils::ExecutionGraph>();
                std::unordered_map<std::string, std::string> tensor_producers;
                std::vector<std::string> graph_node_order;
                graph_node_order.reserve(static_cast<size_t>(graph_proto.node_size()));

                size_t node_index = 0;
                for (const auto& node : graph_proto.node()) {
                    ++node_index;
                    std::string node_name = node.name();
                    if (node_name.empty()) {
                        if (!node.output().empty() && !node.output(0).empty()) {
                            node_name = node.output(0);
                        } else {
                            node_name = "node_" + std::to_string(node_index) + "_" + node.op_type();
                        }
                    }

                    if (available_layers.find(node_name) == available_layers.end()) {
                        for (const auto& output : node.output()) {
                            if (!output.empty() && available_layers.find(output) != available_layers.end()) {
                                node_name = output;
                                break;
                            }
                        }
                    }

                    if (available_layers.find(node_name) == available_layers.end()) {
                        continue;
                    }

                    graph->add_node(Utils::GraphNode(node_name, node.op_type()));
                    graph_node_order.push_back(node_name);

                    for (const auto& input : node.input()) {
                        auto producer_it = tensor_producers.find(input);
                        if (producer_it != tensor_producers.end()) {
                            const std::string& producer = producer_it->second;
                            if (!producer.empty() && producer != node_name) {
                                graph->add_edge(producer, node_name);
                            }
                        }
                    }
                    for (const auto& output : node.output()) {
                        if (!output.empty()) {
                            tensor_producers[output] = node_name;
                        }
                    }
                }

                if (!graph_node_order.empty()) {
                    execution_graph_ = std::move(graph);
                    return execution_graph_->get_execution_order();
                }
            } catch (const std::exception& e) {
                std::cerr << "[SDRInferenceEngine] WARNING: Failed to build ONNX execution graph from archive metadata: "
                          << e.what() << std::endl;
            }
        }
    }
#endif

    std::vector<CompressedSegmentHeader> ordered_candidates;
    ordered_candidates.reserve(segments.size());

    for (const auto& seg : segments) {
        if (isWeightLikeSegmentType(seg.original_type) || isExecutableOpLayerType(seg.layer_type)) {
            ordered_candidates.push_back(seg);
        }
    }

    std::sort(ordered_candidates.begin(), ordered_candidates.end(), [](const CompressedSegmentHeader& a, const CompressedSegmentHeader& b) {
        const bool a_has_order = !a.layer_name.empty();
        const bool b_has_order = !b.layer_name.empty();
        if (a_has_order != b_has_order) {
            return a_has_order > b_has_order;
        }
        if (a_has_order && b_has_order) {
            if (a.layer_index != b.layer_index) {
                return a.layer_index < b.layer_index;
            }
            if (a.layer_name != b.layer_name) {
                return a.layer_name < b.layer_name;
            }
        }
        return a.name < b.name;
    });

    std::vector<std::string> layer_names;
    layer_names.reserve(ordered_candidates.size());
    std::unordered_set<std::string> seen;
    for (const auto& seg : ordered_candidates) {
        const std::string exec_name = deriveExecutionName(seg);
        if (!exec_name.empty() && seen.insert(exec_name).second) {
            layer_names.push_back(exec_name);
        }
    }

    return layer_names;
}

std::vector<float> SDRInferenceEngine::runLayers(const std::vector<std::string>& layer_names, const std::vector<float>& input) {
    return runLayersOnDemand(layer_names, input);
}

/**
 * @brief Execute layers with on-demand loading.
 * @param layer_names Ordered list of layer names to execute.
 * @param input Initial input tensor.
 * @return Final output tensor after all layers processed.
 *
 * @details Times each layer's load/execute phases and logs progress. Optional
 * cache eviction can be enabled to further reduce memory footprint.
 */
std::vector<float> SDRInferenceEngine::runLayersOnDemand(const std::vector<std::string>& layer_names, const std::vector<float>& input) {
    std::vector<float> current = input;
    if (layer_names.empty()) return current;

    // Prefetch first layer if prefetching is enabled
    if (enable_prefetch_ && !layer_names.empty()) {
        prefetchNextLayer(layer_names[0]);
    }

    for (size_t i = 0; i < layer_names.size(); ++i) {
        const std::string& layer_name = layer_names[i];
        
        // Prefetch next layer while processing current one
        if (enable_prefetch_ && i + 1 < layer_names.size()) {
            prefetchNextLayer(layer_names[i + 1]);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        LayerInfo current_layer;
        
        try {
            current_layer = getPrefetchedLayer(layer_name);
        } catch (const std::exception& e) {
            std::cerr << "[SDRInferenceEngine] ERROR: Failed to load layer '" << layer_name << "': " << e.what() << std::endl;
            continue;
        }
        
        auto load_time = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_time - start_time);
        
        
        last_layer_used_compressed_ = false;
        std::vector<float> layer_output = runLayer(current_layer, current);
        
        auto exec_time = std::chrono::high_resolution_clock::now();
        auto exec_duration = std::chrono::duration_cast<std::chrono::milliseconds>(exec_time - load_time);
        
        
        if (layer_output.empty() && !current.empty()) {
            std::cerr << "[SDRInferenceEngine] WARNING: Layer '" << layer_name << "' produced empty output, using pass-through" << std::endl;
        } else {
            current = std::move(layer_output);
        }
        // Record per-layer stats
        LayerExecStat stat;
        stat.name = layer_name;
        stat.load_ms = load_duration.count();
        stat.exec_ms = exec_duration.count();
        stat.output_size = current.size();
        stat.used_compressed = last_layer_used_compressed_;
        stat.op_type = current_layer.layer_type;
        stat.retained_index_ratio = last_layer_retained_ratio_;
        last_run_stats_.layers.push_back(std::move(stat));
        
        // Store intermediate tensor for multi-input layers
        if (execution_graph_) {
            intermediate_tensors_[layer_name] = current;
        }
        
        // Aggressive memory management for large models
        if (aggressive_memory_management_) {
            loader_.clearLayerFromCache(layer_name);
            
            // Clean up old intermediate tensors that are no longer needed
            if (execution_graph_) {
                auto successors = execution_graph_->get_successors(layer_name);
                if (successors.empty()) {
                    // This is an output layer, can clean up
                    intermediate_tensors_.erase(layer_name);
                }
            }
            
            // Force garbage collection every 10 layers for very large models
            if ((i + 1) % 10 == 0) {
                current.shrink_to_fit();
            }
        }
    }
    
    return current;
}

/**
 * @brief Execute layer using dynamic type detection.
 * @param layer LayerInfo with properties for operation detection.
 * @param input Input tensor.
 * @return Output tensor after dynamic operation execution.
 *
 * @details Analyzes properties and naming to select an appropriate operation.
 * Supports heterogeneous architectures including transformers, CNNs, and MLPs.
 */
std::vector<float> SDRInferenceEngine::executeDynamicLayer(const LayerInfo& layer, const std::vector<float>& input) {
    
    // Analyze layer properties for operation detection
    bool has_weights = !layer.weights.empty();
    bool has_biases = !layer.biases.empty();
    bool has_kernel_shape = !layer.properties.kernel_shape.empty();
    bool has_activation = !layer.properties.activation_type.empty();
    bool has_dropout = layer.properties.dropout_rate > 0.0f;
    bool has_batch_norm = layer.properties.use_batch_norm;
    bool has_input_shape = !layer.input_shape.empty();
    bool has_output_shape = !layer.output_shape.empty();
    
    // Case-insensitive operation detection
    std::string lower_name = layer.name;
    std::string lower_type = layer.layer_type;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    
    std::string operation = "unknown";
    
    // Operation detection priority order for diverse model support
    if (lower_type.find("concat") != std::string::npos ||
        lower_type.find("slice") != std::string::npos ||
        lower_type.find("add") != std::string::npos ||
        lower_type.find("sub") != std::string::npos ||
        lower_type.find("mul") != std::string::npos ||
        lower_type.find("div") != std::string::npos ||
        lower_type.find("reshape") != std::string::npos ||
        lower_type.find("transpose") != std::string::npos ||
        lower_type.find("flatten") != std::string::npos) {
        operation = "elementwise";
    }
    else if (lower_name.find("attn") != std::string::npos ||
             lower_name.find("attention") != std::string::npos ||
             lower_type.find("attention") != std::string::npos ||
             lower_type.find("multihead") != std::string::npos) {
        operation = "attention";
    }
    else if (lower_type.find("embedding") != std::string::npos ||
             lower_name.find("token_embd") != std::string::npos ||
             lower_name.find("tok_embeddings") != std::string::npos ||
             lower_name.find("embedding") != std::string::npos) {
        operation = "linear";
    }
    else if (lower_name.find("ln") != std::string::npos ||
             lower_name.find("norm") != std::string::npos ||
             lower_type.find("norm") != std::string::npos ||
             lower_type.find("batch_norm") != std::string::npos ||
             lower_type.find("layer_norm") != std::string::npos) {
        operation = "normalization";
    }
    else if (!has_weights && !has_biases && 
             (has_activation || 
              lower_type.find("relu") != std::string::npos ||
              lower_type.find("gelu") != std::string::npos ||
              lower_type.find("sigmoid") != std::string::npos ||
              lower_type.find("tanh") != std::string::npos ||
              lower_type.find("softmax") != std::string::npos)) {
        operation = "activation";
    }
    else if (!has_weights && !has_biases && has_kernel_shape &&
             (lower_type.find("pool") != std::string::npos ||
              lower_name.find("pool") != std::string::npos)) {
        operation = "pooling";
    }
    else if (has_weights && has_kernel_shape && 
             (lower_type.find("conv") != std::string::npos ||
              lower_name.find("conv") != std::string::npos)) {
        operation = "convolution";
    }
    else if (has_weights && !has_kernel_shape && 
             (lower_type.find("linear") != std::string::npos || 
              lower_type.find("matmul") != std::string::npos ||
              lower_type.find("gemm") != std::string::npos ||
              lower_type.find("dense") != std::string::npos ||
              lower_name.find("linear") != std::string::npos ||
              lower_name.find("fc") != std::string::npos ||
              lower_name.find("dense") != std::string::npos)) {
        operation = "linear";
    }
    else if (has_weights && has_input_shape && has_output_shape) {
        if (has_kernel_shape) {
            operation = "convolution";
        } else {
            operation = "linear";
        }
    }
    else if (!has_weights && !has_biases) {
        operation = "elementwise";
    }
    
    
    try {
        if (operation == "linear") {
            return executeLinearOperation(layer, input);
        } else if (operation == "convolution") {
            return executeConvolutionalOperation(layer, input);
        } else if (operation == "attention") {
            return executeAttentionOperation(layer, input);
        } else if (operation == "normalization") {
            return executeNormalizationOperation(layer, input);
        } else if (operation == "activation") {
            return executeActivationOperation(layer, input);
        } else if (operation == "pooling") {
            return executePoolingOperation(layer, input);
        } else if (operation == "elementwise") {
            return executeElementwiseOperation(layer, input);
        } else {
            return executeAdaptiveFallback(layer, input);
        }
    } catch (const std::exception& e) {
        std::cerr << "[SDRInferenceEngine] Error executing layer '" << layer.name << "': " << e.what() << std::endl;
        return input;
    }
}

/**
 * @brief Execute linear operation with shape adaptation
 * @param layer LayerInfo containing linear layer parameters
 * @param input Input tensor
 * @return Output tensor after linear transformation
 */
std::vector<float> SDRInferenceEngine::executeLinearOperation(const LayerInfo& layer, const std::vector<float>& input) {
    
    if (layer.input_shape.empty() || layer.output_shape.empty()) {
        return input;
    }
    
    size_t input_size = 1;
    for (size_t d : layer.input_shape) input_size *= d;
    size_t output_size = 1;
    for (size_t d : layer.output_shape) output_size *= d;
    
    if (input.size() != input_size) {
        
        if (input.size() > input_size) {
            std::vector<float> adapted_input(input.begin(), input.begin() + input_size);
            return applyLinearLayer(layer, adapted_input);
        } else if (input.size() < input_size) {
            std::vector<float> adapted_input = input;
            adapted_input.resize(input_size, 0.0f);
            return applyLinearLayer(layer, adapted_input);
        }
    }
    
    return applyLinearLayer(layer, input);
}

std::vector<float> SDRInferenceEngine::executeConvolutionalOperation(const LayerInfo& layer, const std::vector<float>& input) {
    
    // Validate shapes for convolutional layers
    if (layer.input_shape.size() != 4 || layer.output_shape.size() != 4) {
        return input;
    }
    
    return applyConvolutionalLayer(layer, input);
}


// Element-wise operation implementations
std::vector<float> SDRInferenceEngine::executeConcatOperation(const LayerInfo& layer, const std::vector<float>& input) {
    // Check if layer has multiple inputs defined in execution graph
    if (execution_graph_) {
        auto predecessors = execution_graph_->get_predecessors(layer.name);
        if (predecessors.size() > 1) {
            // Gather all input tensors
            std::vector<std::vector<float>> inputs;
            size_t total_size = 0;
            
            for (const auto& pred : predecessors) {
                auto it = intermediate_tensors_.find(pred);
                if (it != intermediate_tensors_.end()) {
                    inputs.push_back(it->second);
                    total_size += it->second.size();
                }
            }
            
            // Concatenate along the last dimension (default)
            std::vector<float> result;
            result.reserve(total_size);
            for (const auto& tensor : inputs) {
                result.insert(result.end(), tensor.begin(), tensor.end());
            }
            return result;
        }
    }
    
    // Single input - pass through
    return input;
}

std::vector<float> SDRInferenceEngine::executeSliceOperation(const LayerInfo& layer, const std::vector<float>& input) {
    // Extract slice parameters from layer metadata
    // Default: return first half if no parameters specified
    size_t start = 0;
    size_t end = input.size() / 2;
    
    // Check if layer has slice parameters in metadata
    // (This would need to be parsed from model format)
    if (!layer.output_shape.empty() && layer.output_shape[0] > 0) {
        size_t slice_size = 1;
        for (size_t dim : layer.output_shape) {
            slice_size *= dim;
        }
        end = std::min(start + slice_size, input.size());
    }
    
    if (end <= start || start >= input.size()) {
        return input;
    }
    
    return std::vector<float>(input.begin() + start, input.begin() + end);
}

std::vector<float> SDRInferenceEngine::executeAddOperation(const LayerInfo& layer, const std::vector<float>& input) {
    // Check for multi-input add (residual connection)
    if (execution_graph_) {
        auto predecessors = execution_graph_->get_predecessors(layer.name);
        if (predecessors.size() > 1) {
            // Element-wise addition of multiple inputs
            std::vector<float> result = input;
            
            for (size_t i = 1; i < predecessors.size(); ++i) {
                auto it = intermediate_tensors_.find(predecessors[i]);
                if (it != intermediate_tensors_.end()) {
                    const auto& other = it->second;
                    if (other.size() == result.size()) {
                        for (size_t j = 0; j < result.size(); ++j) {
                            result[j] += other[j];
                        }
                    }
                }
            }
            return result;
        }
    }
    
    // Single input with bias
    std::vector<float> output = input;
    if (!layer.biases.empty() && layer.biases.size() == input.size()) {
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += layer.biases[i];
        }
    }
    return output;
}

std::vector<float> SDRInferenceEngine::executeSubOperation(const LayerInfo& layer, const std::vector<float>& input) {
    
    // Element-wise subtraction with bias if available
    std::vector<float> output = input;
    if (!layer.biases.empty() && layer.biases.size() == input.size()) {
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] -= layer.biases[i];
        }
    }
    return output;
}

std::vector<float> SDRInferenceEngine::executeMulOperation(const LayerInfo& layer, const std::vector<float>& input) {
    
    // Element-wise multiplication with weights if available
    std::vector<float> output = input;
    if (!layer.weights.empty() && layer.weights.size() == input.size()) {
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] *= layer.weights[i];
        }
    }
    return output;
}

std::vector<float> SDRInferenceEngine::executeDivOperation(const LayerInfo& layer, const std::vector<float>& input) {
    
    // Element-wise division with weights if available
    std::vector<float> output = input;
    if (!layer.weights.empty() && layer.weights.size() == input.size()) {
        for (size_t i = 0; i < output.size(); ++i) {
            if (std::abs(layer.weights[i]) > 1e-8f) { // Avoid division by zero
                output[i] /= layer.weights[i];
            }
        }
    }
    return output;
}

std::vector<float> SDRInferenceEngine::executeReshapeOperation(const LayerInfo& layer, const std::vector<float>& input) {
    // Reshape is just a metadata change - data layout stays the same
    // Verify sizes match
    if (!layer.output_shape.empty()) {
        size_t expected_size = 1;
        for (size_t dim : layer.output_shape) {
            expected_size *= dim;
        }
        if (expected_size != input.size()) {
            std::cerr << "[SDRInferenceEngine] WARNING: Reshape size mismatch. Input: " 
                      << input.size() << ", Expected: " << expected_size << std::endl;
        }
    }
    return input;  // Data layout is unchanged
}

std::vector<float> SDRInferenceEngine::executeTransposeOperation(const LayerInfo& layer, const std::vector<float>& input) {
    // Implement 2D matrix transpose (most common case)
    if (layer.input_shape.size() == 2) {
        size_t rows = layer.input_shape[0];
        size_t cols = layer.input_shape[1];
        
        std::vector<float>& output = getNextPingPongBuffer(input.size());
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                output[j * rows + i] = input[i * cols + j];
            }
        }
        return output;
    }
    
    // For higher-dimensional tensors, return as-is (needs permutation spec)
    return input;
}

std::vector<float> SDRInferenceEngine::executeFlattenOperation(const LayerInfo& layer, const std::vector<float>& input) {
    
    // Flatten operation - input is already 1D, so just return as-is
    return input;
}

// --- New Helper Methods ---

/**
 * @brief Get next ping-pong buffer for tensor reuse
 * @param size Required buffer size
 * @return Reference to the next buffer, resized if needed
 */
std::vector<float>& SDRInferenceEngine::getNextPingPongBuffer(size_t size) {
    // Alternate between two buffers to reduce allocations
    current_buffer_idx_ = 1 - current_buffer_idx_;
    auto& buffer = ping_pong_buffers_[current_buffer_idx_];
    
    if (buffer.size() < size) {
        buffer.resize(size);
    }
    
    return buffer;
}

/**
 * @brief Fixed memory pool allocation with free list management
 * @param size Number of floats to allocate
 * @return Pointer to allocated memory in pool
 */
float* SDRInferenceEngine::allocateFromPool(size_t size) {
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    
    // First, try to find a suitable free block using first-fit strategy
    for (auto& block : free_list_) {
        if (block.is_free && block.size >= size) {
            // Found a suitable block
            block.is_free = false;
            
            // If block is significantly larger, split it
            if (block.size > size + 64) {  // 64-element minimum split threshold
                size_t remaining = block.size - size;
                block.size = size;
                
                // Add remaining part back to free list
                free_list_.emplace_back(block.offset + size, remaining, true);
            }
            
            return memory_pool_.data() + block.offset;
        }
    }
    
    // No suitable free block found, allocate from end of pool
    if (memory_pool_offset_ + size > memory_pool_.size()) {
        // Pool exhausted, expand it
        size_t new_size = memory_pool_.size() + std::max(size, memory_pool_.size() / 2);
        if (new_size * sizeof(float) > max_memory_usage_) {
            // Exceeded memory limit - try to compact free list first
            compactFreeList();
            
            // Check again after compaction
            if (memory_pool_offset_ + size > memory_pool_.size()) {
                // Still need more space
                new_size = std::min(new_size, max_memory_usage_ / sizeof(float));
                if (memory_pool_offset_ + size > new_size) {
                    std::cerr << "[SDRInferenceEngine] ERROR: Memory pool exhausted. "
                              << "Requested: " << size * sizeof(float) / (1024*1024) << "MB, "
                              << "Available: " << (new_size - memory_pool_offset_) * sizeof(float) / (1024*1024) << "MB"
                              << std::endl;
                    return nullptr;
                }
            }
        }
        memory_pool_.resize(new_size, 0.0f);
    }
    
    float* ptr = memory_pool_.data() + memory_pool_offset_;
    
    // Add to free list as allocated block
    free_list_.emplace_back(memory_pool_offset_, size, false);
    
    memory_pool_offset_ += size;
    
    return ptr;
}

/**
 * @brief Enable or disable layer prefetching
 * @param enable True to enable prefetching
 */
void SDRInferenceEngine::enableLayerPrefetch(bool enable) {
    enable_prefetch_ = enable;
}

/**
 * @brief Start prefetching next layer asynchronously
 * @param layer_name Name of layer to prefetch
 */
void SDRInferenceEngine::prefetchNextLayer(const std::string& layer_name) const {
    if (!enable_prefetch_) return;
    
    prefetch_layer_name_ = layer_name;
    prefetch_future_ = loader_.loadLayerByNameAsync(layer_name);
}

/**
 * @brief Get prefetched layer or load synchronously if not ready
 * @param layer_name Name of layer to get
 * @return Loaded layer info
 */
LayerInfo SDRInferenceEngine::getPrefetchedLayer(const std::string& layer_name) const {
    if (enable_prefetch_ && layer_name == prefetch_layer_name_ && prefetch_future_.valid()) {
        return prefetch_future_.get();
    }
    
    return loader_.loadLayerByName(layer_name);
}

/**
 * @brief Get performance information about optimizations
 * @return String describing enabled optimizations
 */
std::string SDRInferenceEngine::getPerformanceInfo() const {
    std::ostringstream oss;
    oss << "Performance Optimizations:\n";
    oss << "  BLAS: " << CortexAICompression::Kernels::get_blas_implementation() << "\n";
    oss << "  SIMD: " << CortexAICompression::Kernels::get_simd_level() << "\n";
    oss << "  Prefetch: " << (enable_prefetch_ ? "Enabled" : "Disabled") << "\n";
    oss << "  Ping-pong buffers: Enabled\n";
    oss << "  Memory pool: " << (memory_pool_.size() * sizeof(float) / (1024*1024)) << " MB\n";
    oss << "  Free blocks: " << std::count_if(free_list_.begin(), free_list_.end(),
                                               [](const MemoryBlock& b) { return b.is_free; }) << "\n";
    return oss.str();
}

/**
 * @brief Coalesce adjacent free blocks in the free list
 * 
 * Merges contiguous free blocks to reduce fragmentation and improve
 * allocation efficiency. Called automatically after deallocation.
 */
void SDRInferenceEngine::coalesceFreeBlocks() {
    // Sort free list by offset
    std::sort(free_list_.begin(), free_list_.end(),
              [](const MemoryBlock& a, const MemoryBlock& b) {
                  return a.offset < b.offset;
              });
    
    // Merge adjacent free blocks
    for (size_t i = 0; i < free_list_.size(); ) {
        if (!free_list_[i].is_free) {
            ++i;
            continue;
        }
        
        // Look for adjacent free blocks
        size_t j = i + 1;
        while (j < free_list_.size() && 
               free_list_[j].is_free &&
               free_list_[i].offset + free_list_[i].size == free_list_[j].offset) {
            // Merge block j into block i
            free_list_[i].size += free_list_[j].size;
            free_list_.erase(free_list_.begin() + j);
        }
        
        ++i;
    }
}

/**
 * @brief Compact the free list and defragment memory pool
 * 
 * Removes empty entries and attempts to consolidate allocated memory
 * to reclaim space at the end of the pool. This is an expensive operation
 * and should only be called when memory pressure is high.
 */
void SDRInferenceEngine::compactFreeList() {
    coalesceFreeBlocks();
    
    // Remove blocks that are beyond current offset
    free_list_.erase(
        std::remove_if(free_list_.begin(), free_list_.end(),
                      [this](const MemoryBlock& block) {
                          return block.offset >= memory_pool_offset_;
                      }),
        free_list_.end()
    );
    
    // If there's a large free block at the end, we can reclaim it
    if (!free_list_.empty()) {
        auto& last_block = free_list_.back();
        if (last_block.is_free && 
            last_block.offset + last_block.size == memory_pool_offset_) {
            // Reclaim space
            memory_pool_offset_ = last_block.offset;
            free_list_.pop_back();
        }
    }
}

} // namespace CortexAICompression
