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
#include "kernels/sparse_kernels.hpp"
#include "utils/fp16_convert.hpp"
#include "utils/kv_cache.hpp"
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

/**
 * @brief Decode varint-encoded indices from SDR compressed data.
 * @param data Raw compressed byte buffer containing varint-encoded indices.
 * @return Vector of decoded sparse indices (0-based positions).
 *
 * @details Decodes a sequence of unsigned varints packed consecutively.
 * Each varint uses the MSB as a continuation bit. This routine is used to
 * reconstruct sparse position lists for compressed tensors.
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
void SDRModelLoader::parseLayerMetadata(const std::string& metadata, LayerInfo& layer) {
    std::istringstream iss(metadata);
    std::string key, value;
    
    while (iss >> key >> value) {
        if (key == "type") {
            if (value == "conv")
                layer.layer_type = "CONV2D";
            else
                layer.layer_type = "LINEAR";
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
        }
    }
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

    // Extract tensor metadata if available
    if (model_segment.tensor_metadata) {
        const auto& meta = model_segment.tensor_metadata.value();
        if (layer.input_shape.empty() && !meta.dimensions.empty()) {
             layer.input_shape = meta.dimensions;
        }
        layer.output_shape = meta.dimensions;
    }

    layer.raw_data = model_segment.data;

    // Calculate element count based on data type
    size_t num_elements = 0;
    size_t element_size = 4; // Default: float32

    if(model_segment.type == SegmentType::WEIGHTS_FP16) element_size = 2;
    if(model_segment.type == SegmentType::WEIGHTS_INT8) element_size = 1;

    if (model_segment.original_size > 0) {
        num_elements = model_segment.original_size / element_size;
    }

    // Distinguish between weights and biases by name convention
    if (model_segment.name.find(".bias") != std::string::npos) {
        if (!layer.raw_data.empty()) {
            layer.biases.resize(num_elements);
             if (model_segment.type == SegmentType::WEIGHTS_FP16) {
                // Convert FP16 to FP32
                const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(layer.raw_data.data());
                for (size_t i = 0; i < num_elements; ++i) {
                    layer.biases[i] = fp16_to_fp32(fp16_data[i]);
                }
            } else {
                std::memcpy(layer.biases.data(), layer.raw_data.data(), std::min(layer.raw_data.size(), layer.biases.size() * sizeof(float)));
            }
        }
    } else {
        if (!layer.raw_data.empty()) {
            layer.weights.resize(num_elements);
            if (model_segment.type == SegmentType::WEIGHTS_FP16) {
                // Convert FP16 to FP32
                const uint16_t* fp16_data = reinterpret_cast<const uint16_t*>(layer.raw_data.data());
                for (size_t i = 0; i < num_elements; ++i) {
                    layer.weights[i] = fp16_to_fp32(fp16_data[i]);
                }
            } else {
                std::memcpy(layer.weights.data(), layer.raw_data.data(), std::min(layer.raw_data.size(), layer.weights.size() * sizeof(float)));
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
            std::cerr << "[SDRModelLoader] Failed to open archive: " << archive_path << std::endl;
            return;
        }

        // Read only headers for on-demand loading efficiency
        segments_ = decompressor_->readArchiveHeaders(infile);

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
    auto it = layer_cache_.find(name);
    if (it != layer_cache_.end()) {
        return it->second;
    }

    std::packaged_task<LayerInfo()> task([this, name]() {
        auto seg_it = std::find_if(segments_.begin(), segments_.end(), 
            [&](const CompressedSegmentHeader& seg) { return seg.name == name; });
        if (seg_it == segments_.end()) {
            throw std::runtime_error("Segment info not found for layer: " + name);
        }
        const CompressedSegmentHeader& seg_info = *seg_it;

        // Calculate streaming format offset
        uint64_t offset = sizeof(ARCHIVE_MAGIC) + sizeof(ARCHIVE_VERSION) + sizeof(uint64_t) + sizeof(uint64_t);
        
        for (const auto& seg : segments_) {
            if (seg.name == name) break;
            offset += seg.compressed_size;
        }

        // For weight tensors, prefer keeping compressed bytes for streaming compute
        LayerInfo layer;
        layer.name = name;
        if (seg_info.original_type == SegmentType::WEIGHTS_FP32 ||
            seg_info.original_type == SegmentType::WEIGHTS_FP16 ||
            seg_info.original_type == SegmentType::WEIGHTS_INT8 ||
            seg_info.original_type == SegmentType::WEIGHTS_INT4 ||
            seg_info.original_type == SegmentType::ATTENTION_WEIGHTS ||
            seg_info.original_type == SegmentType::FEED_FORWARD_WEIGHTS ||
            seg_info.original_type == SegmentType::EMBEDDING_WEIGHTS ||
            seg_info.original_type == SegmentType::LAYER_NORM_WEIGHTS) {
            // Read compressed payload only; avoid dense inflation
            std::vector<std::byte> compressedPayload = decompressor_->readCompressedBytes(archive_path_, seg_info, offset);
            layer.raw_data = std::move(compressedPayload);
            // Still need shapes/metadata for execution
            ModelSegment headerOnly;
            headerOnly.name = seg_info.name;
            headerOnly.type = seg_info.original_type;
            headerOnly.original_size = seg_info.original_size;
            headerOnly.tensor_metadata = seg_info.tensor_metadata;
            headerOnly.layer_name = seg_info.layer_name;
            headerOnly.layer_index = seg_info.layer_index;
            headerOnly.layer_type = seg_info.layer_type;
            headerOnly.input_shape = seg_info.input_shape;
            headerOnly.output_shape = seg_info.output_shape;
            fillLayerInfoFromSegment(headerOnly, layer);
        } else {
            // Non-weight tensors or metadata: decompress fully
            ModelSegment model_segment = decompressor_->decompressSegment(archive_path_, seg_info, offset);
            fillLayerInfoFromSegment(model_segment, layer);
        }
        
        return layer;
    });

    std::shared_future<LayerInfo> future = task.get_future();
    layer_cache_[name] = future;
    
    std::thread(std::move(task)).detach();

    return future;
}

/**
 * @brief Clear a layer from cache to free memory.
 * @param name Layer name to remove from cache.
 */
void SDRModelLoader::clearLayerFromCache(const std::string& name) const {
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
    op_dispatch_["MatMul"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyLinearLayer(l, in); 
    };
    op_dispatch_["Gemm"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyLinearLayer(l, in); 
    };
    op_dispatch_["LINEAR"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyLinearLayer(l, in); 
    };
    op_dispatch_["BATCH_NORM"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyBatchNorm(l, in); 
    };
    op_dispatch_["ACTIVATION"] = [this](const LayerInfo& l, const std::vector<float>& in) { 
        return applyActivation(l.properties.activation_type, in); 
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
    // Validate shapes
    try {
        if (layer.input_shape.empty() || layer.output_shape.empty()) {
            throw Utils::TensorValidationError("Missing input/output shape for linear layer: " + layer.name);
        }
        
        size_t input_size = 1;
        for (size_t d : layer.input_shape) input_size *= d;
        size_t output_size = 1;
        for (size_t d : layer.output_shape) output_size *= d;
        
        Utils::TensorValidator::validate_size(input, layer.input_shape, layer.name + "_input");
        
        if (!layer.weights.empty()) {
            Utils::TensorValidator::validate_linear_weights(
                layer.weights.size(), input_size, output_size, layer.name
            );
        }
    } catch (const Utils::TensorValidationError& e) {
        std::cerr << "[SDRInferenceEngine] Validation error: " << e.what() << std::endl;
        return {};
    }
    
    size_t input_size = 1;
    for (size_t d : layer.input_shape) input_size *= d;
    size_t output_size = 1;
    for (size_t d : layer.output_shape) output_size *= d;
    if (force_compressed_compute_ || layer.weights.size() != input_size * output_size) {
        // OPTIMIZED: Use sparse kernels for zero-decompression inference
        std::vector<float> output(output_size, 0.0f);
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
            
            // Use optimized sparse linear forward kernel
            CortexAICompression::SparseKernels::sparse_linear_forward(
                indices,
                values,
                input.data(),
                layer.biases.empty() ? nullptr : layer.biases.data(),
                output.data(),
                input_size,
                output_size
            );
            
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
        std::cerr << "[SDRInferenceEngine] ERROR: Biases size " << layer.biases.size() << " does not match output " << output_size << " for linear layer: " << layer.name << std::endl;
        return {};
    }
    
    // Use ping-pong buffer to reduce allocations
    std::vector<float>& output = getNextPingPongBuffer(output_size);
    
    // Use BLAS-accelerated linear forward: output = input * weights^T + bias
    CortexAICompression::Kernels::linear_forward(
        input.data(),
        layer.weights.data(),
        layer.biases.empty() ? nullptr : layer.biases.data(),
        output.data(),
        batch_size,  // For single sample, batch_size = 1
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
    int batch = layer.input_shape[0];
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
    
    if (input.size() != static_cast<size_t>(batch * in_channels * in_height * in_width)) {
        std::cerr << "[SDRInferenceEngine] ERROR: Input size " << input.size() << " does not match expected " << (batch * in_channels * in_height * in_width) << " for convolutional layer: " << layer.name << std::endl;
        return {};
    }
    const size_t expected_w = out_channels * in_channels * kernel_h * kernel_w;
    if (force_compressed_compute_ || layer.weights.size() != expected_w) {
        // Streaming sparse convolution using compressed weights
        std::vector<float> output(batch_size * out_channels * out_height * out_width, 0.0f);
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
                for (size_t b = 0; b < batch_size; ++b) {
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
                for (size_t b = 0; b < batch_size; ++b) {
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
    std::vector<float> output(batch * out_channels * out_height * out_width, 0.0f);
    
    CortexAICompression::Kernels::conv2d_im2col(
        input.data(),
        layer.weights.data(),
        layer.biases.empty() ? nullptr : layer.biases.data(),
        output.data(),
        batch,
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

    std::vector<float> output = input;
    size_t channels = layer.input_shape[1];
    size_t spatial_size = input.size() / (batch_size * channels);

    for (size_t b = 0; b < batch_size; ++b) {
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

    size_t channels = layer.input_shape[1];
    size_t spatial_size = input.size() / (batch_size * channels);
    float momentum = 0.1f;

    for (size_t c = 0; c < channels; ++c) {
        float mean = 0.0f;
        float var = 0.0f;

        // Calculate batch mean
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < spatial_size; ++i) {
                size_t idx = b * channels * spatial_size + c * spatial_size + i;
                mean += input[idx];
            }
        }
        mean /= (batch_size * spatial_size);

        // Calculate batch variance
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < spatial_size; ++i) {
                size_t idx = b * channels * spatial_size + c * spatial_size + i;
                float diff = input[idx] - mean;
                var += diff * diff;
            }
        }
        var /= (batch_size * spatial_size);

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
        // Shape mismatch in reshape. Input size: " 
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
 * @brief Allocate tensor from memory pool
 * @param size Number of floats to allocate
 * @return Vector view into memory pool
 */
std::vector<float> SDRInferenceEngine::allocateFromPool(size_t size) {
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    
    if (memory_pool_offset_ + size > memory_pool_.size()) {
        // Pool exhausted, fall back to regular allocation
        return std::vector<float>(size, 0.0f);
    }
    
    size_t start_offset = memory_pool_offset_;
    memory_pool_offset_ += size;
    
    // Return a view into the memory pool
    return std::vector<float>(memory_pool_.begin() + start_offset, 
                             memory_pool_.begin() + start_offset + size);
}

/**
 * @brief Return memory to pool (simplified - just tracks usage)
 * @param size Number of floats to deallocate
 */
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
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
    
    if (lower_type.find("layer_norm") != std::string::npos ||
        lower_type.find("layernorm") != std::string::npos) {
        // Use SIMD-optimized LayerNorm
        std::vector<float>& output = getNextPingPongBuffer(input.size());
        
        if (!layer.properties.bn_weights.empty() && !layer.properties.bn_biases.empty()) {
            CortexAICompression::Kernels::layer_norm(
                input.data(),
                output.data(),
                layer.properties.bn_weights.data(),
                layer.properties.bn_biases.data(),
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
        activation_type = "relu"; // Default activation
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
    
    // Use existing pooling implementations
    if (layer.properties.activation_type == "max") {
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
    std::vector<std::string> layer_names;
    std::set<std::string> unique_layers;
    
    // Extract actual weight segment names directly
    for (const auto& seg : segments) {
        if (seg.original_type == SegmentType::WEIGHTS_FP32 ||
            seg.original_type == SegmentType::WEIGHTS_FP16 ||
            seg.original_type == SegmentType::WEIGHTS_INT8) {
            
            // Use the actual segment name, not a derived base name
            unique_layers.insert(seg.name);
        }
    }
    
    // Convert set to vector for consistent ordering
    for (const auto& layer_name : unique_layers) {
        layer_names.push_back(layer_name);
    }
    
    // Intelligent sorting for neural network layer names
    std::sort(layer_names.begin(), layer_names.end(), [](const std::string& a, const std::string& b) {
        auto extract_numbers = [](const std::string& str) -> std::vector<int> {
            std::vector<int> numbers;
            std::regex number_regex(R"(\d+)");
            std::sregex_iterator iter(str.begin(), str.end(), number_regex);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                numbers.push_back(std::stoi(iter->str()));
            }
            return numbers;
        };
        
        auto nums_a = extract_numbers(a);
        auto nums_b = extract_numbers(b);
        
        if (!nums_a.empty() && !nums_b.empty()) {
            for (size_t i = 0; i < std::min(nums_a.size(), nums_b.size()); ++i) {
                if (nums_a[i] != nums_b[i]) {
                    return nums_a[i] < nums_b[i];
                }
            }
            return nums_a.size() < nums_b.size();
        }
        
        return a < b;
    });
    
    return layer_names;
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
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
    
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