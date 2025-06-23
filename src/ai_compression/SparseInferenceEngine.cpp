#include "SparseInferenceEngine.hpp"
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

// Added for the new implementation
#include "strategies/AdaptiveSDRStrategy.hpp"
#include "strategies/GzipStrategy.hpp"
#include "strategies/NumericalRLE.hpp"
#include "strategies/QuantizedTensorStrategy.hpp"
#include "strategies/SDRIndexStorage.hpp"

#ifdef ENABLE_ONNX_PROTOBUF
#include <../onnx_proto/onnx.pb.h>
#endif

namespace CortexAICompression {

// Helper: decode varint-encoded indices from SDR data
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

SDRModelLoader::SDRModelLoader(const std::string& archive_path) : archive_path_(archive_path) {
    // Initialize the decompressor and register all potential strategies
    decompressor_ = std::make_unique<AIDecompressor>();
    
    // These IDs MUST match the ones used in c_api.cpp during compression
    const uint8_t SDR_STRATEGY_ID = 1;
    const uint8_t RLE_STRATEGY_ID = 2;
    const uint8_t GZIP_STRATEGY_ID = 3;
    const uint8_t QUANT_STRATEGY_ID = 4;
    
    // The AdaptiveSDRStrategy is the main one used for SDR compression.
    // Sparsity value is only needed for compression, so a default is fine for decompression.
    auto adaptiveStrategy = std::make_shared<AdaptiveSDRStrategy>(0.02f); 
    
    decompressor_->registerStrategy(SDR_STRATEGY_ID, adaptiveStrategy);
    decompressor_->registerStrategy(RLE_STRATEGY_ID, std::make_shared<NumericalRLEStrategy>());
    decompressor_->registerStrategy(GZIP_STRATEGY_ID, std::make_shared<GzipStrategy>());
#ifdef ENABLE_QUANTIZATION
    decompressor_->registerStrategy(QUANT_STRATEGY_ID, std::make_shared<QuantizedTensorStrategy>());
#endif

    // For backwards compatibility, also register the legacy SDR strategy under a different ID
    // if older files might be loaded.
    auto legacySdrStrategy = std::make_shared<SDRIndexStorageStrategy>();
    decompressor_->registerStrategy(SDR_STRATEGY_ID + 10, legacySdrStrategy);

    loadFromArchive(archive_path);

    // Eagerly populate layer_map_ with decompressed data for all segments
    for (const auto& seg : segments_) {
        // Only load weight/bias segments, skip metadata/structure
        if (seg.original_type == SegmentType::WEIGHTS_FP32 ||
            seg.original_type == SegmentType::WEIGHTS_FP16 ||
            seg.original_type == SegmentType::WEIGHTS_INT8) {
            try {
                LayerInfo layer = loadLayerByName(seg.name);
                layer_map_[seg.name] = std::move(layer);
            } catch (const std::exception& e) {
                std::cerr << "[SDRModelLoader] Failed to load layer '" << seg.name << "': " << e.what() << std::endl;
            }
        }
    }
}

// Helper to fill in LayerInfo properties from a decompressed ModelSegment
static void fillLayerInfoFromSegment(const ModelSegment& model_segment, LayerInfo& layer) {
    // Use the true layer type from the segment
    if (layer.layer_type.empty()) {
        layer.layer_type = model_segment.layer_type;
    }

    // Use metadata from the segment if available
    if (model_segment.tensor_metadata) {
        const auto& meta = model_segment.tensor_metadata.value();
        if (layer.input_shape.empty() && !meta.dimensions.empty()) {
             layer.input_shape = meta.dimensions;
        }
        layer.output_shape = meta.dimensions;
    }

    layer.raw_data = model_segment.data;

    size_t num_elements = 0;
    size_t element_size = 4; // Default to float32

    if(model_segment.type == SegmentType::WEIGHTS_FP16) element_size = 2;
    if(model_segment.type == SegmentType::WEIGHTS_INT8) element_size = 1;

    if (model_segment.original_size > 0) {
        num_elements = model_segment.original_size / element_size;
    }

    // Check if the segment is weights or biases based on name suffix
    if (model_segment.name.find(".bias") != std::string::npos) {
        if (!layer.raw_data.empty()) {
            layer.biases.resize(num_elements);
             if (model_segment.type == SegmentType::WEIGHTS_FP16) {
                // TODO: Handle FP16 to FP32 conversion if necessary for the engine
            }
            std::memcpy(layer.biases.data(), layer.raw_data.data(), layer.raw_data.size());
        }
    } else { // Assume it's weights if not bias
        if (!layer.raw_data.empty()) {
            layer.weights.resize(num_elements);
            if (model_segment.type == SegmentType::WEIGHTS_FP16) {
                // TODO: Handle FP16 to FP32 conversion if necessary for the engine
            }
            std::memcpy(layer.weights.data(), layer.raw_data.data(), layer.raw_data.size());
        }
    }

    // Propagate true input/output shapes from ModelSegment if present
    if (!model_segment.input_shape.empty()) {
        layer.input_shape = model_segment.input_shape;
    }
    if (!model_segment.output_shape.empty()) {
        layer.output_shape = model_segment.output_shape;
    }
}

// Example SDR archive format:
// [uint16_t name_len][char name[]][uint32_t data_size][byte data[]] ...
void SDRModelLoader::loadFromArchive(const std::string& archive_path) {
    try {
        std::ifstream infile(archive_path, std::ios::binary);
        if (!infile) {
            std::cerr << "[SDRModelLoader] Failed to open archive: " << archive_path << std::endl;
            return;
        }

        // For on-demand loading, we only need to read the headers.
        segments_ = decompressor_->readArchiveHeaders(infile);

    } catch (const std::exception& e) {
        std::cerr << "Error loading compressed model headers: " << e.what() << std::endl;
        throw;
    }
}

LayerInfo SDRModelLoader::loadLayerByName(const std::string& name) const {
    return loadLayerByNameAsync(name).get();
}

std::shared_future<LayerInfo> SDRModelLoader::loadLayerByNameAsync(const std::string& name) const {
    // Check cache first
    auto it = layer_cache_.find(name);
    if (it != layer_cache_.end()) {
        return it->second;
    }

    // If not in cache, launch a new async task to load it
    std::packaged_task<LayerInfo()> task([this, name]() {
        // Find the segment info for the requested layer/tensor name
        auto seg_it = std::find_if(segments_.begin(), segments_.end(), [&](const CompressedSegmentHeader& seg) { return seg.name == name; });
        if (seg_it == segments_.end()) {
            throw std::runtime_error("Segment info not found for layer: " + name);
        }
        const CompressedSegmentHeader& seg_info = *seg_it;

        // Calculate the offset for streaming format
        // In streaming format: header (magic + version + placeholder count + placeholder offset) + data blocks
        uint64_t offset = sizeof(ARCHIVE_MAGIC) + sizeof(ARCHIVE_VERSION) + sizeof(uint64_t) + sizeof(uint64_t);
        
        // Add up the compressed sizes of all segments that come before this one
        for (const auto& seg : segments_) {
            if (seg.name == name) break;
            offset += seg.compressed_size;
        }

        // Use the decompressor to load this specific segment
        ModelSegment model_segment = decompressor_->decompressSegment(archive_path_, seg_info, offset);

        LayerInfo layer;
        layer.name = name;
        fillLayerInfoFromSegment(model_segment, layer);
        return layer;
    });

    std::shared_future<LayerInfo> future = task.get_future();
    layer_cache_[name] = future;
    
    // Move the task to a detached thread to execute
    std::thread(std::move(task)).detach();

    return future;
}

// Helper function to decompress SDR data
std::vector<std::byte> SDRModelLoader::decompressSDR(const std::vector<std::byte>& compressed_data, size_t original_size) const {
    // This method is now deprecated and should not be called.
    // Decompression is handled by the AIDecompressor framework.
    throw std::logic_error("SDRModelLoader::decompressSDR is deprecated.");
}

const std::vector<LayerInfo>& SDRModelLoader::getLayers() const {
    // This method is now less useful as layers are loaded on demand.
    // It will only return layers loaded by the initial (legacy) full load.
    return layers;
}

const std::vector<CompressedSegmentHeader>& SDRModelLoader::getSegmentIndex() const {
    return segments_;
}

SDRInferenceEngine::SDRInferenceEngine(SDRModelLoader& model_loader)
    : loader_(model_loader), batch_size(1), dropout_enabled(false), training_mode(false) {
    // Register op handlers
    op_dispatch_["Conv"] = [this](const LayerInfo& l, const std::vector<float>& in) { return applyConvolutionalLayer(l, in); };
    op_dispatch_["MatMul"] = [this](const LayerInfo& l, const std::vector<float>& in) { return applyLinearLayer(l, in); };
    op_dispatch_["Gemm"] = [this](const LayerInfo& l, const std::vector<float>& in) { return applyLinearLayer(l, in); };
    op_dispatch_["LINEAR"] = [this](const LayerInfo& l, const std::vector<float>& in) { return applyLinearLayer(l, in); };
    op_dispatch_["BATCH_NORM"] = [this](const LayerInfo& l, const std::vector<float>& in) { return applyBatchNorm(l, in); };
    op_dispatch_["ACTIVATION"] = [this](const LayerInfo& l, const std::vector<float>& in) { return applyActivation(l.properties.activation_type, in); };
    // Default handler: log and pass through
    default_handler_ = [](const LayerInfo& layer, const std::vector<float>& input) {
        std::cerr << "[SDRInferenceEngine] Unhandled layer type: '" << layer.layer_type << "' for layer '" << layer.name << "'. Passing input through." << std::endl;
        return input;
    };
}

void SDRInferenceEngine::setBatchSize(size_t size) {
    batch_size = size;
}

void SDRInferenceEngine::enableDropout(bool enable) {
    dropout_enabled = enable;
}

void SDRInferenceEngine::setInferenceMode(bool training) {
    training_mode = training;
}

std::vector<float> SDRInferenceEngine::applyLinearLayer(const LayerInfo& layer, const std::vector<float>& input) {
    // Validate input and layer shapes
    if (layer.input_shape.empty() || layer.output_shape.empty()) {
        std::cerr << "[SDRInferenceEngine] ERROR: Missing input/output shape for linear layer: " << layer.name << std::endl;
        return {};
    }
    size_t input_size = 1;
    for (size_t d : layer.input_shape) input_size *= d;
    size_t output_size = 1;
    for (size_t d : layer.output_shape) output_size *= d;
    if (input.size() != input_size) {
        std::cerr << "[SDRInferenceEngine] ERROR: Input size " << input.size() << " does not match expected " << input_size << " for linear layer: " << layer.name << std::endl;
        return {};
    }
    if (layer.weights.size() != input_size * output_size) {
        std::cerr << "[SDRInferenceEngine] ERROR: Weights size " << layer.weights.size() << " does not match input*output " << input_size * output_size << " for linear layer: " << layer.name << std::endl;
        return {};
    }
    if (!layer.biases.empty() && layer.biases.size() != output_size) {
        std::cerr << "[SDRInferenceEngine] ERROR: Biases size " << layer.biases.size() << " does not match output " << output_size << " for linear layer: " << layer.name << std::endl;
        return {};
    }
    std::vector<float> output(output_size, 0.0f);
    const int BLOCK_SIZE = 32;

    for (size_t i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (size_t j_block = 0; j_block < input_size; j_block += BLOCK_SIZE) {
            size_t j_end = std::min(j_block + BLOCK_SIZE, input_size);
            for (size_t j = j_block; j < j_end; ++j) {
                sum += input[j] * layer.weights[i * input_size + j];
            }
        }
        output[i] = sum;
    }

    if (!layer.biases.empty()) {
        for (size_t i = 0; i < output_size; ++i) {
            output[i] += layer.biases[i];
        }
    }
    return output;
}

std::vector<float> SDRInferenceEngine::applyConvolutionalLayer(const LayerInfo& layer, const std::vector<float>& input) {
    // Validate input and layer shapes
    if (layer.input_shape.size() != 4 || layer.output_shape.size() != 4 ||
        layer.properties.kernel_shape.size() != 2 || layer.properties.strides.size() != 2 || layer.properties.padding.size() != 2) {
        std::cerr << "[SDRInferenceEngine] ERROR: Invalid or missing shape/params for convolutional layer: " << layer.name << std::endl;
        return {};
    }
    size_t batch_size = layer.input_shape[0];
    size_t in_channels = layer.input_shape[1];
    size_t in_height = layer.input_shape[2];
    size_t in_width = layer.input_shape[3];
    size_t out_channels = layer.output_shape[1];
    size_t kernel_h = layer.properties.kernel_shape[0];
    size_t kernel_w = layer.properties.kernel_shape[1];
    size_t stride_h = layer.properties.strides[0];
    size_t stride_w = layer.properties.strides[1];
    size_t pad_h = layer.properties.padding[0];
    size_t pad_w = layer.properties.padding[1];
    size_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    if (input.size() != batch_size * in_channels * in_height * in_width) {
        std::cerr << "[SDRInferenceEngine] ERROR: Input size " << input.size() << " does not match expected " << (batch_size * in_channels * in_height * in_width) << " for convolutional layer: " << layer.name << std::endl;
        return {};
    }
    if (layer.weights.size() != out_channels * in_channels * kernel_h * kernel_w) {
        std::cerr << "[SDRInferenceEngine] ERROR: Weights size " << layer.weights.size() << " does not match expected " << (out_channels * in_channels * kernel_h * kernel_w) << " for convolutional layer: " << layer.name << std::endl;
        return {};
    }
    if (!layer.biases.empty() && layer.biases.size() != out_channels) {
        std::cerr << "[SDRInferenceEngine] ERROR: Biases size " << layer.biases.size() << " does not match out_channels " << out_channels << " for convolutional layer: " << layer.name << std::endl;
        return {};
    }
    std::vector<float> output(batch_size * out_channels * out_height * out_width, 0.0f);
    // Convolution operation with im2col optimization
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_h; ++kh) {
                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * stride_h + kh - pad_h;
                                int iw = ow * stride_w + kw - pad_w;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    size_t input_idx = b * in_channels * in_height * in_width +
                                                     ic * in_height * in_width +
                                                     ih * in_width + iw;
                                    size_t weight_idx = oc * in_channels * kernel_h * kernel_w +
                                                      ic * kernel_h * kernel_w +
                                                      kh * kernel_w + kw;
                                    sum += input[input_idx] * layer.weights[weight_idx];
                                }
                            }
                        }
                    }
                    if (!layer.biases.empty()) {
                        sum += layer.biases[oc];
                    }
                    size_t output_idx = b * out_channels * out_height * out_width +
                                      oc * out_height * out_width +
                                      oh * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
    return output;
}

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

std::vector<float> SDRInferenceEngine::applyActivation(const std::string& type, const std::vector<float>& input) {
    std::vector<float> output = input;

    if (type == "relu") {
        for (float& val : output) {
            val = std::max(0.0f, val);
        }
    } else if (type == "leaky_relu") {
        for (float& val : output) {
            val = val > 0 ? val : 0.01f * val;
        }
    } else if (type == "sigmoid") {
        for (float& val : output) {
            val = 1.0f / (1.0f + std::exp(-val));
        }
    } else if (type == "tanh") {
        for (float& val : output) {
            val = std::tanh(val);
        }
    }

    return output;
}

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

void SDRInferenceEngine::updateBatchNormStats(const LayerInfo& layer, const std::vector<float>& input) {
    if (!layer.properties.use_batch_norm) return;

    size_t channels = layer.input_shape[1];
    size_t spatial_size = input.size() / (batch_size * channels);
    float momentum = 0.1f;

    for (size_t c = 0; c < channels; ++c) {
        float mean = 0.0f;
        float var = 0.0f;

        // Calculate mean
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < spatial_size; ++i) {
                size_t idx = b * channels * spatial_size + c * spatial_size + i;
                mean += input[idx];
            }
        }
        mean /= (batch_size * spatial_size);

        // Calculate variance
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < spatial_size; ++i) {
                size_t idx = b * channels * spatial_size + c * spatial_size + i;
                float diff = input[idx] - mean;
                var += diff * diff;
            }
        }
        var /= (batch_size * spatial_size);

        // Update running statistics
        layer.properties.bn_running_mean[c] = (1 - momentum) * layer.properties.bn_running_mean[c] + momentum * mean;
        layer.properties.bn_running_var[c] = (1 - momentum) * layer.properties.bn_running_var[c] + momentum * var;
    }
}

std::vector<float> SDRInferenceEngine::reshapeTensor(const std::vector<float>& input, const std::vector<size_t>& shape) {
    return input;  // For now, just return the input as is
}

std::vector<float> SDRInferenceEngine::flattenTensor(const std::vector<float>& input) {
    return input;  // For now, just return the input as is
}

std::vector<float> SDRInferenceEngine::run(const std::vector<float>& input_tensor) {
    const auto& segments = loader_.getSegmentIndex();
    if (segments.empty()) {
        std::cerr << "[SDRInferenceEngine] No segments found in the model loader!" << std::endl;
        return input_tensor;
    }

    // Heuristic: derive execution order by sorting segment names that look like layers.
    std::vector<std::string> layer_names;
    for (const auto& seg : segments) {
        if (seg.original_type == SegmentType::WEIGHTS_FP32 ||
            seg.original_type == SegmentType::WEIGHTS_FP16 ||
            seg.original_type == SegmentType::WEIGHTS_INT8) {
            // Get base name
            std::string base_name = seg.name;
            size_t pos = base_name.rfind('.');
            if (pos != std::string::npos) {
                base_name = base_name.substr(0, pos);
            }
            // Avoid duplicates
            if (std::find(layer_names.begin(), layer_names.end(), base_name) == layer_names.end()) {
                layer_names.push_back(base_name);
            }
        }
    }
    // A simple alphanumeric sort often works for transformer models (e.g., h.0, h.1, h.10, h.11...)
    std::sort(layer_names.begin(), layer_names.end());

    std::cout << "[SDRInferenceEngine] Running layers in heuristically determined order..." << std::endl;
    for(const auto& name : layer_names) {
        std::cout << "  -> " << name << std::endl;
    }

    return runLayers(layer_names, input_tensor);
}

#ifdef ENABLE_ONNX_PROTOBUF
const std::optional<onnx::ModelProto>& SDRModelLoader::getLoadedModelProto() const {
    return loaded_model_proto_;
}
#endif

// Run a single layer by name
std::vector<float> SDRInferenceEngine::runLayer(const LayerInfo& layer, const std::vector<float>& input) {
    // Strict shape validation
    size_t expected_input_size = 1;
    for (size_t d : layer.input_shape) expected_input_size *= d;
    if (input.size() != expected_input_size) {
        std::cerr << "[SDRInferenceEngine] ERROR: Input tensor size " << input.size()
                  << " does not match expected input_shape product " << expected_input_size
                  << " for layer: " << layer.name << std::endl;
        return {};
    }
    auto it = op_dispatch_.find(layer.layer_type);
    if (it != op_dispatch_.end()) {
        return it->second(layer, input);
    } else {
        return default_handler_(layer, input);
    }
}

// Run a sequence of layers by name
std::vector<float> SDRInferenceEngine::runLayers(const std::vector<std::string>& layer_names, const std::vector<float>& input) {
    std::vector<float> current = input;
    if (layer_names.empty()) return current;

    // Start loading the first layer
    auto next_layer_future = loader_.loadLayerByNameAsync(layer_names[0]);

    for (size_t i = 0; i < layer_names.size(); ++i) {
        // Wait for the current layer to be loaded
        LayerInfo current_layer = next_layer_future.get();

        // Start pre-fetching the next layer (if it exists)
        if (i + 1 < layer_names.size()) {
            next_layer_future = loader_.loadLayerByNameAsync(layer_names[i + 1]);
        }
        
        // Run computation for the current layer
        current = runLayer(current_layer, current);
        if (current.empty()) {
            std::cerr << "[SDRInferenceEngine] ERROR: Aborting runLayers due to previous error or shape mismatch." << std::endl;
            break;
        }
    }
    return current;
}

// Suggest all possible layer chains (pairs) based on output/input shape matching
void print_possible_layer_chains(const std::vector<LayerInfo>& layers) {
    std::cout << "\n[Heuristic Layer Chaining] Possible layer chains (output_shape == input_shape):" << std::endl;
    bool found = false;
    for (const auto& l1 : layers) {
        for (const auto& l2 : layers) {
            if (l1.name == l2.name) continue;
            if (l1.output_shape == l2.input_shape && !l1.output_shape.empty()) {
                std::cout << "  " << l1.name << " (" << l1.layer_type << ") -> "
                          << l2.name << " (" << l2.layer_type << ")" << std::endl;
                found = true;
            }
        }
    }
    if (!found) {
        std::cout << "  [None found]" << std::endl;
    }
}

} // namespace CortexAICompression 