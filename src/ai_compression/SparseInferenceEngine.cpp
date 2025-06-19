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
    loadFromArchive(archive_path);
}

// Helper to fill in LayerInfo properties from SegmentInfo and previous layer
static void fillLayerInfoFromSegment(const SegmentInfo& seg, const std::unordered_map<std::string, std::vector<size_t>>& tensor_shapes, const std::vector<LayerInfo>& prev_layers, LayerInfo& layer) {
    layer.name = seg.name;
    // Only set shape/type if metadata is present
    auto it_shape = tensor_shapes.find(seg.name);
    const std::vector<size_t>* shape = (it_shape != tensor_shapes.end()) ? &it_shape->second : &seg.shape;
    if (shape && !shape->empty()) {
        layer.input_shape = *shape; // Just record the shape as-is
        // Optionally, set output_shape if metadata provides it
        layer.output_shape = *shape;
    } else {
        layer.input_shape.clear();
        layer.output_shape.clear();
    }
    // Use the true layer type from the segment (set by the parser, as a string)
    layer.layer_type = seg.layer_type;
    // --- Decode weights or biases ---
    size_t num_elements = 1;
    for (size_t d : *shape) num_elements *= d;
    if (seg.name.find("bias") != std::string::npos) {
        // Bias segment
        if (seg.type == SegmentType::WEIGHTS_FP32 && layer.raw_data.size() == num_elements * sizeof(float)) {
            layer.biases.resize(num_elements);
            std::memcpy(layer.biases.data(), layer.raw_data.data(), num_elements * sizeof(float));
        }
    } else {
        // Weight segment
        if (seg.type == SegmentType::WEIGHTS_FP32 && layer.raw_data.size() == num_elements * sizeof(float)) {
            layer.weights.resize(num_elements);
            std::memcpy(layer.weights.data(), layer.raw_data.data(), num_elements * sizeof(float));
        }
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

    // Read archive header
    char magic[8];
    infile.read(magic, sizeof(magic));
        if (std::memcmp(magic, ARCHIVE_MAGIC, sizeof(ARCHIVE_MAGIC)) != 0) {
        std::cerr << "[SDRModelLoader] Invalid archive format" << std::endl;
        return;
    }

    uint32_t version;
    infile.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != ARCHIVE_VERSION) {
        std::cerr << "[SDRModelLoader] Unsupported archive version" << std::endl;
        return;
    }

    // Read number of segments
    uint32_t num_segments;
    infile.read(reinterpret_cast<char*>(&num_segments), sizeof(num_segments));

        if (num_segments > 1000) {  // Sanity check
            std::cerr << "[SDRModelLoader] Invalid number of segments: " << num_segments << std::endl;
            return;
        }
        
        std::cout << "Indexing " << num_segments << " segments from archive..." << std::endl;

        segments_.clear();
        segments_.reserve(num_segments);
    std::unordered_map<std::string, std::vector<size_t>> tensor_shapes;
    
    for (uint32_t i = 0; i < num_segments; ++i) {
        SegmentInfo seg;
            // Read name length
        uint16_t name_len;
        infile.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            if (name_len > 10000) {
                std::cerr << "[SDRModelLoader] Invalid name length: " << name_len << std::endl;
                return;
            }
        seg.name.resize(name_len);
        infile.read(&seg.name[0], name_len);
            // Read layer_type as a length-prefixed string (uint16_t + string)
            uint16_t type_len = 0;
            infile.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            if (type_len > 0 && type_len < 1024) {
                seg.layer_type.resize(type_len);
                infile.read(&seg.layer_type[0], type_len);
            } else {
                seg.layer_type.clear();
            }
        uint8_t type_val;
        infile.read(reinterpret_cast<char*>(&type_val), sizeof(type_val));
        seg.type = static_cast<SegmentType>(type_val);
        infile.read(reinterpret_cast<char*>(&seg.strategy_id), sizeof(seg.strategy_id));
            uint64_t original_size, compressed_size;
            infile.read(reinterpret_cast<char*>(&original_size), sizeof(original_size));
            infile.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));
            const size_t MAX_SEGMENT_SIZE = 1024 * 1024 * 1024;
            if (original_size > MAX_SEGMENT_SIZE || compressed_size > MAX_SEGMENT_SIZE) {
                std::cerr << "[SDRModelLoader] Invalid segment size for " << seg.name 
                          << ": original=" << original_size 
                          << ", compressed=" << compressed_size << std::endl;
                return;
            }
            seg.original_size = original_size;
            seg.compressed_size = compressed_size;
        infile.read(reinterpret_cast<char*>(&seg.offset), sizeof(seg.offset));
            uint8_t has_metadata = 0;
            infile.read(reinterpret_cast<char*>(&has_metadata), sizeof(has_metadata));
            if (has_metadata) {
            uint8_t dims;
            infile.read(reinterpret_cast<char*>(&dims), sizeof(dims));
                if (dims > 8) {
                    std::cerr << "[SDRModelLoader] Invalid number of dimensions: " << (int)dims << std::endl;
                    return;
                }
            seg.shape.resize(dims);
            for (uint8_t d = 0; d < dims; ++d) {
                uint32_t dim_size;
                infile.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
                    if (dim_size > MAX_SEGMENT_SIZE) {
                        std::cerr << "[SDRModelLoader] Invalid dimension size: " << dim_size << std::endl;
                        return;
                    }
                seg.shape[d] = dim_size;
            }
            tensor_shapes[seg.name] = seg.shape;
            } else {
                seg.shape.clear();
            }
            segments_.push_back(seg);
    }
        // Legacy: load all layers (not memory efficient)
        layers.clear();
        layers.reserve(segments_.size());
        std::ifstream data_infile(archive_path, std::ios::binary);
        for (const auto& seg : segments_) {
            try {
                // Only process valid weight tensors
                if ((seg.type == SegmentType::WEIGHTS_FP32 || seg.type == SegmentType::WEIGHTS_FP16 || seg.type == SegmentType::WEIGHTS_INT8) && (seg.shape.size() == 2 || seg.shape.size() == 4)) {
                    data_infile.seekg(seg.offset);
        std::vector<std::byte> compressed_data(seg.compressed_size);
                    data_infile.read(reinterpret_cast<char*>(compressed_data.data()), seg.compressed_size);
        std::vector<std::byte> decompressed_data;
                    if (seg.strategy_id == 0) {
            decompressed_data = compressed_data;
                    } else if (seg.strategy_id == 1) {
            decompressed_data = decompressSDR(compressed_data, seg.original_size);
        }
                    LayerInfo layer;
                    layer.raw_data = std::move(decompressed_data);
                    fillLayerInfoFromSegment(seg, tensor_shapes, layers, layer);
                    layers.push_back(std::move(layer));
                }
            } catch (const std::bad_alloc& e) {
                std::cerr << "Failed to process segment " << seg.name 
                          << ": Memory allocation failed" << std::endl;
                throw;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading compressed model: " << e.what() << std::endl;
        throw;
    }
}

LayerInfo SDRModelLoader::loadLayerByName(const std::string& name) const {
    auto it = std::find_if(segments_.begin(), segments_.end(), [&](const SegmentInfo& seg) { return seg.name == name; });
    if (it == segments_.end()) {
        throw std::runtime_error("Layer not found: " + name);
    }
    const SegmentInfo& seg = *it;
    // Only allow valid weight tensors for on-demand loading
    if (!((seg.type == SegmentType::WEIGHTS_FP32 || seg.type == SegmentType::WEIGHTS_FP16 || seg.type == SegmentType::WEIGHTS_INT8) && (seg.shape.size() == 2 || seg.shape.size() == 4))) {
        throw std::runtime_error("Requested segment is not a valid weight tensor: " + name);
            }
    std::ifstream infile(archive_path_, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Failed to open archive for on-demand layer loading: " + archive_path_);
    }
    infile.seekg(seg.offset);
    std::vector<std::byte> compressed_data(seg.compressed_size);
    infile.read(reinterpret_cast<char*>(compressed_data.data()), seg.compressed_size);
    std::vector<std::byte> decompressed_data;
    if (seg.strategy_id == 0) {
        decompressed_data = compressed_data;
    } else if (seg.strategy_id == 1) {
        decompressed_data = decompressSDR(compressed_data, seg.original_size);
    }
    LayerInfo layer;
    layer.raw_data = std::move(decompressed_data);
    std::unordered_map<std::string, std::vector<size_t>> tensor_shapes;
    tensor_shapes[seg.name] = seg.shape;
    fillLayerInfoFromSegment(seg, tensor_shapes, {}, layer);
    return layer;
}

// Helper function to decompress SDR data
std::vector<std::byte> SDRModelLoader::decompressSDR(const std::vector<std::byte>& compressed_data, size_t original_size) const {
    try {
        // Pre-allocate with reserve to avoid reallocation
        std::vector<std::byte> decompressed_data;
        decompressed_data.reserve(original_size);
        decompressed_data.resize(original_size);

        // Extract indices from compressed data
        auto indices = decode_varint_indices(compressed_data);
        
        // Clear the buffer first
    std::fill(decompressed_data.begin(), decompressed_data.end(), std::byte{0});
    
        // Set active bits
        for (size_t idx : indices) {
            if (idx < original_size) {
                decompressed_data[idx] = std::byte{1};
        }
    }

    return decompressed_data;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate memory for decompression. Required size: " 
                  << original_size << " bytes" << std::endl;
        throw;
    }
}

const std::vector<LayerInfo>& SDRModelLoader::getLayers() const {
    return layers;
}

SDRInferenceEngine::SDRInferenceEngine(const SDRModelLoader& model)
    : layers(model.getLayers()), batch_size(1), dropout_enabled(false), training_mode(false) {
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
    if (layer.weights.empty()) {
        std::cerr << "[SDRInferenceEngine] No weights for layer: " << layer.name << std::endl;
        return input;
    }
    // Generalized batch handling: if input.size() is a multiple of input_size, treat as batch
    size_t input_size = layer.input_shape.empty() ? 0 : layer.input_shape.back();
    size_t output_size = layer.output_shape.empty() ? 0 : layer.output_shape.back();
    if (input_size == 0 || output_size == 0) {
        std::cerr << "[SDRInferenceEngine] Invalid input/output size for layer: " << layer.name << std::endl;
        return input;
    }
    size_t batch_size = input.size() / input_size;
    if (input.size() % input_size != 0) {
        std::cerr << "[SDRInferenceEngine] Input size " << input.size() << " is not a multiple of input_size " << input_size << " for layer: " << layer.name << std::endl;
        return input;
    }
    std::vector<float> output(batch_size * output_size, 0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
        const float* in_ptr = input.data() + b * input_size;
        float* out_ptr = output.data() + b * output_size;
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < input_size; ++j) {
                out_ptr[i] += in_ptr[j] * layer.weights[i * input_size + j];
        }
        if (!layer.biases.empty()) {
                out_ptr[i] += layer.biases[i];
            }
        }
    }
    return output;
}

std::vector<float> SDRInferenceEngine::applyConvolutionalLayer(const LayerInfo& layer, const std::vector<float>& input) {
    if (layer.weights.empty()) {
        std::cerr << "[SDRInferenceEngine] No weights for layer: " << layer.name << std::endl;
        return input;
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
    if (layers.empty()) {
        std::cerr << "[SDRInferenceEngine] No layers loaded!" << std::endl;
        return input_tensor;
    }

    std::vector<float> current = input_tensor;
    
    for (const auto& layer : layers) {
        bool handled = false;
        std::cerr << "[SDRInferenceEngine] Layer: " << layer.name << ", input_shape: ";
        for (auto d : layer.input_shape) std::cerr << d << " ";
        std::cerr << ", output_shape: ";
        for (auto d : layer.output_shape) std::cerr << d << " ";
        std::cerr << ", weights: " << layer.weights.size() << ", biases: " << layer.biases.size() << std::endl;
        // Heuristic: try to infer operation from shape/properties, not type string
        if (!layer.weights.empty() && layer.input_shape.size() == 2 && layer.output_shape.size() == 2) {
            // Looks like a linear layer (by shape)
            current = applyLinearLayer(layer, current);
            handled = true;
        } else if (!layer.weights.empty() && layer.input_shape.size() == 4 && layer.output_shape.size() == 4 && !layer.properties.kernel_shape.empty()) {
            // Looks like a conv layer (by shape)
            current = applyConvolutionalLayer(layer, current);
            handled = true;
        } else if (layer.properties.use_batch_norm) {
            current = applyBatchNorm(layer, current);
            handled = true;
        } else if (!layer.properties.activation_type.empty()) {
            current = applyActivation(layer.properties.activation_type, current);
            handled = true;
        } else if (!layer.properties.kernel_shape.empty() && layer.input_shape.size() == 4) {
            // Could be pooling
            current = applyMaxPool(current, layer.input_shape);
            handled = true;
        }
        // Add more heuristics as needed

        if (!handled) {
            std::cerr << "[SDRInferenceEngine] Unknown or unhandled layer: " << layer.name << " (type: " << layer.layer_type << "). Skipping.\n";
            // Pass-through: current = current;
        }
        // Dropout, if present
        if (dropout_enabled && layer.properties.dropout_rate > 0.0f) {
            current = applyDropout(layer, current);
        }
    }

    return current;
}

} // namespace CortexAICompression 