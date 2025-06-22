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
    loadFromArchive(archive_path);
}

// Helper to fill in LayerInfo properties from SegmentInfo and previous layer
static void fillLayerInfoFromSegment(const SegmentInfo& seg, const std::unordered_map<std::string, std::vector<size_t>>& tensor_shapes, LayerInfo& layer) {
    // The layer name should be the base name, without .weights or .bias
    // This logic will be handled in the calling loop.
    // layer.name is assumed to be set correctly before calling.
    
    // Use the true layer type from the segment
    if (layer.layer_type.empty()) {
        layer.layer_type = seg.layer_type;
    }

    const std::vector<size_t>* shape = &seg.shape;
    auto it_shape = tensor_shapes.find(seg.name);
    if (it_shape != tensor_shapes.end()) {
        shape = &it_shape->second;
    }

    if (!shape->empty()) {
        if (layer.input_shape.empty()) {
             layer.input_shape = *shape;
        }
        layer.output_shape = *shape;
    }

    size_t num_elements = 1;
    for (size_t d : *shape) num_elements *= d;

    // Check if the segment is weights or biases based on name suffix
    if (seg.name.find(".bias") != std::string::npos) {
        if (!layer.raw_data.empty()) {
            if (layer.raw_data.size() == num_elements * 2) {
                // Stored in FP16
                layer.biases.resize(num_elements);
                const uint16_t* src = reinterpret_cast<const uint16_t*>(layer.raw_data.data());
                for (size_t i = 0; i < num_elements; ++i) {
                    uint16_t h = src[i];
                    uint32_t sign = (h & 0x8000) << 16;
                    uint32_t exp  = (h & 0x7C00) >> 10;
                    uint32_t mant = (h & 0x03FF);
                    uint32_t f;
                    if (exp == 0) {
                        if (mant == 0) f = sign;
                        else {
                            exp = 1;
                            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
                            mant &= 0x03FF;
                            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                        }
                    } else if (exp == 0x1F) {
                        f = sign | 0x7F800000 | (mant << 13);
                    } else {
                        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                    }
                    layer.biases[i] = *reinterpret_cast<float*>(&f);
                }
            } else {
                // Detect possible endianness mismatch for FP32 biases
                layer.biases.resize(num_elements);
                const uint8_t* srcBytes = reinterpret_cast<const uint8_t*>(layer.raw_data.data());
                float probeLittle;
                std::memcpy(&probeLittle, srcBytes, 4);
                uint32_t tmp;
                std::memcpy(&tmp, srcBytes, 4);
                uint32_t swappedTmp = ((tmp & 0x000000FFU) << 24) |
                                       ((tmp & 0x0000FF00U) << 8)  |
                                       ((tmp & 0x00FF0000U) >> 8)  |
                                       ((tmp & 0xFF000000U) >> 24);
                float probeSwap;
                std::memcpy(&probeSwap, &swappedTmp, 4);
                bool needSwap = (std::fabs(probeLittle) < 1e-35f && std::fabs(probeSwap) > 1e-6f && std::fabs(probeSwap) < 1e4f);
                if (!needSwap) {
                    std::memcpy(layer.biases.data(), layer.raw_data.data(), num_elements * sizeof(float));
                } else {
                    for (size_t i = 0; i < num_elements; ++i) {
                        uint32_t word;
                        std::memcpy(&word, srcBytes + i * 4, 4);
                        uint32_t swapped = ((word & 0x000000FFU) << 24) |
                                           ((word & 0x0000FF00U) << 8)  |
                                           ((word & 0x00FF0000U) >> 8)  |
                                           ((word & 0xFF000000U) >> 24);
                        std::memcpy(&layer.biases[i], &swapped, 4);
                    }
                }
            }
        }
    } else { // Assume it's weights if not bias
        if (!layer.raw_data.empty()) {
            // Determine encoding by comparing byte-size
            if (layer.raw_data.size() == num_elements * 4) {
                // FP32 tensor but check for endianness mismatch
                layer.weights.resize(num_elements);
                const uint8_t* srcBytes = reinterpret_cast<const uint8_t*>(layer.raw_data.data());
                float probeLittle;
                std::memcpy(&probeLittle, srcBytes, 4);
                uint32_t tmp;
                std::memcpy(&tmp, srcBytes, 4);
                uint32_t swappedTmp = ((tmp & 0x000000FFU) << 24) |
                                       ((tmp & 0x0000FF00U) << 8)  |
                                       ((tmp & 0x00FF0000U) >> 8)  |
                                       ((tmp & 0xFF000000U) >> 24);
                float probeSwap;
                std::memcpy(&probeSwap, &swappedTmp, 4);
                bool needSwap = (std::fabs(probeLittle) < 1e-35f && std::fabs(probeSwap) > 1e-6f && std::fabs(probeSwap) < 1e4f);
                if (!needSwap) {
                    std::memcpy(layer.weights.data(), layer.raw_data.data(), num_elements * sizeof(float));
                } else {
                    for (size_t i = 0; i < num_elements; ++i) {
                        uint32_t word;
                        std::memcpy(&word, srcBytes + i * 4, 4);
                        uint32_t swapped = ((word & 0x000000FFU) << 24) |
                                           ((word & 0x0000FF00U) << 8)  |
                                           ((word & 0x00FF0000U) >> 8)  |
                                           ((word & 0xFF000000U) >> 24);
                        std::memcpy(&layer.weights[i], &swapped, 4);
                    }
                }
            } else if (layer.raw_data.size() == num_elements * 2) {
                // Promote FP16 -> FP32
                const uint16_t* src = reinterpret_cast<const uint16_t*>(layer.raw_data.data());
                layer.weights.resize(num_elements);
                for (size_t i = 0; i < num_elements; ++i) {
                    uint16_t h = src[i];
                    uint32_t sign = (h & 0x8000) << 16;
                    uint32_t exp  = (h & 0x7C00) >> 10;
                    uint32_t mant = (h & 0x03FF);
                    uint32_t f;
                    if (exp == 0) {
                        if (mant == 0) {
                            f = sign; // zero
                        } else {
                            exp = 1;
                            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
                            mant &= 0x03FF;
                            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                        }
                    } else if (exp == 0x1F) {
                        f = sign | 0x7F800000 | (mant << 13);
                    } else {
                        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                    }
                    layer.weights[i] = *reinterpret_cast<float*>(&f);
                }
            }
        }
    }

    // Propagate true input/output shapes from ModelSegment if present
    if (!seg.input_shape.empty()) {
        layer.input_shape = seg.input_shape;
    }
    if (!seg.output_shape.empty()) {
        layer.output_shape = seg.output_shape;
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
        // Read name length as uint32_t
        uint32_t name_len;
        infile.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        if (name_len > 10000) { // Basic sanity check
            std::cerr << "[SDRModelLoader] Invalid name length: " << name_len << std::endl;
            return;
        }
        seg.name.resize(name_len);
        infile.read(&seg.name[0], name_len);

        // Read layer_type as a length-prefixed string (uint32_t + string)
        uint32_t type_len = 0;
        infile.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
        if (type_len > 0 && type_len < 1024) { // Sanity check
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
        
        const size_t MAX_SEGMENT_SIZE = 1024 * 1024 * 1024; // 1 GB
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
            if (dims > 8) { // Sanity check for dimensions
                std::cerr << "[SDRModelLoader] Invalid number of dimensions: " << (int)dims << std::endl;
                return;
            }
            seg.shape.resize(dims);
            for (uint8_t d = 0; d < dims; ++d) {
                uint32_t dim_size;
                infile.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
                if (dim_size > MAX_SEGMENT_SIZE) { // Sanity check for dimension size
                    std::cerr << "[SDRModelLoader] Invalid dimension size: " << dim_size << std::endl;
                    return;
                }
                seg.shape[d] = dim_size;
            }
            tensor_shapes[seg.name] = seg.shape;
        } else {
            seg.shape.clear();
        }
        // Read input_shape
        uint8_t has_input_shape = 0;
        infile.read(reinterpret_cast<char*>(&has_input_shape), sizeof(has_input_shape));
        if (has_input_shape) {
            uint8_t num_in = 0;
            infile.read(reinterpret_cast<char*>(&num_in), sizeof(num_in));
            seg.input_shape.resize(num_in);
            for (uint8_t d = 0; d < num_in; ++d) {
                uint32_t dim = 0;
                infile.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                seg.input_shape[d] = dim;
            }
        }
        // Read output_shape
        uint8_t has_output_shape = 0;
        infile.read(reinterpret_cast<char*>(&has_output_shape), sizeof(has_output_shape));
        if (has_output_shape) {
            uint8_t num_out = 0;
            infile.read(reinterpret_cast<char*>(&num_out), sizeof(num_out));
            seg.output_shape.resize(num_out);
            for (uint8_t d = 0; d < num_out; ++d) {
                uint32_t dim = 0;
                infile.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                seg.output_shape[d] = dim;
            }
        }
        segments_.push_back(seg);
        // --- NEW: Check for model_structure segment and parse as ONNX ---
#ifdef ENABLE_ONNX_PROTOBUF
        // Preserve the GRAPH_STRUCTURE_PROTO segment but **skip** expensive parsing.
        if (seg.type == SegmentType::GRAPH_STRUCTURE_PROTO && seg.name == "model_structure") {
            // std::ifstream data_infile(archive_path, std::ios::binary);
            // data_infile.seekg(seg.offset, std::ios::beg);
            // std::vector<std::byte> compressed_buffer(seg.compressed_size);
            // data_infile.read(reinterpret_cast<char*>(compressed_buffer.data()), seg.compressed_size);
            // std::vector<std::byte> decompressed_bytes;
            // if (seg.strategy_id == 0) {
            //     decompressed_bytes = compressed_buffer;
            // } else if (seg.strategy_id == 1) {
            //     decompressed_bytes = decompressSDR(compressed_buffer, seg.original_size);
            // } else {
            //     std::cerr << "[SDRModelLoader] Unknown compression strategy for model_structure segment: " << (int)seg.strategy_id << std::endl;
            //     continue;
            // }
            // if (decompressed_bytes.empty()) {
            //     std::cerr << "[SDRModelLoader] Failed to decompress model_structure segment." << std::endl;
            // } else {
            //     std::string model_str(reinterpret_cast<const char*>(decompressed_bytes.data()), decompressed_bytes.size());
            //     onnx::ModelProto proto;
            //     if (proto.ParseFromString(model_str)) {
            //         loaded_model_proto_ = proto;
            //         std::cout << "[SDRModelLoader] Loaded ONNX ModelProto from model_structure segment." << std::endl;
            //     } else {
            //         std::cerr << "[SDRModelLoader] Failed to parse ONNX ModelProto from model_structure segment after decompression." << std::endl;
            //     }
            // }
            std::cout << "[SDRModelLoader] Graph structure segment detected (" << seg.original_size
                      << " bytes). Skipping ONNX parse – not required for inference." << std::endl;
            continue; // Nothing else to do for this segment
        }
#endif
    }
        // Logic to group segments by layer and create LayerInfo objects
        layers.clear();
        layer_map_.clear();
        std::unordered_map<std::string, LayerInfo> layer_map;
        std::ifstream data_infile(archive_path, std::ios::binary);

        for (const auto& seg : segments_) {
            try {
                // Only process valid weight tensors
                if (seg.type != SegmentType::WEIGHTS_FP32 && seg.type != SegmentType::WEIGHTS_FP16 && seg.type != SegmentType::WEIGHTS_INT8) {
                    continue;
                }
                
                // Extract base layer name (e.g., "conv1.weight" -> "conv1")
                std::string base_name = seg.name;
                size_t pos = base_name.rfind('.');
                if (pos != std::string::npos) {
                    base_name = base_name.substr(0, pos);
                }

                // Find or create the LayerInfo for this base name
                LayerInfo& layer = layer_map[base_name];
                if (layer.name.empty()) {
                    layer.name = base_name;
                }
                
                // Decompress data for the current segment
                data_infile.seekg(seg.offset);
                std::vector<std::byte> compressed_data(seg.compressed_size);
                data_infile.read(reinterpret_cast<char*>(compressed_data.data()), seg.compressed_size);
                std::vector<std::byte> decompressed_data;
                if (seg.strategy_id == 0) {
                    decompressed_data = compressed_data;
                } else if (seg.strategy_id == 1) { // Assuming strategy 1 is SDR
                    decompressed_data = decompressSDR(compressed_data, seg.original_size);
                }
                
                layer.raw_data = std::move(decompressed_data);
                
                // Fill layer info from this segment
                fillLayerInfoFromSegment(seg, tensor_shapes, layer);

            } catch (const std::bad_alloc& e) {
                std::cerr << "Failed to process segment " << seg.name 
                          << ": Memory allocation failed" << std::endl;
                throw;
            }
        }
        
        // Move the grouped layers from the map to the final vector
        layers.reserve(layer_map.size());
        for (const auto& kv : layer_map) {
            layers.push_back(kv.second);
            layer_map_[kv.first] = kv.second;
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
    fillLayerInfoFromSegment(seg, tensor_shapes, layer);
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
    // Populate layer_map_ for fast lookup
    for (const auto& layer : layers) {
        layer_map_[layer.name] = layer;
    }
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
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < input_size; ++j) {
            output[i] += input[j] * layer.weights[i * input_size + j];
        }
        if (!layer.biases.empty()) {
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
    if (layers.empty()) {
        std::cerr << "[SDRInferenceEngine] No layers loaded!" << std::endl;
        return input_tensor;
    }

    // Static dispatch map from ONNX op type to member function
    using LayerFn = std::function<std::vector<float>(SDRInferenceEngine*, const LayerInfo&, const std::vector<float>&)>;
    static const std::unordered_map<std::string, LayerFn> dispatch = {
        {"Conv",    [](SDRInferenceEngine* self, const LayerInfo& l, const std::vector<float>& in) { return self->applyConvolutionalLayer(l, in); }},
        {"MatMul",  [](SDRInferenceEngine* self, const LayerInfo& l, const std::vector<float>& in) { return self->applyLinearLayer(l, in); }},
        {"Gemm",    [](SDRInferenceEngine* self, const LayerInfo& l, const std::vector<float>& in) { return self->applyLinearLayer(l, in); }},
        // Add more mappings as needed
    };

    std::vector<float> current = input_tensor;
    
    for (const auto& layer : layers) {
        std::cout << "[SDRInferenceEngine] Processing Layer: " << layer.name 
                  << ", Type: " << layer.layer_type
                  << ", Input Shape: " << current.size()
                  << ", Weights: " << layer.weights.size() 
                  << ", Biases: " << layer.biases.size() << std::endl;

        auto it = dispatch.find(layer.layer_type);
        if (it != dispatch.end()) {
            current = it->second(this, layer, current);
        } else if (layer.layer_type == "Add") {
            std::cout << "[SDRInferenceEngine] Skipping 'Add' layer, assuming bias is handled." << std::endl;
        } else {
            std::cerr << "[SDRInferenceEngine] Unknown or unhandled layer type: " << layer.layer_type 
                      << " for layer " << layer.name << ". Skipping.\n";
        }
    }

    return current;
}

#ifdef ENABLE_ONNX_PROTOBUF
const std::optional<onnx::ModelProto>& SDRModelLoader::getLoadedModelProto() const {
    return loaded_model_proto_;
}
#endif

// Run a single layer by name
std::vector<float> SDRInferenceEngine::runLayer(const std::string& layer_name, const std::vector<float>& input) {
    auto it = layer_map_.find(layer_name);
    if (it == layer_map_.end()) {
        std::cerr << "[SDRInferenceEngine] Layer not found: " << layer_name << std::endl;
        return input;
    }
    const LayerInfo& layer = it->second;
    // Strict shape validation
    size_t expected_input_size = 1;
    for (size_t d : layer.input_shape) expected_input_size *= d;
    if (input.size() != expected_input_size) {
        std::cerr << "[SDRInferenceEngine] ERROR: Input tensor size " << input.size()
                  << " does not match expected input_shape product " << expected_input_size
                  << " for layer: " << layer.name << std::endl;
        return {};
    }
    if (layer.layer_type == "Conv") {
        return applyConvolutionalLayer(layer, input);
    } else if (layer.layer_type == "MatMul" || layer.layer_type == "Gemm" || layer.layer_type == "LINEAR") {
        return applyLinearLayer(layer, input);
    } else if (layer.layer_type == "BATCH_NORM") {
        return applyBatchNorm(layer, input);
    } else if (layer.layer_type == "ACTIVATION") {
        return applyActivation(layer.properties.activation_type, input);
    } else {
        std::cerr << "[SDRInferenceEngine] Unknown or unhandled layer type: " << layer.layer_type << " for layer " << layer.name << ". Skipping." << std::endl;
        return input;
    }
}

// Run a sequence of layers by name
std::vector<float> SDRInferenceEngine::runLayers(const std::vector<std::string>& layer_names, const std::vector<float>& input) {
    std::vector<float> current = input;
    for (const auto& name : layer_names) {
        current = runLayer(name, current);
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