/**
 * @file CoreMLModelParser.cpp
 * @brief Implementation of CoreML model parsing into archive segments.
 */
#include "CoreMLModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <zip.h>

namespace CortexAICompression {

CoreMLModelParser::CoreMLModelParser() {
    // Constructor implementation
}

CoreMLModelParser::~CoreMLModelParser() {
    // Destructor implementation
}

SegmentType CoreMLModelParser::coremlDataTypeToSegmentType(const std::string& dataType) const {
    if (dataType == "float32" || dataType == "Float32") {
        return SegmentType::WEIGHTS_FP32;
    } else if (dataType == "float16" || dataType == "Float16") {
        return SegmentType::WEIGHTS_FP16;
    } else if (dataType == "int8" || dataType == "Int8") {
        return SegmentType::WEIGHTS_INT8;
    } else if (dataType == "int4" || dataType == "Int4") {
        return SegmentType::WEIGHTS_INT4;
    } else {
        return SegmentType::UNKNOWN;
    }
}

TensorMetadata CoreMLModelParser::extractTensorMetadata(const CoreMLLayerInfo& layerInfo, const std::vector<int64_t>& shape) const {
    TensorMetadata metadata;
    
    // Convert shape to our format
    for (int64_t dim : shape) {
        metadata.dimensions.push_back(static_cast<size_t>(dim));
    }
    
    // Calculate sparsity (placeholder - would need to analyze actual data)
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = false;
    
    return metadata;
}

std::string CoreMLModelParser::extractLayerName(const std::string& layerName) const {
    // CoreML layer names often follow patterns like:
    // "conv1", "dense_2", "layer_3"
    std::regex layer_pattern(R"(([^_]+))");
    std::smatch matches;
    if (std::regex_search(layerName, matches, layer_pattern)) {
        return matches[1].str();
    }
    return layerName;
}

size_t CoreMLModelParser::extractLayerIndex(const std::string& layerName) const {
    // Extract numeric index from layer names like "conv1", "dense_2", "layer_3"
    std::regex index_pattern(R"((\d+))");
    std::smatch matches;
    if (std::regex_search(layerName, matches, index_pattern)) {
        try {
            return std::stoul(matches[1].str());
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid layer index in layer name: " << layerName << std::endl;
        }
    }
    return 0;
}

std::vector<std::byte> CoreMLModelParser::readModelData(const std::string& modelPath) const {
    std::vector<std::byte> data;
    
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw ParsingError("Failed to open CoreML model file: " + modelPath);
    }
    
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    data.resize(file_size);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    return data;
}

std::vector<CoreMLModelParser::CoreMLLayerInfo> CoreMLModelParser::extractLayerInfo(const std::string& modelPath) const {
    std::vector<CoreMLLayerInfo> layers;
    
    try {
        // CoreML models are typically ZIP archives containing protobuf files
        // For now, we'll create a basic parser that extracts the model structure
        
        // Read the model file
        auto model_data = readModelData(modelPath);
        
        // Check if it's a ZIP file (CoreML models are typically ZIP archives)
        if (model_data.size() >= 4) {
            // Check ZIP magic number
            if (model_data[0] == std::byte{0x50} && model_data[1] == std::byte{0x4B} &&
                model_data[2] == std::byte{0x03} && model_data[3] == std::byte{0x04}) {
                
                
                // For now, create a placeholder layer structure
                // In a full implementation, you would:
                // 1. Extract the ZIP contents
                // 2. Parse the protobuf files
                // 3. Extract layer information and weights
                
                CoreMLLayerInfo placeholder_layer;
                placeholder_layer.name = "placeholder_layer";
                placeholder_layer.layer_type = "Unknown";
                placeholder_layer.input_shape = {1, 3, 224, 224}; // Common input shape
                placeholder_layer.output_shape = {1, 1000}; // Common output shape
                placeholder_layer.data_type = "float32";
                
                // Create placeholder weight data
                size_t weight_size = 1000 * 3 * 224 * 224 * sizeof(float); // Example size
                placeholder_layer.weights_data.resize(weight_size, std::byte{0});
                
                layers.push_back(std::move(placeholder_layer));
                
                
            } else {
                // Not a ZIP file, might be a raw protobuf
                
                // Create a basic layer structure
                CoreMLLayerInfo basic_layer;
                basic_layer.name = "basic_layer";
                basic_layer.layer_type = "Unknown";
                basic_layer.input_shape = {1, 3, 224, 224};
                basic_layer.output_shape = {1, 1000};
                basic_layer.data_type = "float32";
                basic_layer.weights_data = model_data;
                
                layers.push_back(std::move(basic_layer));
            }
        } else {
            throw ParsingError("CoreML model file is too small or corrupted");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting CoreML layer info: " << e.what() << std::endl;
        throw ParsingError("Failed to extract CoreML layer information: " + std::string(e.what()));
    }
    
    return layers;
}

ModelSegment CoreMLModelParser::createSegmentFromLayer(const CoreMLLayerInfo& layerInfo, const std::string& segmentName, const std::vector<std::byte>& data) const {
    ModelSegment segment;
    segment.name = segmentName;
    segment.type = coremlDataTypeToSegmentType(layerInfo.data_type);
    segment.data = data;
    segment.original_size = data.size();
    segment.layer_name = extractLayerName(layerInfo.name);
    segment.layer_index = extractLayerIndex(layerInfo.name);
    segment.layer_type = layerInfo.layer_type;
    
    // Set input/output shapes
    for (int64_t dim : layerInfo.input_shape) {
        segment.input_shape.push_back(static_cast<size_t>(dim));
    }
    for (int64_t dim : layerInfo.output_shape) {
        segment.output_shape.push_back(static_cast<size_t>(dim));
    }
    
    // Create tensor metadata
    std::vector<int64_t> shape = layerInfo.input_shape;
    if (segmentName.find("output") != std::string::npos) {
        shape = layerInfo.output_shape;
    }
    segment.tensor_metadata = extractTensorMetadata(layerInfo, shape);
    
    return segment;
}

std::vector<ModelSegment> CoreMLModelParser::parse(const std::string& modelPath) const {
    std::vector<ModelSegment> segments;
    
    try {
        auto layerInfos = extractLayerInfo(modelPath);
        segments.reserve(layerInfos.size() * 2); // Assume weights and biases per layer
        
        for (const auto& layerInfo : layerInfos) {
            // Create weight segment
            if (!layerInfo.weights_data.empty()) {
                segments.push_back(createSegmentFromLayer(layerInfo, layerInfo.name + ".weights", layerInfo.weights_data));
            }
            
            // Create bias segment
            if (!layerInfo.bias_data.empty()) {
                segments.push_back(createSegmentFromLayer(layerInfo, layerInfo.name + ".bias", layerInfo.bias_data));
            }
        }
        
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing CoreML model: " << e.what() << std::endl;
        throw ParsingError("Failed to parse CoreML model: " + std::string(e.what()));
    }
    
    return segments;
}

std::vector<ModelSegment> CoreMLModelParser::parseWithChunking(const std::string& modelPath) const {
    auto segments = parse(modelPath);
    
    // Group segments by layer for better compression
    std::map<size_t, std::vector<ModelSegment*>> layerGroups;
    for (auto& segment : segments) {
        layerGroups[segment.layer_index].push_back(&segment);
    }
    
    // Sort segments within each layer by type
    for (auto& [layer, group] : layerGroups) {
        std::sort(group.begin(), group.end(),
                 [](const ModelSegment* a, const ModelSegment* b) {
                     return static_cast<int>(a->type) < static_cast<int>(b->type);
                 });
    }
    
    // Reorder segments for optimal compression
    std::vector<ModelSegment> reorderedSegments;
    reorderedSegments.reserve(segments.size());
    
    // First, add non-layer segments (global params, etc.)
    for (const auto& segment : segments) {
        if (segment.layer_index == 0 && segment.layer_name.empty()) {
            reorderedSegments.push_back(segment);
        }
    }
    
    // Then add layer segments in order
    for (const auto& [layer, group] : layerGroups) {
        if (layer > 0 || !group.front()->layer_name.empty()) {
            for (const auto* segment : group) {
                reorderedSegments.push_back(*segment);
            }
        }
    }
    
    return reorderedSegments;
}

} // namespace CortexAICompression
