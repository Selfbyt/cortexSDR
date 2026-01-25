/**
 * @file TensorFlowModelParser.cpp
 * @brief Implementation of TensorFlow SavedModel parsing into archive segments.
 */
#include "TensorFlowModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>

#ifdef ENABLE_TENSORFLOW
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/util/command_line_flags.h>
#endif

namespace CortexAICompression {

TensorFlowModelParser::TensorFlowModelParser() {
#ifdef ENABLE_TENSORFLOW
    // Initialize TensorFlow
    tensorflow::port::InitMain("", 0, nullptr);
#endif
}

TensorFlowModelParser::~TensorFlowModelParser() {
    // Destructor implementation
}

#ifdef ENABLE_TENSORFLOW

SegmentType TensorFlowModelParser::tensorflowDataTypeToSegmentType(tensorflow::DataType tf_type) const {
    switch (tf_type) {
        case tensorflow::DT_FLOAT:
            return SegmentType::WEIGHTS_FP32;
        case tensorflow::DT_HALF:
            return SegmentType::WEIGHTS_FP16;
        case tensorflow::DT_INT8:
            return SegmentType::WEIGHTS_INT8;
        case tensorflow::DT_INT32:
            return SegmentType::WEIGHTS_FP32; // Placeholder
        case tensorflow::DT_INT64:
            return SegmentType::WEIGHTS_FP32; // Placeholder
        case tensorflow::DT_UINT8:
            return SegmentType::WEIGHTS_INT8; // Placeholder
        default:
            std::cerr << "Warning: Unknown TensorFlow data type (" << tf_type << ") encountered." << std::endl;
            return SegmentType::UNKNOWN;
    }
}

TensorMetadata TensorFlowModelParser::extractTensorMetadata(const TFVariableInfo& varInfo) const {
    TensorMetadata metadata;
    
    // Convert TensorFlow shape to our format
    for (int64_t dim : varInfo.shape) {
        metadata.dimensions.push_back(static_cast<size_t>(dim));
    }
    
    // Calculate sparsity (placeholder - would need to analyze actual data)
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = false;
    
    return metadata;
}

std::string TensorFlowModelParser::extractLayerName(const std::string& variableName) const {
    // TensorFlow variable names often follow patterns like:
    // "layer_1/weights", "dense_2/bias", "conv2d_3/kernel"
    std::regex layer_pattern(R"(([^/]+)/)");
    std::smatch matches;
    if (std::regex_search(variableName, matches, layer_pattern)) {
        return matches[1].str();
    }
    return variableName;
}

size_t TensorFlowModelParser::extractLayerIndex(const std::string& variableName) const {
    // Extract numeric index from layer names like "layer_1", "dense_2", "conv2d_3"
    std::regex index_pattern(R"((\d+))");
    std::smatch matches;
    if (std::regex_search(variableName, matches, index_pattern)) {
        try {
            return std::stoul(matches[1].str());
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid layer index in variable name: " << variableName << std::endl;
        }
    }
    return 0;
}

std::vector<TensorFlowModelParser::TFVariableInfo> TensorFlowModelParser::extractVariableInfo(const std::string& modelPath) const {
    std::vector<TFVariableInfo> variables;
    
    try {
        // Load the SavedModel
        tensorflow::SavedModelBundle bundle;
        tensorflow::SessionOptions session_options;
        tensorflow::RunOptions run_options;
        
        auto status = tensorflow::LoadSavedModel(
            session_options, run_options, modelPath, 
            {tensorflow::kSavedModelTagServe}, &bundle);
        
        if (!status.ok()) {
            throw ParsingError("Failed to load TensorFlow SavedModel: " + status.ToString());
        }
        
        
        // Get all variables from the session
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = bundle.session->Run(
            {}, {}, {"tf.trainable_variables()"}, &outputs);
        
        if (!run_status.ok()) {
            // Fallback: try to extract variables from the graph
            const auto& graph_def = bundle.meta_graph_def.graph_def();
            
            for (const auto& node : graph_def.node()) {
                if (node.op() == "Variable" || node.op() == "VariableV2") {
                    TFVariableInfo varInfo;
                    varInfo.name = node.name();
                    
                    // Extract shape from node attributes
                    if (node.attr().count("shape")) {
                        const auto& shape_attr = node.attr().at("shape");
                        if (shape_attr.has_shape()) {
                            const auto& shape = shape_attr.shape();
                            for (const auto& dim : shape.dim()) {
                                varInfo.shape.push_back(dim.size());
                            }
                        }
                    }
                    
                    // Extract data type
                    if (node.attr().count("dtype")) {
                        varInfo.data_type = node.attr().at("dtype").type();
                    } else {
                        varInfo.data_type = tensorflow::DT_FLOAT; // Default
                    }
                    
                    // Calculate size
                    size_t element_size = 4; // Default for float32
                    if (varInfo.data_type == tensorflow::DT_HALF) element_size = 2;
                    if (varInfo.data_type == tensorflow::DT_INT8) element_size = 1;
                    
                    size_t total_elements = 1;
                    for (int64_t dim : varInfo.shape) {
                        total_elements *= static_cast<size_t>(dim);
                    }
                    varInfo.size_bytes = total_elements * element_size;
                    
                    variables.push_back(std::move(varInfo));
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting TensorFlow variable info: " << e.what() << std::endl;
        throw ParsingError("Failed to extract TensorFlow variable information: " + std::string(e.what()));
    }
    
    return variables;
}

std::vector<std::byte> TensorFlowModelParser::readVariableData(const std::string& modelPath, const TFVariableInfo& varInfo) const {
    std::vector<std::byte> data;
    
    try {
        // Load the SavedModel
        tensorflow::SavedModelBundle bundle;
        tensorflow::SessionOptions session_options;
        tensorflow::RunOptions run_options;
        
        auto status = tensorflow::LoadSavedModel(
            session_options, run_options, modelPath, 
            {tensorflow::kSavedModelTagServe}, &bundle);
        
        if (!status.ok()) {
            throw ParsingError("Failed to load TensorFlow SavedModel for data extraction: " + status.ToString());
        }
        
        // Try to read the variable data
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = bundle.session->Run(
            {}, {varInfo.name}, {}, &outputs);
        
        if (run_status.ok() && !outputs.empty()) {
            const auto& tensor = outputs[0];
            const auto& tensor_data = tensor.tensor_data();
            
            data.resize(tensor_data.size());
            std::memcpy(data.data(), tensor_data.data(), tensor_data.size());
        } else {
            // Fallback: create placeholder data
            data.resize(varInfo.size_bytes, std::byte{0});
            std::cerr << "Warning: Could not read actual data for variable " << varInfo.name 
                      << ", using placeholder data" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error reading TensorFlow variable data: " << e.what() << std::endl;
        // Return placeholder data
        data.resize(varInfo.size_bytes, std::byte{0});
    }
    
    return data;
}

ModelSegment TensorFlowModelParser::createSegmentFromVariable(const TFVariableInfo& varInfo, const std::vector<std::byte>& data) const {
    ModelSegment segment;
    segment.name = varInfo.name;
    segment.type = tensorflowDataTypeToSegmentType(varInfo.data_type);
    segment.data = data;
    segment.original_size = data.size();
    segment.tensor_metadata = extractTensorMetadata(varInfo);
    segment.layer_name = extractLayerName(varInfo.name);
    segment.layer_index = extractLayerIndex(varInfo.name);
    
    // Determine layer type based on variable name
    if (varInfo.name.find("kernel") != std::string::npos || varInfo.name.find("weight") != std::string::npos) {
        segment.layer_type = "WEIGHTS";
    } else if (varInfo.name.find("bias") != std::string::npos) {
        segment.layer_type = "BIAS";
    } else if (varInfo.name.find("gamma") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_GAMMA";
    } else if (varInfo.name.find("beta") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_BETA";
    } else {
        segment.layer_type = "UNKNOWN";
    }
    
    return segment;
}

#endif // ENABLE_TENSORFLOW

std::vector<ModelSegment> TensorFlowModelParser::parse(const std::string& modelPath) const {
    std::vector<ModelSegment> segments;
    
#ifdef ENABLE_TENSORFLOW
    try {
        
        auto variableInfos = extractVariableInfo(modelPath);
        segments.reserve(variableInfos.size());
        
        for (const auto& varInfo : variableInfos) {
            auto data = readVariableData(modelPath, varInfo);
            segments.push_back(createSegmentFromVariable(varInfo, data));
        }
        
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing TensorFlow model: " << e.what() << std::endl;
        throw ParsingError("Failed to parse TensorFlow model: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("TensorFlow model support is disabled. Please enable ENABLE_TENSORFLOW to use this feature.");
#endif
    
    return segments;
}

std::vector<ModelSegment> TensorFlowModelParser::parseWithChunking(const std::string& modelPath) const {
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
