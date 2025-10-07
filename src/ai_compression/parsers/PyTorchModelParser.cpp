/**
 * @file PyTorchModelParser.cpp
 * @brief Implementation of PyTorch model parsing into archive segments.
 */
#include "PyTorchModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>

#ifdef ENABLE_PYTORCH
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/serialization/export.h>
#endif

namespace CortexAICompression {

PyTorchModelParser::PyTorchModelParser() {
    // Constructor implementation
}

PyTorchModelParser::~PyTorchModelParser() {
    // Destructor implementation
}

#ifdef ENABLE_PYTORCH

SegmentType PyTorchModelParser::pytorchScalarTypeToSegmentType(torch::ScalarType scalar_type) const {
    switch (scalar_type) {
        case torch::kFloat32:
            return SegmentType::WEIGHTS_FP32;
        case torch::kFloat16:
            return SegmentType::WEIGHTS_FP16;
        case torch::kInt8:
            return SegmentType::WEIGHTS_INT8;
        case torch::kInt32:
            return SegmentType::WEIGHTS_FP32; // Placeholder
        case torch::kInt64:
            return SegmentType::WEIGHTS_FP32; // Placeholder
        case torch::kUInt8:
            return SegmentType::WEIGHTS_INT8; // Placeholder
        default:
            std::cerr << "Warning: Unknown PyTorch scalar type (" << static_cast<int>(scalar_type) << ") encountered." << std::endl;
            return SegmentType::UNKNOWN;
    }
}

TensorMetadata PyTorchModelParser::extractTensorMetadata(const PyTorchTensorInfo& tensorInfo) const {
    TensorMetadata metadata;
    
    // Convert PyTorch shape to our format
    for (int64_t dim : tensorInfo.shape) {
        metadata.dimensions.push_back(static_cast<size_t>(dim));
    }
    
    // Calculate sparsity (placeholder - would need to analyze actual data)
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = false;
    
    return metadata;
}

std::string PyTorchModelParser::extractLayerName(const std::string& tensorName) const {
    // PyTorch tensor names often follow patterns like:
    // "layers.0.weight", "conv1.bias", "fc.weight"
    std::regex layer_pattern(R"(([^.]+)\.)");
    std::smatch matches;
    if (std::regex_search(tensorName, matches, layer_pattern)) {
        return matches[1].str();
    }
    return tensorName;
}

size_t PyTorchModelParser::extractLayerIndex(const std::string& tensorName) const {
    // Extract numeric index from tensor names like "layers.0", "conv1", "fc"
    std::regex index_pattern(R"((\d+))");
    std::smatch matches;
    if (std::regex_search(tensorName, matches, index_pattern)) {
        try {
            return std::stoul(matches[1].str());
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid layer index in tensor name: " << tensorName << std::endl;
        }
    }
    return 0;
}

std::vector<std::byte> PyTorchModelParser::tensorToBytes(const torch::Tensor& tensor) const {
    std::vector<std::byte> data;
    
    // Ensure tensor is contiguous
    auto contiguous_tensor = tensor.contiguous();
    
    // Get tensor data
    auto tensor_data = contiguous_tensor.data_ptr();
    size_t tensor_size = contiguous_tensor.numel() * contiguous_tensor.element_size();
    
    data.resize(tensor_size);
    std::memcpy(data.data(), tensor_data, tensor_size);
    
    return data;
}

std::vector<PyTorchModelParser::PyTorchTensorInfo> PyTorchModelParser::extractTensorInfo(const std::string& modelPath) const {
    std::vector<PyTorchTensorInfo> tensors;
    
    try {
        std::cout << "Loading PyTorch model from: " << modelPath << std::endl;
        
        // Try to load as a TorchScript model first
        try {
            torch::jit::script::Module module = torch::jit::load(modelPath);
            
            // Extract parameters from the module
            for (const auto& param : module.named_parameters()) {
                PyTorchTensorInfo tensorInfo;
                tensorInfo.name = param.name;
                
                const auto& tensor = param.value;
                tensorInfo.shape = tensor.sizes().vec();
                tensorInfo.scalar_type = tensor.scalar_type();
                tensorInfo.data = tensorToBytes(tensor);
                tensorInfo.size_bytes = tensorInfo.data.size();
                
                tensors.push_back(std::move(tensorInfo));
            }
            
            std::cout << "Successfully loaded TorchScript model with " << tensors.size() << " parameters" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Failed to load as TorchScript model, trying as state dict: " << e.what() << std::endl;
            
            // Fallback: try to load as a pickled state dict (.pth/.pt saved via torch.save)
            try {
                std::ifstream in(modelPath, std::ios::binary);
                if (!in) {
                    throw std::runtime_error("Unable to open file for reading: " + modelPath);
                }
                // Detect zip-archive style checkpoints (new PyTorch serialization)
                {
                    std::ifstream sig(modelPath, std::ios::binary);
                    unsigned char magic[4] = {0};
                    sig.read(reinterpret_cast<char*>(magic), 4);
                    // ZIP local file header signature: 'P' 'K' 0x03 0x04
                    if (sig.gcount() == 4 && magic[0] == 'P' && magic[1] == 'K' && magic[2] == 0x03 && magic[3] == 0x04) {
                        throw ParsingError(
                            "This .pth uses the new zip archive format. Please resave with _use_new_zipfile_serialization=False or provide ONNX/TorchScript.");
                    }
                }

                // Stream-unpickle directly from file to avoid large in-memory buffers
                auto reader = [&in](char* buf, size_t n) -> size_t {
                    if (!in.good()) return 0;
                    in.read(buf, static_cast<std::streamsize>(n));
                    return static_cast<size_t>(in.gcount());
                };
                torch::IValue iv = torch::jit::unpickle(reader, /*type_resolver*/nullptr, /*tensor_table*/{});

                // Helper to recursively collect tensors with hierarchical names
                auto collect = [&](const torch::IValue& value,
                                   const std::string& prefix,
                                   auto&& collect_ref) -> void {
                    if (value.isTensor()) {
                        const torch::Tensor tensor = value.toTensor();
                        PyTorchTensorInfo info;
                        info.name = prefix.empty() ? std::string("tensor") : prefix;
                        const auto contig = tensor.contiguous();
                        info.shape = contig.sizes().vec();
                        info.scalar_type = contig.scalar_type();
                        info.data = tensorToBytes(contig);
                        info.size_bytes = info.data.size();
                        tensors.push_back(std::move(info));
                        return;
                    }

                    if (value.isGenericDict()) {
                        auto dict = value.toGenericDict();
                        for (const auto& item : dict) {
                            std::string key_str;
                            if (item.key().isString()) {
                                key_str = item.key().toStringRef();
                            } else if (item.key().isInt()) {
                                key_str = std::to_string(item.key().toInt());
                            } else {
                                // Unsupported key type; skip
                                continue;
                            }
                            const std::string next = prefix.empty() ? key_str : (prefix + "." + key_str);
                            collect_ref(item.value(), next, collect_ref);
                        }
                        return;
                    }

                    // No value.isDict() in this LibTorch version; GenericDict covers dicts

                    if (value.isList()) {
                        auto list = value.toList();
                        for (size_t i = 0; i < list.size(); ++i) {
                            const std::string next = prefix.empty() ? std::to_string(i) : (prefix + "." + std::to_string(i));
                            collect_ref(list.get(i), next, collect_ref);
                        }
                        return;
                    }

                    if (value.isTuple()) {
                        auto elements = value.toTuple()->elements();
                        for (size_t i = 0; i < elements.size(); ++i) {
                            const std::string next = prefix.empty() ? std::to_string(i) : (prefix + "." + std::to_string(i));
                            collect_ref(elements[i], next, collect_ref);
                        }
                        return;
                    }

                    // Ignore other IValue types (e.g., objects, scalars) as they are not parameters
                };

                collect(iv, std::string(), collect);

                if (tensors.empty()) {
                    throw std::runtime_error("No tensors found in state dict");
                }

                std::cout << "Successfully loaded state dict with " << tensors.size() << " tensors" << std::endl;
            } catch (const std::exception& e2) {
                std::cerr << "Failed to load as state dict: " << e2.what() << std::endl;
                throw ParsingError("Failed to load PyTorch model in any supported format");
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting PyTorch tensor info: " << e.what() << std::endl;
        throw ParsingError("Failed to extract PyTorch tensor information: " + std::string(e.what()));
    }
    
    return tensors;
}

ModelSegment PyTorchModelParser::createSegmentFromTensor(const PyTorchTensorInfo& tensorInfo) const {
    ModelSegment segment;
    segment.name = tensorInfo.name;
    segment.type = pytorchScalarTypeToSegmentType(tensorInfo.scalar_type);
    segment.data = tensorInfo.data;
    segment.original_size = tensorInfo.size_bytes;
    segment.tensor_metadata = extractTensorMetadata(tensorInfo);
    segment.layer_name = extractLayerName(tensorInfo.name);
    segment.layer_index = extractLayerIndex(tensorInfo.name);
    
    // Determine layer type based on tensor name
    if (tensorInfo.name.find("weight") != std::string::npos) {
        segment.layer_type = "WEIGHTS";
    } else if (tensorInfo.name.find("bias") != std::string::npos) {
        segment.layer_type = "BIAS";
    } else if (tensorInfo.name.find("running_mean") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_MEAN";
    } else if (tensorInfo.name.find("running_var") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_VAR";
    } else if (tensorInfo.name.find("num_batches_tracked") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_COUNT";
    } else {
        segment.layer_type = "UNKNOWN";
    }
    
    return segment;
}

#endif // ENABLE_PYTORCH

std::vector<ModelSegment> PyTorchModelParser::parse(const std::string& modelPath) const {
    std::vector<ModelSegment> segments;
    
#ifdef ENABLE_PYTORCH
    try {
        std::cout << "Parsing PyTorch model: " << modelPath << std::endl;
        
        auto tensorInfos = extractTensorInfo(modelPath);
        segments.reserve(tensorInfos.size());
        
        for (const auto& tensorInfo : tensorInfos) {
            segments.push_back(createSegmentFromTensor(tensorInfo));
        }
        
        std::cout << "Successfully parsed PyTorch model with " << segments.size() << " tensors" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing PyTorch model: " << e.what() << std::endl;
        throw ParsingError("Failed to parse PyTorch model: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("PyTorch model support is disabled. Please enable ENABLE_PYTORCH to use this feature.");
#endif
    
    return segments;
}

std::vector<ModelSegment> PyTorchModelParser::parseWithChunking(const std::string& modelPath) const {
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
