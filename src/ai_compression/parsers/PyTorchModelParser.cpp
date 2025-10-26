/**
 * @file PyTorchModelParser.cpp
 * @brief Implementation of PyTorch model parsing with ZIP/Pickle support for LLaMA models
 */
#include "PyTorchModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>

// Add libzip for handling ZIP archives
#include <zip.h>

// c10 optional utilities for unpickle API
#include <c10/util/Optional.h>
// (no additional includes required for pointer+size unpickle overload)

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

// Helper function to check if file is a ZIP archive
bool PyTorchModelParser::isZipFile(const std::string& modelPath) const {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file) return false;
    
    unsigned char magic[4] = {0};
    file.read(reinterpret_cast<char*>(magic), 4);
    
    // ZIP local file header signature: 'P' 'K' 0x03 0x04
    return (magic[0] == 'P' && magic[1] == 'K' && magic[2] == 0x03 && magic[3] == 0x04);
}

// Extract tensors from LLaMA-style ZIP archive containing pickle + data files
std::vector<PyTorchModelParser::PyTorchTensorInfo> PyTorchModelParser::extractFromZipArchive(const std::string& modelPath) const {
    std::vector<PyTorchTensorInfo> tensors;
    
    std::cout << "Detected ZIP archive format (LLaMA-style consolidated.pth)" << std::endl;
    
    int err = 0;
    zip_t* archive = zip_open(modelPath.c_str(), ZIP_RDONLY, &err);
    if (!archive) {
        zip_error_t error;
        zip_error_init_with_code(&error, err);
        std::string error_msg = zip_error_strerror(&error);
        zip_error_fini(&error);
        throw std::runtime_error("Failed to open ZIP archive: " + error_msg);
    }
    
    try {
        // Find the .pkl file in the archive
        std::string pklFile;
        zip_int64_t num_entries = zip_get_num_entries(archive, 0);
        
        for (zip_int64_t i = 0; i < num_entries; ++i) {
            const char* name = zip_get_name(archive, i, 0);
            if (name) {
                std::string filename(name);
                if (filename.find(".pkl") != std::string::npos) {
                    pklFile = filename;
                    std::cout << "Found pickle file: " << pklFile << std::endl;
                    break;
                }
            }
        }
        
        if (pklFile.empty()) {
            throw std::runtime_error("No .pkl file found in ZIP archive");
        }
        
        // Read pickle file into memory (pkl is small) and unpickle via pointer+size
        zip_stat_t stat;
        zip_stat_init(&stat);
        if (zip_stat(archive, pklFile.c_str(), 0, &stat) != 0) {
            throw std::runtime_error("Failed to stat pickle file");
        }
        zip_file_t* pkl_file = zip_fopen(archive, pklFile.c_str(), 0);
        if (!pkl_file) {
            throw std::runtime_error("Failed to open pickle file in archive");
        }
        std::vector<char> pkl_data(static_cast<size_t>(stat.size));
        zip_int64_t bytes_read = zip_fread(pkl_file, pkl_data.data(), static_cast<zip_uint64_t>(pkl_data.size()));
        zip_fclose(pkl_file);
        if (bytes_read != static_cast<zip_int64_t>(pkl_data.size())) {
            throw std::runtime_error("Failed to read complete pickle file");
        }
        std::cout << "Unpickling model structure..." << std::endl;
        // Use vector-based pickle loader (compatible across libtorch versions)
        torch::IValue iv = torch::pickle_load(pkl_data);
        
        // Collect tensors from the unpickled structure
        std::function<void(const torch::IValue&, const std::string&)> collect_tensors;
        collect_tensors = [&](const torch::IValue& value, const std::string& prefix) {
            if (value.isTensor()) {
                const torch::Tensor tensor = value.toTensor();
                PyTorchTensorInfo info;
                info.name = prefix.empty() ? "tensor" : prefix;
                
                // For LLaMA models, tensors are stored as references to data files
                // We need to read them from the ZIP archive
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
                        continue;
                    }
                    const std::string next = prefix.empty() ? key_str : (prefix + "." + key_str);
                    collect_tensors(item.value(), next);
                }
                return;
            }
            
            if (value.isList()) {
                auto list = value.toList();
                for (size_t i = 0; i < list.size(); ++i) {
                    const std::string next = prefix.empty() ? std::to_string(i) : (prefix + "." + std::to_string(i));
                    collect_tensors(list.get(i), next);
                }
                return;
            }
            
            if (value.isTuple()) {
                auto elements = value.toTuple()->elements();
                for (size_t i = 0; i < elements.size(); ++i) {
                    const std::string next = prefix.empty() ? std::to_string(i) : (prefix + "." + std::to_string(i));
                    collect_tensors(elements[i], next);
                }
                return;
            }
        };
        
        collect_tensors(iv, "");
        
        zip_close(archive);
        
        std::cout << "Successfully loaded ZIP archive with " << tensors.size() << " tensors" << std::endl;
        
    } catch (const std::exception& e) {
        zip_close(archive);
        throw;
    }
    
    return tensors;
}

std::vector<PyTorchModelParser::PyTorchTensorInfo> PyTorchModelParser::extractTensorInfo(const std::string& modelPath) const {
    std::vector<PyTorchTensorInfo> tensors;
    
    try {
        std::cout << "Loading PyTorch model from: " << modelPath << std::endl;
        
        // Check if this is a ZIP archive (LLaMA consolidated.pth format)
        if (isZipFile(modelPath)) {
            return extractFromZipArchive(modelPath);
        }
        
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
            
            // Fallback: try to load as a pickled state dict
            try {
                std::ifstream in(modelPath, std::ios::binary);
                if (!in) {
                    throw std::runtime_error("Unable to open file for reading: " + modelPath);
                }
                
                // Try native C++ torch::load on a tensor dictionary
                try {
                    torch::serialize::InputArchive archive;
                    archive.load_from(in);

                    std::function<void(const std::string&, torch::serialize::InputArchive&)> walk;
                    walk = [&](const std::string& prefix, torch::serialize::InputArchive& ar) {
                        for (const auto& key : ar.keys()) {
                            torch::Tensor t;
                            if (ar.try_read(key, t)) {
                                PyTorchTensorInfo info;
                                info.name = prefix.empty() ? key : (prefix + "." + key);
                                auto contig = t.contiguous();
                                info.shape = contig.sizes().vec();
                                info.scalar_type = contig.scalar_type();
                                info.data = tensorToBytes(contig);
                                info.size_bytes = info.data.size();
                                tensors.push_back(std::move(info));
                                continue;
                            }

                            torch::serialize::InputArchive child;
                            if (ar.try_read(key, child)) {
                                const std::string next = prefix.empty() ? key : (prefix + "." + key);
                                walk(next, child);
                            }
                        }
                    };

                    walk(std::string(), archive);
                    if (!tensors.empty()) {
                        std::cout << "Successfully loaded zip-archive state dict with " << tensors.size() << " tensors" << std::endl;
                        return tensors;
                    }
                } catch (const std::exception& zipe) {
                    std::cerr << "Zip-archive state dict load via InputArchive failed: " << zipe.what() << std::endl;
                    in.clear();
                    in.seekg(0);
                }

                // Stream-unpickle directly from file without buffering whole contents
                // As a last resort, read file into memory and unpickle via pointer+size
                in.seekg(0, std::ios::end);
                const std::streamoff fsize = in.tellg();
                if (fsize <= 0) {
                    throw std::runtime_error("Empty or unreadable state dict file");
                }
                in.seekg(0, std::ios::beg);
                std::vector<char> buf(static_cast<size_t>(fsize));
                in.read(buf.data(), fsize);
                if (in.gcount() != fsize) {
                    throw std::runtime_error("Failed to read full state dict");
                }
                // Use vector-based pickle loader
                torch::IValue iv = torch::pickle_load(buf);

                // Recursively collect tensors
                auto collect = [&](const torch::IValue& value,
                                   const std::string& prefix,
                                   auto&& collect_ref) -> void {
                    if (value.isTensor()) {
                        const torch::Tensor tensor = value.toTensor();
                        PyTorchTensorInfo info;
                        info.name = prefix.empty() ? "tensor" : prefix;
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
                                continue;
                            }
                            const std::string next = prefix.empty() ? key_str : (prefix + "." + key_str);
                            collect_ref(item.value(), next, collect_ref);
                        }
                        return;
                    }

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