#include "ONNXModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream> // For string stream serialization
#include <fstream> // For file reading
#include <unordered_set> // For initializer check

#ifdef ENABLE_ONNX_PROTOBUF
#include <../onnx_proto/onnx.pb.h> // Include ONNX Protobuf headers
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#endif


namespace CortexAICompression {

ONNXModelParser::ONNXModelParser() {
#ifdef ENABLE_ONNX
    // Keep ORT environment for potential future use or fallback, but parsing now uses Protobuf primarily
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CortexONNXParser");
    sessionOptions = std::make_unique<Ort::SessionOptions>();
    sessionOptions->SetIntraOpNumThreads(1);
    sessionOptions->SetInterOpNumThreads(1);
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL); // Disable ORT optimizations if using Protobuf directly
#endif
}

ONNXModelParser::~ONNXModelParser() {
    // Destructor implementation - resources are automatically cleaned up by unique_ptr
}

// --- Protobuf Helper Functions (if ENABLE_ONNX_PROTOBUF is defined) ---
#ifdef ENABLE_ONNX_PROTOBUF

// Helper to convert ONNX Tensor Type to SegmentType
SegmentType ONNXModelParser::onnxTensorTypeToSegmentType(int32_t onnx_type) const {
     switch (onnx_type) {
        case onnx::TensorProto::FLOAT:
            return SegmentType::WEIGHTS_FP32;
        case onnx::TensorProto::FLOAT16:
            return SegmentType::WEIGHTS_FP16;
        case onnx::TensorProto::INT8:
            return SegmentType::WEIGHTS_INT8;
        // Add mappings for other types as needed
        case onnx::TensorProto::INT32:
             return SegmentType::WEIGHTS_FP32; // Placeholder
        case onnx::TensorProto::INT64:
             return SegmentType::WEIGHTS_FP32; // Placeholder
        case onnx::TensorProto::UINT8:
             return SegmentType::WEIGHTS_INT8; // Placeholder
        // ... other types
        default:
            std::cerr << "Warning: Unknown ONNX tensor type (" << onnx_type << ") encountered." << std::endl;
            return SegmentType::UNKNOWN;
    }
}

// Extracts metadata from an ONNX TensorProto
TensorMetadata ONNXModelParser::extractTensorMetadataProto(const onnx::TensorProto& tensor) const {
    TensorMetadata metadata;
    metadata.sparsity_ratio = 0.0f; // Placeholder - Calculation requires parsing data
    metadata.is_sorted = false; // Placeholder - Calculation requires parsing data

    // Extract dimensions
    for (int64_t dim : tensor.dims()) {
        metadata.dimensions.push_back(static_cast<size_t>(dim));
    }
    // TODO: Calculate sparsity and sorted status if needed during compression analysis
    return metadata;
}

// Extracts raw byte data from an ONNX TensorProto.
std::vector<std::byte> ONNXModelParser::tensorProtoToBytes(const onnx::TensorProto& tensor) const {
    std::vector<std::byte> bytes;
    size_t element_size = 0;
    size_t num_elements = 1;
    for(int64_t dim : tensor.dims()) {
        // Handle potential symbolic dimensions (though unlikely for initializers)
        // Treat dim <= 0 as 1 for size calculation, but actual shape might be dynamic
        num_elements *= (dim > 0 ? static_cast<size_t>(dim) : 1);
    }

    if (tensor.has_raw_data()) {
        const std::string& raw_data = tensor.raw_data();
        bytes.assign(reinterpret_cast<const std::byte*>(raw_data.data()),
                     reinterpret_cast<const std::byte*>(raw_data.data() + raw_data.size()));
        // Optional: Verify raw_data size matches expected size based on dims and data_type
    } else {
        // Handle data stored in type-specific fields
        switch (tensor.data_type()) {
            case onnx::TensorProto::FLOAT:
                element_size = sizeof(float);
                if (num_elements == static_cast<size_t>(tensor.float_data_size())) {
                    bytes.resize(num_elements * element_size);
                    std::memcpy(bytes.data(), tensor.float_data().data(), bytes.size());
                }
                break;
            case onnx::TensorProto::INT32:
                element_size = sizeof(int32_t);
                 if (num_elements == static_cast<size_t>(tensor.int32_data_size())) {
                    bytes.resize(num_elements * element_size);
                    std::memcpy(bytes.data(), tensor.int32_data().data(), bytes.size());
                 }
                break;
            case onnx::TensorProto::INT64:
                 element_size = sizeof(int64_t);
                 if (num_elements == static_cast<size_t>(tensor.int64_data_size())) {
                    bytes.resize(num_elements * element_size);
                    std::memcpy(bytes.data(), tensor.int64_data().data(), bytes.size());
                 }
                break;
            case onnx::TensorProto::DOUBLE:
                 element_size = sizeof(double);
                 if (num_elements == static_cast<size_t>(tensor.double_data_size())) {
                    bytes.resize(num_elements * element_size);
                    std::memcpy(bytes.data(), tensor.double_data().data(), bytes.size());
                 }
                break;
            case onnx::TensorProto::UINT64:
                 element_size = sizeof(uint64_t);
                  if (num_elements == static_cast<size_t>(tensor.uint64_data_size())) {
                    bytes.resize(num_elements * element_size);
                    std::memcpy(bytes.data(), tensor.uint64_data().data(), bytes.size());
                  }
                 break;
            // --- Cases requiring careful handling ---
            case onnx::TensorProto::FLOAT16: // Stored as uint16 usually
            case onnx::TensorProto::BFLOAT16: // Stored as uint16 usually
                 element_size = sizeof(uint16_t);
                 // ONNX stores these in int32_data field, needs casting/interpretation
                 if (num_elements * element_size <= static_cast<size_t>(tensor.int32_data_size()) * sizeof(int32_t)) {
                     bytes.resize(num_elements * element_size);
                     // Direct memcpy might work if underlying storage is uint16 compatible
                     std::memcpy(bytes.data(), tensor.int32_data().data(), bytes.size());
                 }
                 break;
            case onnx::TensorProto::INT8:
            case onnx::TensorProto::UINT8:
            case onnx::TensorProto::INT16:
            case onnx::TensorProto::UINT16:
            case onnx::TensorProto::BOOL:
                 element_size = (tensor.data_type() == onnx::TensorProto::INT16 || tensor.data_type() == onnx::TensorProto::UINT16) ? 2 : 1;
                 if (num_elements * element_size <= static_cast<size_t>(tensor.int32_data_size()) * sizeof(int32_t)) {
                     bytes.resize(num_elements * element_size);
                     std::memcpy(bytes.data(), tensor.int32_data().data(), bytes.size());
                 }
                 break;
            case onnx::TensorProto::STRING:
                 if (num_elements == static_cast<size_t>(tensor.string_data_size())) {
                     std::ostringstream ss;
                     for(const auto& s : tensor.string_data()) {
                         uint32_t len = s.length();
                         ss.write(reinterpret_cast<const char*>(&len), sizeof(len));
                         ss.write(s.data(), len);
                     }
                     std::string combined = ss.str();
                      bytes.assign(reinterpret_cast<const std::byte*>(combined.data()),
                                   reinterpret_cast<const std::byte*>(combined.data() + combined.size()));
                 }
                 break;
            default:
                std::cerr << "Warning: Tensor '" << tensor.name() << "' has data type "
                          << tensor.data_type() << " which is not handled for non-raw data extraction." << std::endl;
                break;
        }

        if (bytes.empty() && num_elements > 0 && tensor.data_type() != onnx::TensorProto::STRING) {
             std::cerr << "Warning: Failed to extract data for tensor '" << tensor.name()
                       << "' from type-specific fields (expected elements: " << num_elements
                       << ", field size mismatch or unhandled type?)." << std::endl;
        }
    }
    return bytes;
}
#endif // ENABLE_ONNX_PROTOBUF


#ifdef ENABLE_ONNX
SegmentType ONNXModelParser::determineSegmentType(const std::string& tensorName, const Ort::Value& tensor) const {
    if (tensorName.find("attention") != std::string::npos) {
        return SegmentType::ATTENTION_WEIGHTS;
    } else if (tensorName.find("feed_forward") != std::string::npos) {
        return SegmentType::FEED_FORWARD_WEIGHTS;
    } else if (tensorName.find("embedding") != std::string::npos) {
        return SegmentType::EMBEDDING_WEIGHTS;
    } else if (tensorName.find("layer_norm") != std::string::npos) {
        return SegmentType::LAYER_NORM_WEIGHTS;
    }
    auto type_info = tensor.GetTypeInfo();
    auto tensor_type = type_info.GetTensorTypeAndShapeInfo();
    auto element_type = tensor_type.GetElementType();
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return SegmentType::WEIGHTS_FP32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return SegmentType::WEIGHTS_FP16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return SegmentType::WEIGHTS_INT8;
        default:
             std::cerr << "Warning: Unknown ORT tensor element type (" << element_type << ") encountered." << std::endl;
            return SegmentType::UNKNOWN;
    }
}

TensorMetadata ONNXModelParser::extractTensorMetadata(const Ort::Value& tensor) const {
    TensorMetadata metadata;
    auto type_info = tensor.GetTypeInfo();
    auto tensor_type = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_type.GetShape();
    for (int64_t dim : shape) {
        metadata.dimensions.push_back(static_cast<size_t>(dim));
    }
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = false;
    if (tensor_type.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        auto num_elements = tensor_type.GetElementCount();
        if (num_elements > 0) {
            const float* data = tensor.GetTensorData<float>();
            size_t num_zeros = 0;
            for (size_t i = 0; i < num_elements; ++i) {
                if (data[i] == 0.0f) {
                    ++num_zeros;
                }
            }
            metadata.sparsity_ratio = static_cast<float>(num_zeros) / static_cast<float>(num_elements);
        }
    }
    return metadata;
}

std::vector<std::byte> ONNXModelParser::tensorToBytes(const Ort::Value& tensor) const {
    auto type_info = tensor.GetTypeInfo();
    auto tensor_type = type_info.GetTensorTypeAndShapeInfo();
    auto element_type = tensor_type.GetElementType();
    auto num_elements = tensor_type.GetElementCount();
    size_t element_size = 0;
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   element_size = sizeof(float); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   element_size = sizeof(uint8_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    element_size = sizeof(int8_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  element_size = sizeof(uint16_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   element_size = sizeof(int16_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   element_size = sizeof(int32_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   element_size = sizeof(int64_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            element_size = 1; 
            std::cerr << "Warning: tensorToBytes using GetTensorRawData for STRING type. Behavior depends on specific tensor structure. Consider using GetStringTensorData for tensors of strings." << std::endl;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    element_size = sizeof(uint8_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: element_size = 2; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  element_size = sizeof(double); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  element_size = sizeof(uint32_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  element_size = sizeof(uint64_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: element_size = sizeof(float) * 2; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:element_size = sizeof(double) * 2; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:element_size = 2; break;
        default:
            std::cerr << "Warning: Unknown or unsupported ONNX tensor element type (" << element_type 
                      << ") for calculating element size in tensorToBytes. Defaulting element_size to 0." << std::endl;
            element_size = 0;
            break;
    }
    if (element_size == 0 && num_elements > 0 && 
        element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED && 
        element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        std::cerr << "Error: Element size is 0 for a non-empty tensor (type: " << element_type
                  << ", num_elements: " << num_elements << "). Cannot determine data size for raw copy." << std::endl;
    }
    std::vector<std::byte> bytes;
    size_t data_size = num_elements * element_size;
    bytes.resize(data_size);
    void* data_ptr = const_cast<void*>(tensor.GetTensorRawData());
    if (data_ptr && data_size > 0) {
        std::memcpy(bytes.data(), data_ptr, data_size);
    } else if (num_elements > 0) {
         std::cerr << "Warning: Failed to get raw data pointer or data size is zero for tensor. Type: "
                   << element_type << ", Elements: " << num_elements << std::endl;
         bytes.clear();
    }
    return bytes;
}
#endif


std::string ONNXModelParser::extractLayerName(const std::string& tensorName) const {
    std::regex layer_pattern(R"(layers\.(\d+)\.([\w\.]+))");
    std::smatch matches;
    if (std::regex_search(tensorName, matches, layer_pattern)) {
        return matches[2].str();
    }
    size_t last_dot = tensorName.find_last_of('.');
    if (last_dot != std::string::npos) {
        return tensorName.substr(last_dot + 1);
    }
    return tensorName;
}

size_t ONNXModelParser::extractLayerIndex(const std::string& tensorName) const {
    std::regex layer_pattern(R"(layers\.(\d+))");
    std::smatch matches;
    if (std::regex_search(tensorName, matches, layer_pattern)) {
        try {
            return std::stoul(matches[1].str());
        } catch (const std::out_of_range& oor) {
             std::cerr << "Warning: Layer index out of range in tensor name: " << tensorName << std::endl;
        } catch (const std::invalid_argument& ia) {
             std::cerr << "Warning: Invalid layer index in tensor name: " << tensorName << std::endl;
        }
    }
    return 0;
}

std::vector<ModelSegment> ONNXModelParser::parse(const std::string& modelPath) const {
#if defined(ENABLE_ONNX) && defined(ENABLE_ONNX_PROTOBUF)
    std::vector<ModelSegment> segments;
    onnx::ModelProto model_proto;

    std::ifstream model_stream(modelPath, std::ios::binary);
    if (!model_stream) {
        throw ParsingError("Failed to open ONNX model file: " + modelPath);
    }
    google::protobuf::io::IstreamInputStream zero_copy_input(&model_stream);
    google::protobuf::io::CodedInputStream coded_input(&zero_copy_input);
    if (!model_proto.ParseFromCodedStream(&coded_input)) {
         throw ParsingError("Failed to parse ONNX model protobuf from file: " + modelPath);
    }
    model_stream.close();

    const onnx::GraphProto& graph_proto = model_proto.graph();
    std::cout << "ONNX Model Proto Loaded. Graph: " << (graph_proto.has_name() ? graph_proto.name() : "N/A") << std::endl;
    std::cout << "  Initializers: " << graph_proto.initializer_size() << std::endl;
    std::cout << "  Nodes: " << graph_proto.node_size() << std::endl;
    std::cout << "  Inputs: " << graph_proto.input_size() << std::endl;
    std::cout << "  Outputs: " << graph_proto.output_size() << std::endl;

    ModelSegment meta_segment;
    meta_segment.name = "model_metadata";
    meta_segment.type = SegmentType::METADATA_JSON;
    std::ostringstream metadata_ss;
    metadata_ss << "Producer: " << (model_proto.has_producer_name() ? model_proto.producer_name() : "N/A") << "\n";
    metadata_ss << "GraphName: " << (graph_proto.has_name() ? graph_proto.name() : "N/A") << "\n";
    metadata_ss << "Domain: " << (model_proto.has_domain() ? model_proto.domain() : "N/A") << "\n";
    metadata_ss << "Description: " << (model_proto.has_doc_string() ? model_proto.doc_string() : "N/A") << "\n";
    if (model_proto.opset_import_size() > 0) {
        metadata_ss << "OpsetVersion: " << model_proto.opset_import(0).version() << "\n";
    } else {
         metadata_ss << "OpsetVersion: N/A\n";
    }
    metadata_ss << "IRVersion: " << model_proto.ir_version() << "\n";
    for(const auto& prop : model_proto.metadata_props()) {
        metadata_ss << prop.key() << ": " << prop.value() << "\n";
    }
    std::string metadata_str = metadata_ss.str();
     if (!metadata_str.empty()) {
        meta_segment.data.assign(reinterpret_cast<const std::byte*>(metadata_str.data()),
                                 reinterpret_cast<const std::byte*>(metadata_str.data() + metadata_str.size()));
        meta_segment.original_size = meta_segment.data.size();
        segments.push_back(std::move(meta_segment));
        std::cout << "  Added metadata segment (" << meta_segment.original_size << " bytes)." << std::endl;
    } else {
         std::cout << "  No significant metadata found to add." << std::endl;
    }

     std::cout << "Processing graph structure..." << std::endl;
    // Serialize the original graph_proto directly
    std::string serialized_graph_structure;
    if (!graph_proto.SerializeToString(&serialized_graph_structure)) {
        throw ParsingError("Failed to serialize graph structure protobuf.");
    }

    // DEBUG: Print first 32 bytes of serialized graph structure
    std::cerr << "DEBUG: Serialized graph structure size: " << serialized_graph_structure.size() << std::endl;
    std::cerr << "DEBUG: First 32 bytes of serialized graph structure: ";
    for (size_t i = 0; i < std::min<size_t>(32, serialized_graph_structure.size()); ++i) {
        std::cerr << std::hex << (static_cast<unsigned int>(static_cast<unsigned char>(serialized_graph_structure[i]))) << " ";
    }
    std::cerr << std::dec << std::endl;

    bool is_trivial_graph = false;
    if (serialized_graph_structure.empty()) {
        is_trivial_graph = true;
        std::cerr << "COMPRESSION INFO: Serialized graph structure for '" << modelPath << "' is EMPTY. Skipping segment." << std::endl;
    } else {
        is_trivial_graph = false;
    }

    if (!is_trivial_graph) {
        ModelSegment graph_segment;
        graph_segment.name = "graph_structure";
        graph_segment.type = SegmentType::GRAPH_STRUCTURE_PROTO;
        graph_segment.data.assign(reinterpret_cast<const std::byte*>(serialized_graph_structure.data()),
                                 reinterpret_cast<const std::byte*>(serialized_graph_structure.data() + serialized_graph_structure.size()));
        graph_segment.original_size = graph_segment.data.size();
        segments.push_back(std::move(graph_segment));
        std::cout << "  Added graph structure segment (" << graph_segment.original_size << " bytes)." << std::endl;
    }

    std::cout << "Processing initializers (weights)..." << std::endl;
    for (const auto& initializer : graph_proto.initializer()) {
        if (!initializer.has_name()) {
            std::cerr << "Warning: Skipping initializer without a name." << std::endl;
            continue;
        }
        std::cout << "  Processing initializer: " << initializer.name() << std::endl;
        ModelSegment weight_segment;
        weight_segment.name = initializer.name();
        weight_segment.layer_name = extractLayerName(initializer.name());
        weight_segment.layer_index = extractLayerIndex(initializer.name());
        weight_segment.type = onnxTensorTypeToSegmentType(initializer.data_type());
        if (weight_segment.type == SegmentType::UNKNOWN) {
             std::cerr << "    Warning: Skipping initializer '" << initializer.name()
                       << "' due to unknown data type: " << initializer.data_type() << std::endl;
             continue;
        }
        weight_segment.tensor_metadata = extractTensorMetadataProto(initializer);
        weight_segment.data = tensorProtoToBytes(initializer);
        weight_segment.original_size = weight_segment.data.size();
        if (weight_segment.original_size > 0) {
             segments.push_back(std::move(weight_segment));
             std::cout << "    Added weight segment (" << weight_segment.original_size << " bytes)." << std::endl;
        } else {
             std::cerr << "    Warning: Skipping empty weight segment for: " << initializer.name() << std::endl;
        }
    }
    return segments;
#elif defined(ENABLE_ONNX)
     throw ParsingError("ONNX Protobuf support is required for graph structure extraction but is not enabled/found. Cannot proceed.");
#else
    throw ParsingError("ONNX model support is disabled. Please enable ENABLE_ONNX and ENABLE_ONNX_PROTOBUF to use this feature.");
#endif
}

std::vector<ModelSegment> ONNXModelParser::parseWithChunking(const std::string& modelPath) const {
#ifdef ENABLE_ONNX
    auto segments = parse(modelPath);
    std::map<size_t, std::vector<ModelSegment*>> layerGroups;
    for (auto& segment : segments) {
        layerGroups[segment.layer_index].push_back(&segment);
    }
    for (auto& [layer, group] : layerGroups) {
        std::sort(group.begin(), group.end(),
                 [](const ModelSegment* a, const ModelSegment* b) {
                     return static_cast<int>(a->type) < static_cast<int>(b->type);
                 });
    }
    std::vector<ModelSegment> reorderedSegments;
    reorderedSegments.reserve(segments.size());
    std::unordered_set<std::string> addedSegmentNames;
    for (const ModelSegment& segment : segments) {
        if ( (segment.layer_index == 0 && segment.layer_name.empty()) ||
             segment.type == SegmentType::METADATA_JSON ||
             segment.type == SegmentType::GRAPH_STRUCTURE_PROTO ) {
            if (addedSegmentNames.find(segment.name) == addedSegmentNames.end()) {
                reorderedSegments.push_back(segment);
                addedSegmentNames.insert(segment.name);
            }
        }
    }
    for (const auto& pair : layerGroups) {
        size_t layer_idx = pair.first;
        const auto& group = pair.second;
        bool is_potentially_global_group = (layer_idx == 0);
        if (is_potentially_global_group && !group.empty()) {
            if (addedSegmentNames.count(group.front()->name) > 0 && group.front()->layer_name.empty()) {
                continue;
            }
        }
        for (const ModelSegment* segment_ptr : group) {
            if (addedSegmentNames.find(segment_ptr->name) == addedSegmentNames.end()) {
                reorderedSegments.push_back(*segment_ptr);
                addedSegmentNames.insert(segment_ptr->name);
            }
        }
    }
    if (reorderedSegments.size() < segments.size()) {
        for (const ModelSegment& original_segment : segments) {
            if (addedSegmentNames.find(original_segment.name) == addedSegmentNames.end()) {
                std::cerr << "Warning: Segment '" << original_segment.name 
                          << "' was missed in reordering, adding at the end." << std::endl;
                reorderedSegments.push_back(original_segment);
            }
        }
    }
    return reorderedSegments;
#else
    return parse(modelPath);
#endif
}

} // namespace CortexAICompression
