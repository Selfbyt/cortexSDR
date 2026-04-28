/**
 * @file ONNXModelParser.cpp
 * @brief Implementation of ONNX neural network model parsing for compression
 * 
 * This file implements the ONNXModelParser class which provides comprehensive
 * parsing of ONNX (Open Neural Network Exchange) format models for compression
 * preprocessing. Supports various ONNX operators and model architectures.
 * 
 * Key Features:
 * - Complete ONNX protobuf model parsing
 * - Layer-by-layer weight and bias extraction
 * - Metadata preservation for inference and extraction
 * - Multi-threading for large model processing
 * - Memory-efficient streaming parsing
 * - Support for quantized and specialized layer types
 */

#include "ONNXModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <array>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <future>
#include <thread>

#ifdef ENABLE_ONNX_PROTOBUF
#include <../onnx_proto/onnx.pb.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#endif


namespace CortexAICompression {

namespace {
#ifdef ENABLE_ONNX_PROTOBUF
template <typename T>
size_t rawElementCount(const onnx::TensorProto& tensor) {
    return static_cast<size_t>(tensor.raw_data().size()) / sizeof(T);
}
#endif

template <typename T>
void analyzeTypedData(const T* data, size_t count, size_t& num_zeros, bool& is_sorted) {
    if (!data || count == 0) {
        num_zeros = 0;
        is_sorted = true;
        return;
    }

    num_zeros = 0;
    is_sorted = true;
    T prev_value = data[0];
    if (prev_value == static_cast<T>(0)) {
        ++num_zeros;
    }
    for (size_t index = 1; index < count; ++index) {
        const T value = data[index];
        if (value == static_cast<T>(0)) {
            ++num_zeros;
        }
        if (value < prev_value) {
            is_sorted = false;
        }
        prev_value = value;
    }
}

std::string stripTensorSuffix(std::string tensor_name) {
    static const std::array<const char*, 10> suffixes = {
        ".weight", ".weights", ".bias", "/weight", "/weights",
        "/bias", "_weight", "_weights", "_bias", ".packed"
    };
    for (const char* suffix : suffixes) {
        const std::string suffix_str(suffix);
        if (tensor_name.size() > suffix_str.size() &&
            tensor_name.compare(tensor_name.size() - suffix_str.size(), suffix_str.size(), suffix_str) == 0) {
            tensor_name.erase(tensor_name.size() - suffix_str.size());
            break;
        }
    }
    return tensor_name;
}

#ifdef ENABLE_ONNX_PROTOBUF
std::string serializeNodeMetadata(const onnx::NodeProto& node) {
    std::ostringstream metadata;
    metadata << "op_type " << node.op_type() << " ";
    for (const auto& attr : node.attribute()) {
        metadata << attr.name() << " ";
        switch (attr.type()) {
            case onnx::AttributeProto::INT:
                metadata << attr.i() << " ";
                break;
            case onnx::AttributeProto::FLOAT:
                metadata << attr.f() << " ";
                break;
            case onnx::AttributeProto::STRING:
                metadata << attr.s() << " ";
                break;
            case onnx::AttributeProto::INTS:
                for (int index = 0; index < attr.ints_size(); ++index) {
                    metadata << attr.ints(index);
                    if (index + 1 < attr.ints_size()) {
                        metadata << ",";
                    }
                }
                metadata << " ";
                break;
            case onnx::AttributeProto::FLOATS:
                for (int index = 0; index < attr.floats_size(); ++index) {
                    metadata << attr.floats(index);
                    if (index + 1 < attr.floats_size()) {
                        metadata << ",";
                    }
                }
                metadata << " ";
                break;
            case onnx::AttributeProto::STRINGS:
                for (int index = 0; index < attr.strings_size(); ++index) {
                    metadata << attr.strings(index);
                    if (index + 1 < attr.strings_size()) {
                        metadata << ",";
                    }
                }
                metadata << " ";
                break;
            default:
                break;
        }
    }
    return metadata.str();
}

std::vector<size_t> extractShapeFromValueInfo(const onnx::ValueInfoProto& value_info) {
    std::vector<size_t> shape;
    if (value_info.has_type() && value_info.type().has_tensor_type() &&
        value_info.type().tensor_type().has_shape()) {
        const auto& shape_proto = value_info.type().tensor_type().shape();
        for (const auto& dim : shape_proto.dim()) {
            if (dim.has_dim_value()) {
                shape.push_back(static_cast<size_t>(dim.dim_value()));
            }
        }
    }
    return shape;
}

std::unordered_map<std::string, std::string> buildTensorMetadataMap(const onnx::GraphProto& graph_proto) {
    std::unordered_map<std::string, std::string> tensor_to_metadata;
    tensor_to_metadata.reserve(static_cast<size_t>(graph_proto.node_size()));
    for (const auto& node : graph_proto.node()) {
        const std::string metadata = serializeNodeMetadata(node);
        for (const auto& output_name : node.output()) {
            tensor_to_metadata[output_name] = metadata;
        }
    }
    return tensor_to_metadata;
}

std::unordered_map<std::string, std::vector<size_t>> buildTensorShapeMap(const onnx::GraphProto& graph_proto) {
    std::unordered_map<std::string, std::vector<size_t>> tensor_shapes;
    tensor_shapes.reserve(static_cast<size_t>(graph_proto.value_info_size() + graph_proto.input_size() + graph_proto.output_size()));

    for (const auto& value_info : graph_proto.value_info()) {
        tensor_shapes[value_info.name()] = extractShapeFromValueInfo(value_info);
    }
    for (const auto& input : graph_proto.input()) {
        tensor_shapes[input.name()] = extractShapeFromValueInfo(input);
    }
    for (const auto& output : graph_proto.output()) {
        tensor_shapes[output.name()] = extractShapeFromValueInfo(output);
    }

    return tensor_shapes;
}

std::unordered_map<std::string, const onnx::NodeProto*> buildInitializerNodeMap(const onnx::GraphProto& graph_proto) {
    std::unordered_map<std::string, const onnx::NodeProto*> tensor_to_node;
    for (const auto& node : graph_proto.node()) {
        for (const auto& input_name : node.input()) {
            auto existing = tensor_to_node.find(input_name);
            if (existing == tensor_to_node.end() || (existing->second != nullptr && existing->second->name().empty() && !node.name().empty())) {
                tensor_to_node[input_name] = &node;
            }
        }
    }
    return tensor_to_node;
}
#endif

} // namespace

/**
 * @brief Constructor for ONNXModelParser with protobuf initialization
 * 
 * Initializes the ONNX model parser with protobuf support and sets up
 * the parsing environment for efficient model processing.
 */
ONNXModelParser::ONNXModelParser() {
#ifdef ENABLE_ONNX
    // Initialize ORT environment for potential fallback parsing
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
        case onnx::TensorProto::DOUBLE:
            return SegmentType::WEIGHTS_FP32;
        case onnx::TensorProto::FLOAT16:
            return SegmentType::WEIGHTS_FP16;
        case onnx::TensorProto::INT8:
        case onnx::TensorProto::UINT8:
            return SegmentType::WEIGHTS_INT8;
        case onnx::TensorProto::INT32:
        case onnx::TensorProto::INT64:
        case onnx::TensorProto::BOOL:
            return SegmentType::UNKNOWN;
        default:
            std::cerr << "Warning: Unknown ONNX tensor type (" << onnx_type << ") encountered." << std::endl;
            return SegmentType::UNKNOWN;
    }
}

// Extracts metadata from an ONNX TensorProto
TensorMetadata ONNXModelParser::extractTensorMetadataProto(const onnx::TensorProto& tensor) const {
    TensorMetadata metadata;

    // Extract dimensions
    for (int64_t dim : tensor.dims()) {
        metadata.dimensions.push_back(static_cast<size_t>(dim));
    }

    // Calculate sparsity ratio
    size_t total_elements = 1;
    for (int64_t dim : tensor.dims()) {
        total_elements *= static_cast<size_t>(dim);
    }

    if (total_elements > 0) {
        size_t num_zeros = 0;
        bool is_sorted = true;

        // Handle different data types
        switch (tensor.data_type()) {
            case onnx::TensorProto::FLOAT: {
                if (tensor.has_raw_data()) {
                    const float* data = reinterpret_cast<const float*>(tensor.raw_data().data());
                    analyzeTypedData(data, std::min(total_elements, rawElementCount<float>(tensor)), num_zeros, is_sorted);
                } else {
                    std::vector<float> values;
                    values.reserve(static_cast<size_t>(tensor.float_data_size()));
                    for (int index = 0; index < tensor.float_data_size(); ++index) {
                        values.push_back(tensor.float_data(index));
                    }
                    analyzeTypedData(values.data(), values.size(), num_zeros, is_sorted);
                }
                break;
            }
            case onnx::TensorProto::INT32: {
                if (tensor.has_raw_data()) {
                    const int32_t* data = reinterpret_cast<const int32_t*>(tensor.raw_data().data());
                    analyzeTypedData(data, std::min(total_elements, rawElementCount<int32_t>(tensor)), num_zeros, is_sorted);
                } else {
                    std::vector<int32_t> values;
                    values.reserve(static_cast<size_t>(tensor.int32_data_size()));
                    for (int index = 0; index < tensor.int32_data_size(); ++index) {
                        values.push_back(tensor.int32_data(index));
                    }
                    analyzeTypedData(values.data(), values.size(), num_zeros, is_sorted);
                }
                break;
            }
            case onnx::TensorProto::INT64: {
                if (tensor.has_raw_data()) {
                    const int64_t* data = reinterpret_cast<const int64_t*>(tensor.raw_data().data());
                    analyzeTypedData(data, std::min(total_elements, rawElementCount<int64_t>(tensor)), num_zeros, is_sorted);
                } else {
                    std::vector<int64_t> values;
                    values.reserve(static_cast<size_t>(tensor.int64_data_size()));
                    for (int index = 0; index < tensor.int64_data_size(); ++index) {
                        values.push_back(tensor.int64_data(index));
                    }
                    analyzeTypedData(values.data(), values.size(), num_zeros, is_sorted);
                }
                break;
            }
            case onnx::TensorProto::INT8:
            case onnx::TensorProto::UINT8: {
                if (tensor.has_raw_data()) {
                    const uint8_t* data = reinterpret_cast<const uint8_t*>(tensor.raw_data().data());
                    analyzeTypedData(data, std::min(total_elements, rawElementCount<uint8_t>(tensor)), num_zeros, is_sorted);
                } else if (tensor.int32_data_size() > 0) {
                    std::vector<uint8_t> values;
                    values.reserve(static_cast<size_t>(tensor.int32_data_size()));
                    for (int index = 0; index < tensor.int32_data_size(); ++index) {
                        values.push_back(static_cast<uint8_t>(tensor.int32_data(index) & 0xFF));
                    }
                    analyzeTypedData(values.data(), values.size(), num_zeros, is_sorted);
                }
                break;
            }
            default:
                num_zeros = 0;
                is_sorted = false;
        }

        metadata.sparsity_ratio = static_cast<float>(num_zeros) / static_cast<float>(total_elements);
        metadata.is_sorted = is_sorted;
    } else {
        metadata.sparsity_ratio = 0.0f;
        metadata.is_sorted = true;
    }

    return metadata;
}

// Extracts raw byte data from an ONNX TensorProto.
std::vector<std::byte> ONNXModelParser::tensorProtoToBytes(const onnx::TensorProto& tensor) const {
    std::vector<std::byte> data;
    
    // If raw_data is present, use it directly
    if (tensor.has_raw_data()) {
        const std::string& raw_data = tensor.raw_data();
        data.assign(
            reinterpret_cast<const std::byte*>(raw_data.data()),
            reinterpret_cast<const std::byte*>(raw_data.data() + raw_data.size())
        );
        return data;
    }
    
    // Otherwise, handle specific data types
    switch (tensor.data_type()) {
        case onnx::TensorProto::FLOAT: {
            size_t num_elements = tensor.float_data_size();
            data.resize(num_elements * sizeof(float));
            float* float_data = reinterpret_cast<float*>(data.data());
            for (int index = 0; index < tensor.float_data_size(); ++index) {
                float_data[index] = tensor.float_data(index);
            }
            break;
        }
        case onnx::TensorProto::DOUBLE: {
            size_t num_elements = static_cast<size_t>(tensor.double_data_size());
            data.resize(num_elements * sizeof(double));
            double* double_data = reinterpret_cast<double*>(data.data());
            for (int index = 0; index < tensor.double_data_size(); ++index) {
                double_data[index] = tensor.double_data(index);
            }
            break;
        }
        case onnx::TensorProto::FLOAT16: {
            size_t num_elements = static_cast<size_t>(tensor.int32_data_size());
            if (num_elements > 0) {
                data.resize(num_elements * sizeof(uint16_t));
                uint16_t* half_data = reinterpret_cast<uint16_t*>(data.data());
                for (int index = 0; index < tensor.int32_data_size(); ++index) {
                    half_data[index] = static_cast<uint16_t>(tensor.int32_data(index) & 0xFFFF);
                }
            }
            break;
        }
        case onnx::TensorProto::INT32: {
            size_t num_elements = tensor.int32_data_size();
            data.resize(num_elements * sizeof(int32_t));
            int32_t* int_data = reinterpret_cast<int32_t*>(data.data());
            for (int index = 0; index < tensor.int32_data_size(); ++index) {
                int_data[index] = tensor.int32_data(index);
            }
            break;
        }
        case onnx::TensorProto::INT64: {
            size_t num_elements = tensor.int64_data_size();
            data.resize(num_elements * sizeof(int64_t));
            int64_t* int_data = reinterpret_cast<int64_t*>(data.data());
            for (int index = 0; index < tensor.int64_data_size(); ++index) {
                int_data[index] = tensor.int64_data(index);
            }
            break;
        }
        case onnx::TensorProto::INT8:
        case onnx::TensorProto::UINT8: {
            size_t num_elements = tensor.int32_data_size();
            if (num_elements > 0) {
                data.resize(num_elements);
                for (int index = 0; index < tensor.int32_data_size(); ++index) {
                    data[static_cast<size_t>(index)] = static_cast<std::byte>(tensor.int32_data(index) & 0xFF);
                }
            }
            break;
        }
        case onnx::TensorProto::BOOL: {
            size_t num_elements = static_cast<size_t>(tensor.int32_data_size());
            if (num_elements > 0) {
                data.resize(num_elements);
                for (int index = 0; index < tensor.int32_data_size(); ++index) {
                    data[static_cast<size_t>(index)] = static_cast<std::byte>(tensor.int32_data(index) ? 1 : 0);
                }
            }
            break;
        }
        default:
            std::cerr << "Warning: Unsupported tensor data type for conversion to bytes: " 
                      << tensor.data_type() << std::endl;
    }
    
    return data;
}
#endif // ENABLE_ONNX_PROTOBUF


#ifdef ENABLE_ONNX
SegmentType ONNXModelParser::determineSegmentType(const std::string& tensorName) const {
    std::string lower_name = tensorName;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (lower_name.find("embed") != std::string::npos ||
        lower_name.find("wte") != std::string::npos ||
        lower_name.find("embedding") != std::string::npos) {
        return SegmentType::EMBEDDING_WEIGHTS;
    }
    if (lower_name.find("attn") != std::string::npos ||
        lower_name.find("attention") != std::string::npos) {
        return SegmentType::ATTENTION_WEIGHTS;
    }
    if (lower_name.find("mlp") != std::string::npos ||
        lower_name.find("ffn") != std::string::npos ||
        lower_name.find("feed_forward") != std::string::npos) {
        return SegmentType::FEED_FORWARD_WEIGHTS;
    }
    if (lower_name.find("norm") != std::string::npos ||
        lower_name.find("ln") != std::string::npos) {
        return SegmentType::LAYER_NORM_WEIGHTS;
    }
    return SegmentType::WEIGHTS_FP32;
}

SegmentType ONNXModelParser::determineSegmentType(const std::string& tensorName, const Ort::Value& tensor) const {
    // Determine segment type based on tensor name and properties
    ONNXTensorElementDataType element_type = tensor.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
    const SegmentType name_based_type = determineSegmentType(tensorName);
    if (name_based_type != SegmentType::WEIGHTS_FP32) {
        return name_based_type;
    }
    
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return SegmentType::WEIGHTS_FP32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return SegmentType::WEIGHTS_FP16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return SegmentType::WEIGHTS_INT8;
        // Add more mappings as needed
        default:
            return SegmentType::UNKNOWN;
    }
}

TensorMetadata ONNXModelParser::extractTensorMetadata(const Ort::Value& tensor) const {
    TensorMetadata metadata;
    
    // Get tensor shape
    auto tensor_info = tensor.GetTypeInfo().GetTensorTypeAndShapeInfo();
    auto dims = tensor_info.GetShape();
    for (auto dim : dims) {
        metadata.dimensions.push_back(static_cast<size_t>(dim));
    }
    
    // Calculate sparsity and check if sorted
    ONNXTensorElementDataType element_type = tensor_info.GetElementType();
    size_t total_elements = 1;
    for (auto dim : dims) {
        total_elements *= static_cast<size_t>(dim);
    }
    
    if (total_elements > 0) {
        size_t num_zeros = 0;
        bool is_sorted = true;
        
        // Handle different data types
        switch (element_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                const float* data = tensor.GetTensorData<float>();
                analyzeTypedData(data, total_elements, num_zeros, is_sorted);
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                const int32_t* data = tensor.GetTensorData<int32_t>();
                analyzeTypedData(data, total_elements, num_zeros, is_sorted);
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                const int64_t* data = tensor.GetTensorData<int64_t>();
                analyzeTypedData(data, total_elements, num_zeros, is_sorted);
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                const uint8_t* data = tensor.GetTensorData<uint8_t>();
                analyzeTypedData(data, total_elements, num_zeros, is_sorted);
                break;
            }
            default:
                num_zeros = 0;
                is_sorted = false;
        }
        
        metadata.sparsity_ratio = static_cast<float>(num_zeros) / static_cast<float>(total_elements);
        metadata.is_sorted = is_sorted;
    } else {
        metadata.sparsity_ratio = 0.0f;
        metadata.is_sorted = true;
    }
    
    return metadata;
}

std::vector<std::byte> ONNXModelParser::tensorToBytes(const Ort::Value& tensor) const {
    std::vector<std::byte> data;
    
    auto tensor_info = tensor.GetTypeInfo().GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType element_type = tensor_info.GetElementType();
    size_t total_elements = 1;
    auto dims = tensor_info.GetShape();
    for (auto dim : dims) {
        total_elements *= static_cast<size_t>(dim);
    }
    
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            const float* float_data = tensor.GetTensorData<float>();
            size_t byte_size = total_elements * sizeof(float);
            data.resize(byte_size);
            std::memcpy(data.data(), float_data, byte_size);
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            const uint16_t* half_data = tensor.GetTensorData<uint16_t>();
            size_t byte_size = total_elements * sizeof(uint16_t);
            data.resize(byte_size);
            std::memcpy(data.data(), half_data, byte_size);
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
            const uint8_t* int8_data = tensor.GetTensorData<uint8_t>();
            size_t byte_size = total_elements * sizeof(uint8_t);
            data.resize(byte_size);
            std::memcpy(data.data(), int8_data, byte_size);
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            const int32_t* int_data = tensor.GetTensorData<int32_t>();
            size_t byte_size = total_elements * sizeof(int32_t);
            data.resize(byte_size);
            std::memcpy(data.data(), int_data, byte_size);
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            const int64_t* int_data = tensor.GetTensorData<int64_t>();
            size_t byte_size = total_elements * sizeof(int64_t);
            data.resize(byte_size);
            std::memcpy(data.data(), int_data, byte_size);
            break;
        }
        default:
            std::cerr << "Warning: Unsupported tensor data type for conversion to bytes: " 
                      << element_type << std::endl;
    }
    
    return data;
}

ModelSegment ONNXModelParser::createSegmentFromTensor(const std::string& name, const Ort::Value& tensor) const {
    ModelSegment segment;
    segment.name = name;
    segment.type = determineSegmentType(name, tensor);
    segment.data = tensorToBytes(tensor);
    segment.original_size = segment.data.size();
    segment.tensor_metadata = extractTensorMetadata(tensor);
    segment.layer_name = extractLayerName(name);
    segment.layer_index = extractLayerIndex(name);
    segment.data_format = "ONNX_RUNTIME";

    const std::string lower_name = [&name]() {
        std::string normalized = name;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return normalized;
    }();
    if (lower_name.find("bias") != std::string::npos) {
        segment.layer_type = "BIAS";
    } else {
        segment.layer_type = "WEIGHTS";
    }

    return segment;
}
#endif

#ifndef ENABLE_ONNX
SegmentType ONNXModelParser::determineSegmentType(const std::string& tensorName) const {
    std::string lower_name = tensorName;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (lower_name.find("embed") != std::string::npos ||
        lower_name.find("wte") != std::string::npos ||
        lower_name.find("embedding") != std::string::npos) {
        return SegmentType::EMBEDDING_WEIGHTS;
    }
    if (lower_name.find("attn") != std::string::npos ||
        lower_name.find("attention") != std::string::npos) {
        return SegmentType::ATTENTION_WEIGHTS;
    }
    if (lower_name.find("mlp") != std::string::npos ||
        lower_name.find("ffn") != std::string::npos ||
        lower_name.find("feed_forward") != std::string::npos) {
        return SegmentType::FEED_FORWARD_WEIGHTS;
    }
    if (lower_name.find("norm") != std::string::npos ||
        lower_name.find("ln") != std::string::npos) {
        return SegmentType::LAYER_NORM_WEIGHTS;
    }
    return SegmentType::WEIGHTS_FP32;
}
#endif

std::string ONNXModelParser::extractLayerName(const std::string& tensorName) const {
    static const std::array<std::regex, 5> indexed_patterns = {
        std::regex(R"((?:^|\.|/)(?:layers|layer|blocks|block|h)\.(\d+)(?:\.|/|$))"),
        std::regex(R"((?:^|\.|/)(?:layers|layer|blocks|block|h)/(\d+)(?:\.|/|$))"),
        std::regex(R"((?:^|\.|/)(?:layers|layer|blocks|block|h)_(\d+)(?:\.|/|$))"),
        std::regex(R"((?:^|\.|/)(\d+)(?:\.attn|\.mlp|\.ln|/attn|/mlp|/ln))"),
        std::regex(R"((?:transformer|model)\.(?:h|layers)\.(\d+)(?:\.|/|$))")
    };

    std::smatch matches;
    for (const auto& pattern : indexed_patterns) {
        if (std::regex_search(tensorName, matches, pattern)) {
            return "layer_" + matches[1].str();
        }
    }
    return stripTensorSuffix(tensorName);
}

size_t ONNXModelParser::extractLayerIndex(const std::string& tensorName) const {
    static const std::array<std::regex, 5> indexed_patterns = {
        std::regex(R"((?:^|\.|/)(?:layers|layer|blocks|block|h)\.(\d+)(?:\.|/|$))"),
        std::regex(R"((?:^|\.|/)(?:layers|layer|blocks|block|h)/(\d+)(?:\.|/|$))"),
        std::regex(R"((?:^|\.|/)(?:layers|layer|blocks|block|h)_(\d+)(?:\.|/|$))"),
        std::regex(R"((?:^|\.|/)(\d+)(?:\.attn|\.mlp|\.ln|/attn|/mlp|/ln))"),
        std::regex(R"((?:transformer|model)\.(?:h|layers)\.(\d+)(?:\.|/|$))")
    };

    std::smatch matches;
    for (const auto& pattern : indexed_patterns) {
        if (std::regex_search(tensorName, matches, pattern)) {
            try {
                return static_cast<size_t>(std::stoull(matches[1].str()));
            } catch (const std::out_of_range&) {
                std::cerr << "Warning: Layer index out of range in tensor name: " << tensorName << std::endl;
            } catch (const std::invalid_argument&) {
                std::cerr << "Warning: Invalid layer index in tensor name: " << tensorName << std::endl;
            }
        }
    }
    return 0;
}

std::vector<ModelSegment> ONNXModelParser::parse(const std::string& modelPath) const {
    std::vector<ModelSegment> segments;
    
#ifdef ENABLE_ONNX_PROTOBUF
    onnx::ModelProto model_proto;
    std::ifstream model_file(modelPath, std::ios::binary);
    if (!model_file) {
        throw std::runtime_error("Failed to open model file: " + modelPath);
    }

    if (!model_proto.ParseFromIstream(&model_file)) {
        throw std::runtime_error("Failed to parse model file using Protobuf");
    }
    processMetadata(model_proto, segments);
    processGraphStructure(model_proto.graph(), segments);
    processInitializersParallel(model_proto.graph(), segments);

    return segments;
#else
    throw std::runtime_error("ONNX model support is disabled. Please enable ENABLE_ONNX_PROTOBUF to use this feature.");
#endif
}


std::vector<ModelSegment> ONNXModelParser::parseWithChunking(const std::string& modelPath) const {
    auto segments = parse(modelPath);

    std::map<size_t, std::vector<ModelSegment*>> layer_groups;
    for (auto& segment : segments) {
        layer_groups[segment.layer_index].push_back(&segment);
    }

    for (auto& [layer_index, group] : layer_groups) {
        (void)layer_index;
        std::sort(group.begin(), group.end(), [](const ModelSegment* lhs, const ModelSegment* rhs) {
            if (lhs->type == rhs->type) {
                return lhs->name < rhs->name;
            }
            return static_cast<int>(lhs->type) < static_cast<int>(rhs->type);
        });
    }

    std::vector<ModelSegment> reordered_segments;
    reordered_segments.reserve(segments.size());

    for (const auto& segment : segments) {
        if (segment.layer_index == 0 && segment.layer_name.empty()) {
            reordered_segments.push_back(segment);
        }
    }

    for (const auto& [layer_index, group] : layer_groups) {
        (void)layer_index;
        for (const auto* segment : group) {
            if (segment->layer_index == 0 && segment->layer_name.empty()) {
                continue;
            }
            reordered_segments.push_back(*segment);
        }
    }

    return reordered_segments;
}

#ifdef ENABLE_ONNX_PROTOBUF
void ONNXModelParser::processMetadata(const onnx::ModelProto& model_proto, std::vector<ModelSegment>& segments) const {
    ModelSegment meta_segment;
    meta_segment.name = "model_metadata";
    meta_segment.type = SegmentType::METADATA_JSON;

    TensorMetadata metadata;
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = true;
    meta_segment.tensor_metadata = metadata;

    std::ostringstream metadata_stream;
    metadata_stream << "Producer: " << (model_proto.has_producer_name() ? model_proto.producer_name() : "N/A") << "\n";
    metadata_stream << "Domain: " << (model_proto.has_domain() ? model_proto.domain() : "N/A") << "\n";
    metadata_stream << "IRVersion: " << model_proto.ir_version() << "\n";
    if (model_proto.opset_import_size() > 0) {
        metadata_stream << "OpsetVersion: " << model_proto.opset_import(0).version() << "\n";
    }

    const std::string metadata_payload = metadata_stream.str();
    if (metadata_payload.empty()) {
        return;
    }

    meta_segment.data.assign(
        reinterpret_cast<const std::byte*>(metadata_payload.data()),
        reinterpret_cast<const std::byte*>(metadata_payload.data() + metadata_payload.size())
    );
    meta_segment.original_size = meta_segment.data.size();
    segments.push_back(std::move(meta_segment));
}

void ONNXModelParser::processGraphStructure(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const {
    ModelSegment graph_segment;
    graph_segment.name = "model_structure";
    graph_segment.type = SegmentType::GRAPH_STRUCTURE_PROTO;

    TensorMetadata metadata;
    metadata.dimensions = {
        static_cast<size_t>(graph_proto.node_size()),
        static_cast<size_t>(graph_proto.initializer_size()),
        static_cast<size_t>(graph_proto.input_size()),
        static_cast<size_t>(graph_proto.output_size())
    };
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = true;
    graph_segment.tensor_metadata = metadata;

    std::string serialized_graph;
    if (!graph_proto.SerializeToString(&serialized_graph) || serialized_graph.empty()) {
        std::cerr << "Warning: Failed to serialize ONNX graph structure." << std::endl;
        return;
    }

    graph_segment.data.assign(
        reinterpret_cast<const std::byte*>(serialized_graph.data()),
        reinterpret_cast<const std::byte*>(serialized_graph.data() + serialized_graph.size())
    );
    graph_segment.original_size = graph_segment.data.size();
    segments.push_back(std::move(graph_segment));

    const auto tensor_shapes = buildTensorShapeMap(graph_proto);
    size_t node_index = 0;
    for (const auto& node : graph_proto.node()) {
        ++node_index;
        std::string op_type = node.op_type();
        std::string lower_op = op_type;
        std::transform(lower_op.begin(), lower_op.end(), lower_op.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });

        bool supported_op = true;
        std::string metadata_type_value = lower_op;
        if (lower_op == "add") {
            op_type = "Add";
        } else if (lower_op == "sub") {
            op_type = "Sub";
        } else if (lower_op == "mul") {
            op_type = "Mul";
        } else if (lower_op == "div") {
            op_type = "Div";
        } else if (lower_op == "relu") {
            op_type = "Relu";
        } else if (lower_op == "sigmoid") {
            op_type = "Sigmoid";
        } else if (lower_op == "hardsigmoid") {
            op_type = "Sigmoid";
            metadata_type_value = "hardsigmoid";
        } else if (lower_op == "tanh") {
            op_type = "Tanh";
        } else if (lower_op == "softmax") {
            op_type = "Softmax";
        } else if (lower_op == "gelu") {
            op_type = "Gelu";
        } else if (lower_op == "leakyrelu") {
            op_type = "LeakyRelu";
        } else if (lower_op == "elu") {
            op_type = "Elu";
        } else if (lower_op == "silu" || lower_op == "swish") {
            op_type = "Silu";
        } else if (lower_op == "reshape") {
            op_type = "Reshape";
        } else if (lower_op == "transpose") {
            op_type = "Transpose";
        } else if (lower_op == "flatten") {
            op_type = "Flatten";
        } else if (lower_op == "concat") {
            op_type = "Concat";
        } else if (lower_op == "slice") {
            op_type = "Slice";
        } else if (lower_op == "gather") {
            op_type = "Gather";
        } else if (lower_op == "matmul") {
            op_type = "MatMul";
        } else if (lower_op == "gemm") {
            op_type = "Gemm";
        } else if (lower_op == "conv") {
            op_type = "Conv";
            metadata_type_value = "conv";
        } else if (lower_op == "convtranspose") {
            op_type = "ConvTranspose";
            metadata_type_value = "conv";
        } else if (lower_op == "batchnormalization") {
            op_type = "BatchNormalization";
            metadata_type_value = "batchnormalization";
        } else if (lower_op == "layernormalization" ||
                   lower_op == "skiplayernormalization" ||
                   lower_op == "simplifiedlayernormalization") {
            op_type = "LayerNormalization";
            metadata_type_value = "layernormalization";
        } else if (lower_op == "maxpool") {
            op_type = "MaxPool";
            metadata_type_value = "maxpool";
        } else if (lower_op == "averagepool" || lower_op == "avgpool") {
            op_type = "AveragePool";
            metadata_type_value = "averagepool";
        } else if (lower_op == "globalaveragepool") {
            op_type = "GlobalAveragePool";
            metadata_type_value = "averagepool";
        } else if (lower_op == "attention") {
            op_type = "ATTENTION";
            metadata_type_value = "attention";
        } else {
            supported_op = false;
        }

        if (!supported_op) {
            continue;
        }

        ModelSegment op_segment;
        op_segment.type = SegmentType::METADATA_JSON;
        op_segment.layer_type = op_type;
        op_segment.layer_index = node_index;
        op_segment.layer_name = !node.name().empty() ? node.name() : ("node_" + std::to_string(node_index) + "_" + op_type);
        op_segment.name = "op." + op_segment.layer_name;
        op_segment.data_format = "ONNX_OP";

        if (!node.input().empty()) {
            const auto input_it = tensor_shapes.find(node.input(0));
            if (input_it != tensor_shapes.end()) {
                op_segment.input_shape = input_it->second;
            }
        }
        if (!node.output().empty()) {
            const auto output_it = tensor_shapes.find(node.output(0));
            if (output_it != tensor_shapes.end()) {
                op_segment.output_shape = output_it->second;
            }
        }

        TensorMetadata op_metadata;
        op_metadata.is_sorted = true;
        op_metadata.sparsity_ratio = 0.0f;
        if (!op_segment.output_shape.empty()) {
            op_metadata.dimensions = op_segment.output_shape;
        } else if (!op_segment.input_shape.empty()) {
            op_metadata.dimensions = op_segment.input_shape;
        }
        op_segment.tensor_metadata = op_metadata;

        std::ostringstream meta_stream;
        meta_stream << "type " << metadata_type_value << " ";
        meta_stream << "op_type " << op_type << " ";
        if (op_type == "Relu") {
            meta_stream << "activation relu ";
        } else if (op_type == "Sigmoid") {
            meta_stream << "activation sigmoid ";
        } else if (op_type == "Tanh") {
            meta_stream << "activation tanh ";
        } else if (op_type == "Softmax") {
            meta_stream << "activation softmax ";
        } else if (op_type == "Gelu") {
            meta_stream << "activation gelu ";
        } else if (op_type == "LeakyRelu") {
            meta_stream << "activation leaky_relu ";
        } else if (op_type == "Elu") {
            meta_stream << "activation elu ";
        } else if (op_type == "Silu") {
            meta_stream << "activation silu ";
        } else if (op_type == "MaxPool") {
            meta_stream << "activation max ";
        } else if (op_type == "AveragePool" || op_type == "GlobalAveragePool") {
            meta_stream << "activation avg ";
        }

        for (const auto& attr : node.attribute()) {
            if (attr.type() == onnx::AttributeProto::INT) {
                if (attr.name() == "axis") {
                    meta_stream << "axis " << attr.i() << " ";
                }
            } else if (attr.type() == onnx::AttributeProto::FLOAT) {
                if (attr.name() == "alpha") {
                    meta_stream << "alpha " << attr.f() << " ";
                } else if (attr.name() == "beta") {
                    meta_stream << "beta " << attr.f() << " ";
                } else if (attr.name() == "epsilon") {
                    meta_stream << "epsilon " << attr.f() << " ";
                }
            }

            if (attr.type() == onnx::AttributeProto::INTS && attr.ints_size() > 0) {
                if (attr.name() == "kernel_shape" || attr.name() == "strides") {
                    meta_stream << attr.name() << " ";
                    for (int index = 0; index < attr.ints_size(); ++index) {
                        if (index > 0) {
                            meta_stream << ",";
                        }
                        meta_stream << attr.ints(index);
                    }
                    meta_stream << " ";
                } else if (attr.name() == "perm") {
                    meta_stream << "perm ";
                    for (int index = 0; index < attr.ints_size(); ++index) {
                        if (index > 0) {
                            meta_stream << ",";
                        }
                        meta_stream << attr.ints(index);
                    }
                    meta_stream << " ";
                } else if (attr.name() == "starts" || attr.name() == "ends" || attr.name() == "axes") {
                    meta_stream << attr.name() << " ";
                    for (int index = 0; index < attr.ints_size(); ++index) {
                        if (index > 0) {
                            meta_stream << ",";
                        }
                        meta_stream << attr.ints(index);
                    }
                    meta_stream << " ";
                } else if (attr.name() == "pads") {
                    meta_stream << "padding ";
                    const int pad_count = std::min(2, attr.ints_size());
                    for (int index = 0; index < pad_count; ++index) {
                        if (index > 0) {
                            meta_stream << ",";
                        }
                        meta_stream << attr.ints(index);
                    }
                    meta_stream << " ";
                }
            }
        }

        const std::string payload = meta_stream.str();
        op_segment.data.assign(
            reinterpret_cast<const std::byte*>(payload.data()),
            reinterpret_cast<const std::byte*>(payload.data() + payload.size())
        );
        op_segment.original_size = op_segment.data.size();
        segments.push_back(std::move(op_segment));
    }
}

void ONNXModelParser::processInitializers(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const {
    const auto tensor_to_metadata = buildTensorMetadataMap(graph_proto);
    const auto tensor_shapes = buildTensorShapeMap(graph_proto);
    const auto tensor_to_node = buildInitializerNodeMap(graph_proto);

    auto make_segment = [&](const onnx::TensorProto& tensor_proto) {
        ModelSegment segment;
        segment.name = tensor_proto.name();
        segment.type = onnxTensorTypeToSegmentType(tensor_proto.data_type());
        if (segment.type == SegmentType::WEIGHTS_FP32) {
            const SegmentType name_based_type = determineSegmentType(tensor_proto.name());
            if (name_based_type != SegmentType::WEIGHTS_FP32) {
                segment.type = name_based_type;
            }
        }
        segment.data = tensorProtoToBytes(tensor_proto);
        segment.original_size = segment.data.size();
        segment.tensor_metadata = extractTensorMetadataProto(tensor_proto);
        segment.layer_name = extractLayerName(tensor_proto.name());
        segment.layer_index = extractLayerIndex(tensor_proto.name());
        segment.data_format = "ONNX";

        const auto node_it = tensor_to_node.find(tensor_proto.name());
        if (node_it != tensor_to_node.end() && node_it->second != nullptr) {
            const onnx::NodeProto& node = *node_it->second;
            if (segment.layer_name.empty() || segment.layer_name == segment.name) {
                if (!node.name().empty()) {
                    segment.layer_name = node.name();
                } else if (!node.output().empty()) {
                    segment.layer_name = extractLayerName(node.output(0));
                }
            }
            if (segment.layer_index == 0 && !segment.layer_name.empty()) {
                segment.layer_index = extractLayerIndex(segment.layer_name);
            }

            segment.layer_type = node.op_type();
            if (segment.type == SegmentType::WEIGHTS_FP32) {
                const std::string& type_hint_name = !segment.layer_name.empty() ? segment.layer_name : segment.name;
                segment.type = determineSegmentType(type_hint_name);
            }

            if (!node.output().empty()) {
                const auto metadata_it = tensor_to_metadata.find(node.output(0));
                if (metadata_it != tensor_to_metadata.end()) {
                    segment.data_format = metadata_it->second;
                }
                const auto output_shape_it = tensor_shapes.find(node.output(0));
                if (output_shape_it != tensor_shapes.end()) {
                    segment.output_shape = output_shape_it->second;
                }
            }
            if (!node.input().empty()) {
                const auto input_shape_it = tensor_shapes.find(node.input(0));
                if (input_shape_it != tensor_shapes.end()) {
                    segment.input_shape = input_shape_it->second;
                }
            }
        }

        return segment;
    };

    segments.reserve(segments.size() + static_cast<size_t>(graph_proto.initializer_size()));
    for (const auto& tensor_proto : graph_proto.initializer()) {
        segments.push_back(make_segment(tensor_proto));
    }
}

void ONNXModelParser::processInitializersParallel(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const {
    const int initializer_count = graph_proto.initializer_size();
    if (initializer_count <= 0) {
        return;
    }
    if (initializer_count < 8 || std::thread::hardware_concurrency() <= 1U) {
        processInitializers(graph_proto, segments);
        return;
    }

    const auto tensor_to_metadata = buildTensorMetadataMap(graph_proto);
    const auto tensor_shapes = buildTensorShapeMap(graph_proto);
    const auto tensor_to_node = buildInitializerNodeMap(graph_proto);

    auto make_segment = [&](const onnx::TensorProto& tensor_proto) {
        ModelSegment segment;
        segment.name = tensor_proto.name();
        segment.type = onnxTensorTypeToSegmentType(tensor_proto.data_type());
        if (segment.type == SegmentType::WEIGHTS_FP32) {
            const SegmentType name_based_type = determineSegmentType(tensor_proto.name());
            if (name_based_type != SegmentType::WEIGHTS_FP32) {
                segment.type = name_based_type;
            }
        }
        segment.data = tensorProtoToBytes(tensor_proto);
        segment.original_size = segment.data.size();
        segment.tensor_metadata = extractTensorMetadataProto(tensor_proto);
        segment.layer_name = extractLayerName(tensor_proto.name());
        segment.layer_index = extractLayerIndex(tensor_proto.name());
        segment.data_format = "ONNX";

        const auto node_it = tensor_to_node.find(tensor_proto.name());
        if (node_it != tensor_to_node.end() && node_it->second != nullptr) {
            const onnx::NodeProto& node = *node_it->second;
            if (segment.layer_name.empty() || segment.layer_name == segment.name) {
                if (!node.name().empty()) {
                    segment.layer_name = node.name();
                } else if (!node.output().empty()) {
                    segment.layer_name = extractLayerName(node.output(0));
                }
            }
            if (segment.layer_index == 0 && !segment.layer_name.empty()) {
                segment.layer_index = extractLayerIndex(segment.layer_name);
            }

            segment.layer_type = node.op_type();
            if (segment.type == SegmentType::WEIGHTS_FP32) {
                const std::string& type_hint_name = !segment.layer_name.empty() ? segment.layer_name : segment.name;
                segment.type = determineSegmentType(type_hint_name);
            }

            if (!node.output().empty()) {
                const auto metadata_it = tensor_to_metadata.find(node.output(0));
                if (metadata_it != tensor_to_metadata.end()) {
                    segment.data_format = metadata_it->second;
                }
                const auto output_shape_it = tensor_shapes.find(node.output(0));
                if (output_shape_it != tensor_shapes.end()) {
                    segment.output_shape = output_shape_it->second;
                }
            }
            if (!node.input().empty()) {
                const auto input_shape_it = tensor_shapes.find(node.input(0));
                if (input_shape_it != tensor_shapes.end()) {
                    segment.input_shape = input_shape_it->second;
                }
            }
        }

        return segment;
    };

    const size_t max_workers = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t worker_count = std::min(max_workers, static_cast<size_t>(initializer_count));
    const size_t chunk_size = (static_cast<size_t>(initializer_count) + worker_count - 1) / worker_count;

    std::vector<std::future<std::vector<ModelSegment>>> futures;
    futures.reserve(worker_count);

    for (size_t worker = 0; worker < worker_count; ++worker) {
        const size_t start_index = worker * chunk_size;
        const size_t end_index = std::min(start_index + chunk_size, static_cast<size_t>(initializer_count));
        if (start_index >= end_index) {
            continue;
        }

        futures.push_back(std::async(std::launch::async, [&, start_index, end_index]() {
            std::vector<ModelSegment> local_segments;
            local_segments.reserve(end_index - start_index);
            for (size_t index = start_index; index < end_index; ++index) {
                local_segments.push_back(make_segment(graph_proto.initializer(static_cast<int>(index))));
            }
            return local_segments;
        }));
    }

    segments.reserve(segments.size() + static_cast<size_t>(initializer_count));
    for (auto& future : futures) {
        std::vector<ModelSegment> local_segments = future.get();
        for (auto& segment : local_segments) {
            segments.push_back(std::move(segment));
        }
    }
}
#endif

} // namespace CortexAICompression
