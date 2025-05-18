#include "ONNXModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream> // For string stream serialization
#include <fstream> // For file reading
#include <unordered_set> // For initializer check
#include <future>
#include <thread>

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
        float prev_value = std::numeric_limits<float>::lowest();

        // Handle different data types
        switch (tensor.data_type()) {
            case onnx::TensorProto::FLOAT: {
                if (tensor.has_raw_data()) {
                    const float* data = reinterpret_cast<const float*>(tensor.raw_data().data());
                    for (size_t i = 0; i < total_elements; ++i) {
                        if (data[i] == 0.0f) {
                            ++num_zeros;
                        }
                        if (i > 0 && data[i] < prev_value) {
                            is_sorted = false;
                        }
                        prev_value = data[i];
                    }
                } else {
                    for (int i = 0; i < tensor.float_data_size(); ++i) {
                        float val = tensor.float_data(i);
                        if (val == 0.0f) {
                            ++num_zeros;
                        }
                        if (i > 0 && val < prev_value) {
                            is_sorted = false;
                        }
                        prev_value = val;
                    }
                }
                break;
            }
            // Handle other data types similarly...
            default:
                // For unknown types, just estimate
                num_zeros = total_elements / 2; // Placeholder
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
            for (int i = 0; i < tensor.float_data_size(); ++i) {
                float_data[i] = tensor.float_data(i);
            }
            break;
        }
        // Handle other data types similarly...
        default:
            std::cerr << "Warning: Unsupported tensor data type for conversion to bytes: " 
                      << tensor.data_type() << std::endl;
    }
    
    return data;
}
#endif // ENABLE_ONNX_PROTOBUF


#ifdef ENABLE_ONNX
SegmentType ONNXModelParser::determineSegmentType(const std::string& tensorName, const Ort::Value& tensor) const {
    // Determine segment type based on tensor name and properties
    ONNXTensorElementDataType element_type = tensor.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType();
    
    switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return SegmentType::WEIGHTS_FP32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return SegmentType::WEIGHTS_FP16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
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
                float prev_value = std::numeric_limits<float>::lowest();
                
                for (size_t i = 0; i < total_elements; ++i) {
                    if (data[i] == 0.0f) {
                        ++num_zeros;
                    }
                    if (i > 0 && data[i] < prev_value) {
                        is_sorted = false;
                    }
                    prev_value = data[i];
                }
                break;
            }
            // Handle other data types similarly...
            default:
                // For unknown types, just estimate
                num_zeros = total_elements / 2; // Placeholder
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
        // Handle other data types similarly...
        default:
            std::cerr << "Warning: Unsupported tensor data type for conversion to bytes: " 
                      << element_type << std::endl;
    }
    
    return data;
}
#endif

std::string ONNXModelParser::extractLayerName(const std::string& tensorName) const {
    std::regex layer_pattern(R"(layers\.(\d+))");
    std::smatch matches;
    if (std::regex_search(tensorName, matches, layer_pattern)) {
        return "layer_" + matches[1].str();
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
    std::vector<ModelSegment> segments;
    
#ifdef ENABLE_ONNX_PROTOBUF
    // Use a simpler approach for parsing ONNX models, similar to your onnx_cpp_serializer project
    onnx::ModelProto model_proto;
    
    // Open and read the model file
    std::ifstream model_file(modelPath, std::ios::binary);
    if (!model_file) {
        throw std::runtime_error("Failed to open model file: " + modelPath);
    }
    
    // Parse the model using Protobuf's ParseFromIstream
    if (!model_proto.ParseFromIstream(&model_file)) {
        throw std::runtime_error("Failed to parse model file using Protobuf");
    }
    
    std::cout << "Successfully parsed ONNX model using Protobuf" << std::endl;
    std::cout << "Model IR Version: " << model_proto.ir_version() << std::endl;
    if (!model_proto.producer_name().empty()) {
        std::cout << "Producer: " << model_proto.producer_name();
        if (!model_proto.producer_version().empty()) {
            std::cout << " (" << model_proto.producer_version() << ")";
        }
        std::cout << std::endl;
    }
    
    // Process model metadata
    ModelSegment meta_segment;
    meta_segment.name = "model_metadata";
    meta_segment.type = SegmentType::METADATA_JSON;
    
    std::ostringstream metadata_ss;
    metadata_ss << "Producer: " << (model_proto.has_producer_name() ? model_proto.producer_name() : "N/A") << "\n";
    metadata_ss << "Domain: " << (model_proto.has_domain() ? model_proto.domain() : "N/A") << "\n";
    metadata_ss << "IRVersion: " << model_proto.ir_version() << "\n";
    
    if (model_proto.opset_import_size() > 0) {
        metadata_ss << "OpsetVersion: " << model_proto.opset_import(0).version() << "\n";
    }
    
    std::string metadata_str = metadata_ss.str();
    if (!metadata_str.empty()) {
        meta_segment.data.assign(
            reinterpret_cast<const std::byte*>(metadata_str.data()),
            reinterpret_cast<const std::byte*>(metadata_str.data() + metadata_str.size())
        );
        meta_segment.original_size = meta_segment.data.size();
        segments.push_back(std::move(meta_segment));
        std::cout << "Added metadata segment (" << meta_segment.original_size << " bytes)." << std::endl;
    }
    
    // Process graph structure
    const auto& graph_proto = model_proto.graph();
    
    // Create a segment for graph structure
    ModelSegment graph_segment;
    graph_segment.name = "graph_structure";
    graph_segment.type = SegmentType::GRAPH_STRUCTURE_PROTO;
    
    // Serialize the graph structure
    std::string serialized_data;
    if (!graph_proto.SerializeToString(&serialized_data)) {
        throw std::runtime_error("Failed to serialize graph structure");
    }
    
    std::cout << "Successfully serialized graph structure: " << serialized_data.size() << " bytes" << std::endl;
    
    // Create the segment with the serialized data
    graph_segment.data.assign(
        reinterpret_cast<const std::byte*>(serialized_data.data()),
        reinterpret_cast<const std::byte*>(serialized_data.data() + serialized_data.size())
    );
    graph_segment.original_size = graph_segment.data.size();
    graph_segment.data_format = "protobuf";
    
    // Store metadata in tensor_metadata
    TensorMetadata metadata;
    metadata.dimensions = {
        static_cast<size_t>(graph_proto.input_size()),
        static_cast<size_t>(graph_proto.output_size()),
        static_cast<size_t>(graph_proto.node_size()),
        static_cast<size_t>(graph_proto.initializer_size())
    };
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = true;
    metadata.scale = static_cast<float>(model_proto.ir_version());
    metadata.zero_point = 0.0f;
    
    graph_segment.tensor_metadata = metadata;
    segments.push_back(std::move(graph_segment));
    
    std::cout << "Successfully added graph structure segment" << std::endl;
    printSegmentMetadata(segments.back());
    
    // Process initializers (weights)
    std::cout << "Processing " << graph_proto.initializer_size() << " initializers..." << std::endl;
    
    for (int i = 0; i < graph_proto.initializer_size(); ++i) {
        const auto& initializer = graph_proto.initializer(i);
        if (!initializer.has_name()) {
            std::cerr << "Warning: Skipping initializer without name at index " << i << std::endl;
            continue;
        }
        
        ModelSegment weight_segment;
        weight_segment.name = initializer.name();
        weight_segment.layer_name = extractLayerName(initializer.name());
        weight_segment.layer_index = extractLayerIndex(initializer.name());
        weight_segment.type = onnxTensorTypeToSegmentType(initializer.data_type());
        
        if (weight_segment.type == SegmentType::UNKNOWN) {
            std::cerr << "Warning: Unknown segment type for initializer " << initializer.name() << std::endl;
            continue;
        }
        
        weight_segment.tensor_metadata = extractTensorMetadataProto(initializer);
        weight_segment.data = tensorProtoToBytes(initializer);
        weight_segment.original_size = weight_segment.data.size();
        
        if (weight_segment.original_size > 0) {
            segments.push_back(std::move(weight_segment));
            std::cout << "Added weight segment for " << initializer.name() 
                      << " (" << weight_segment.original_size << " bytes)." << std::endl;
        } else {
            std::cerr << "Warning: Empty data for initializer " << initializer.name() << std::endl;
        }
    }
    
    return segments;
#else
    throw std::runtime_error("ONNX model support is disabled. Please enable ENABLE_ONNX_PROTOBUF to use this feature.");
#endif
}

std::vector<ModelSegment> ONNXModelParser::parseWithChunking(const std::string& modelPath) const {
    // For simplicity, just call the regular parse method
    return parse(modelPath);
}

void ONNXModelParser::processMetadata(const onnx::ModelProto& model_proto, std::vector<ModelSegment>& segments) const {
    // This is now handled directly in the parse method
}

void ONNXModelParser::processGraphStructure(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const {
    // This is now handled directly in the parse method
}

void ONNXModelParser::processInitializers(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const {
    // This is now handled directly in the parse method
}

void ONNXModelParser::processInitializersParallel(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const {
    // This is now handled directly in the parse method
}

} // namespace CortexAICompression
