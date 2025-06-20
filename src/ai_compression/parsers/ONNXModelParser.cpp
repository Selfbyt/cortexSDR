#include "ONNXModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>
#include <iostream> // For potential debug output
#include <iomanip> // For std::setw, std::setfill
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
    
    // Process model metadata with ultra-aggressive compression
    ModelSegment meta_segment;
    meta_segment.name = "model_metadata";
    meta_segment.type = SegmentType::METADATA_JSON;
    
    // Set an extremely high sparsity ratio for the metadata
    TensorMetadata metaData;
    metaData.sparsity_ratio = 0.0000001f; // Ultra-aggressive sparsity (0.00001%)
    meta_segment.tensor_metadata = metaData;
    
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
    
    // Create a segment for the entire model structure (not just graph)
    ModelSegment graph_segment;
    graph_segment.name = "model_structure";
    graph_segment.type = SegmentType::GRAPH_STRUCTURE_PROTO;
    
    // Convert the model structure to a binary format more suitable for SDR compression
    // First, set tensor metadata to help the SDR strategy
    TensorMetadata structMetadata;
    structMetadata.dimensions = {1, 13, 2581, 332}; // Add dimensions to help the SDR strategy
    structMetadata.sparsity_ratio = 0.1f; // Higher sparsity (10%) for model structure
    graph_segment.tensor_metadata = structMetadata;
    
    // Create a new ModelProto with minimal required fields
    onnx::ModelProto minimal_model;
    minimal_model.set_ir_version(model_proto.ir_version());
    minimal_model.set_producer_name(model_proto.producer_name());
    minimal_model.set_producer_version(model_proto.producer_version());
    minimal_model.set_domain(model_proto.domain());
    minimal_model.set_model_version(model_proto.model_version());
    minimal_model.set_doc_string(model_proto.doc_string());

    // Copy graph
    *minimal_model.mutable_graph() = model_proto.graph();

    // Copy opset imports
    for (const auto& opset : model_proto.opset_import()) {
        *minimal_model.add_opset_import() = opset;
    }

    // Add metadata properties
    auto* props = minimal_model.mutable_metadata_props();
    for (const auto& prop : model_proto.metadata_props()) {
        auto* new_prop = props->Add();
        new_prop->set_key(prop.key());
        new_prop->set_value(prop.value());
        }

    // Debug output
    std::cout << "Minimal ModelProto state before serialization:" << std::endl;
    std::cout << "  IR Version: " << minimal_model.ir_version() << std::endl;
    std::cout << "  Producer: " << minimal_model.producer_name() << " (" << minimal_model.producer_version() << ")" << std::endl;
    std::cout << "  Domain: " << minimal_model.domain() << std::endl;
    std::cout << "  Model Version: " << minimal_model.model_version() << std::endl;
    std::cout << "  Opset Imports: " << minimal_model.opset_import_size() << std::endl;
    std::cout << "  Has Graph: " << (minimal_model.has_graph() ? "yes" : "no") << std::endl;
    if (minimal_model.has_graph()) {
        const auto& graph = minimal_model.graph();
        std::cout << "  Graph Name: " << graph.name() << std::endl;
        std::cout << "  Nodes: " << graph.node_size() << std::endl;
        std::cout << "  Inputs: " << graph.input_size() << std::endl;
        std::cout << "  Outputs: " << graph.output_size() << std::endl;
        std::cout << "  Initializers: " << graph.initializer_size() << std::endl;
    }
    
    // Try to serialize the minimal model using SerializeToString first
    std::string serialized_data;
    bool success = false;
    try {
        std::cout << "Attempting SerializeToString on minimal ModelProto..." << std::endl;
        if (minimal_model.SerializeToString(&serialized_data)) {
            std::cout << "SerializeToString on ModelProto succeeded. Size: " << serialized_data.size() << " bytes." << std::endl;
            success = true;
        } else {
            std::cerr << "SerializeToString on ModelProto failed. Trying GraphProto..." << std::endl;
            // Try serializing just the graph
            serialized_data.clear();
            if (minimal_model.has_graph() && minimal_model.graph().SerializeToString(&serialized_data)) {
                std::cout << "SerializeToString on GraphProto succeeded. Size: " << serialized_data.size() << " bytes." << std::endl;
                success = true;
            } else {
                std::cerr << "SerializeToString on GraphProto also failed." << std::endl;
            }
        }

        if (success && !serialized_data.empty()) {
            // Add model structure segment
    graph_segment.data.assign(
        reinterpret_cast<const std::byte*>(serialized_data.data()),
        reinterpret_cast<const std::byte*>(serialized_data.data() + serialized_data.size())
    );
    graph_segment.original_size = graph_segment.data.size();
    segments.push_back(std::move(graph_segment));
            std::cout << "Added model structure segment (" << graph_segment.original_size << " bytes)." << std::endl;
        } else {
            std::cerr << "Model structure serialization failed: produced empty data." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during model structure serialization: " << e.what() << std::endl;
    }
    
    // Map tensor names to their producing node's op_type and attributes
    std::unordered_map<std::string, std::string> tensor_to_op_type;
    std::unordered_map<std::string, std::string> tensor_to_metadata;
    for (const auto& node : graph_proto.node()) {
        for (const auto& output_name : node.output()) {
            tensor_to_op_type[output_name] = node.op_type();
            // Serialize all attributes for this node
            std::ostringstream meta;
            meta << "op_type " << node.op_type() << " ";
            for (const auto& attr : node.attribute()) {
                meta << attr.name() << " ";
                switch (attr.type()) {
                    case onnx::AttributeProto::INT:
                        meta << attr.i() << " ";
                        break;
                    case onnx::AttributeProto::FLOAT:
                        meta << attr.f() << " ";
                        break;
                    case onnx::AttributeProto::STRING:
                        meta << attr.s() << " ";
                        break;
                    case onnx::AttributeProto::INTS:
                        for (int i = 0; i < attr.ints_size(); ++i) {
                            meta << attr.ints(i);
                            if (i + 1 < attr.ints_size()) meta << ",";
                        }
                        meta << " ";
                        break;
                    case onnx::AttributeProto::FLOATS:
                        for (int i = 0; i < attr.floats_size(); ++i) {
                            meta << attr.floats(i);
                            if (i + 1 < attr.floats_size()) meta << ",";
                        }
                        meta << " ";
                        break;
                    case onnx::AttributeProto::STRINGS:
                        for (int i = 0; i < attr.strings_size(); ++i) {
                            meta << attr.strings(i);
                            if (i + 1 < attr.strings_size()) meta << ",";
                        }
                        meta << " ";
                        break;
                    default:
                        break;
                }
            }
            tensor_to_metadata[output_name] = meta.str();
        }
    }

    // Build a map from tensor name to its shape from ValueInfoProto
    std::unordered_map<std::string, std::vector<size_t>> tensor_shapes;
    for (const auto& value_info : graph_proto.value_info()) {
        std::vector<size_t> shape;
        if (value_info.has_type() && value_info.type().has_tensor_type() && value_info.type().tensor_type().has_shape()) {
            const auto& shape_proto = value_info.type().tensor_type().shape();
            for (const auto& dim : shape_proto.dim()) {
                if (dim.has_dim_value()) shape.push_back(static_cast<size_t>(dim.dim_value()));
            }
        }
        tensor_shapes[value_info.name()] = shape;
    }
    // Also add input/output shapes from graph inputs/outputs
    for (const auto& input : graph_proto.input()) {
        std::vector<size_t> shape;
        if (input.has_type() && input.type().has_tensor_type() && input.type().tensor_type().has_shape()) {
            const auto& shape_proto = input.type().tensor_type().shape();
            for (const auto& dim : shape_proto.dim()) {
                if (dim.has_dim_value()) shape.push_back(static_cast<size_t>(dim.dim_value()));
            }
        }
        tensor_shapes[input.name()] = shape;
    }
    for (const auto& output : graph_proto.output()) {
        std::vector<size_t> shape;
        if (output.has_type() && output.type().has_tensor_type() && output.type().tensor_type().has_shape()) {
            const auto& shape_proto = output.type().tensor_type().shape();
            for (const auto& dim : shape_proto.dim()) {
                if (dim.has_dim_value()) shape.push_back(static_cast<size_t>(dim.dim_value()));
            }
        }
        tensor_shapes[output.name()] = shape;
    }

    // Process initializers (weights, biases)
    for (const auto& tensor_proto : graph_proto.initializer()) {
        ModelSegment segment;
        segment.name = tensor_proto.name();
        segment.type = onnxTensorTypeToSegmentType(tensor_proto.data_type());
        segment.data = tensorProtoToBytes(tensor_proto);
        segment.original_size = segment.data.size();
        segment.tensor_metadata = extractTensorMetadataProto(tensor_proto);

        // Assign the op_type and metadata from the node that uses this initializer as input
        for (const auto& node : graph_proto.node()) {
            for (const auto& input_name : node.input()) {
                if (input_name == tensor_proto.name()) {
                    segment.layer_type = node.op_type();
                    // Use the first output as key for metadata
                    if (!node.output().empty() && tensor_to_metadata.count(node.output(0))) {
                        segment.data_format = tensor_to_metadata[node.output(0)];
                    }
                    // Store true input/output shapes for this node
                    if (!node.input().empty() && tensor_shapes.count(node.input(0))) {
                        segment.input_shape = tensor_shapes[node.input(0)];
                    }
                    if (!node.output().empty() && tensor_shapes.count(node.output(0))) {
                        segment.output_shape = tensor_shapes[node.output(0)];
                    }
                    goto found_node;
                }
            }
        }
        found_node:;

        segments.push_back(segment);
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
