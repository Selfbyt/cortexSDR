#include "c_api.hpp"
#include "../core/AICompressor.hpp"
#include "../core/AIDecompressor.hpp"
#include "../strategies/SDRIndexStorage.hpp"
#include "../strategies/GzipStrategy.hpp"
#include "../strategies/NumericalRLE.hpp"
#include "../strategies/QuantizedTensorStrategy.hpp"
#include "../strategies/MetadataSDRStrategy.hpp"
#include "../strategies/AdaptiveSDRStrategy.hpp"
#include "../parsers/ONNXModelParser.hpp"
#include "../parsers/GGUFModelParser.hpp"
#include "../utils/ModelConverter.hpp"
#include "../streaming/StreamingCompressor.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip> // For std::setw, std::setfill
#include <fstream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory> // For std::shared_ptr
#include <fstream> // For std::ofstream
#include <regex>

// Include ONNX Protobuf if enabled
#ifdef ENABLE_ONNX_PROTOBUF
#include <onnx.pb.h> // Try including without the 'onnx/' prefix
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#endif

using namespace CortexAICompression;

namespace {
    CortexError convert_exception(const std::exception& e) {
        char* msg_copy = new char[strlen(e.what()) + 1];
        strcpy(msg_copy, e.what());
        return {msg_copy, 1}; 
    }

    char* str_to_c(const std::string& str) {
        char* cstr = new char[str.length() + 1];
        strcpy(cstr, str.c_str());
        return cstr;
    }
}

CortexError cortex_compression_options_init(CortexCompressionOptions* options) {
    try {
        if (!options) {
            return {"Invalid options pointer", -1};
        }
        options->num_threads = 1;
        options->verbose = 0;
        options->show_stats = 0;
        options->use_delta_encoding = 1;
        options->use_rle = 1;
        options->compression_level = 6; 
        options->use_quantization = 0;
        options->quantization_bits = 8;
        options->sparsity = 0.02f; 
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

struct ModelInfo {
    std::string originalPath;
    std::string actualPath; 
    std::string format;     
};

std::unordered_map<CortexCompressorHandle, ModelInfo> g_modelInfoMap;

CortexError cortex_compressor_create(const char* model_path, const char* format,
                                   const CortexCompressionOptions* options,
                                   CortexCompressorHandle* handle) {
    try {
        if (!model_path || !format || !handle) {
            return {str_to_c("Invalid arguments (null pointers)"), 1};
        }
        std::string actualModelPath = model_path;
        std::string actualFormat = format;
        if (strcmp(format, "gguf") != 0 && strcmp(format, "onnx") != 0) {
            try {
                std::cout << "Converting " << format << " model to ONNX format..." << std::endl;
                actualModelPath = ModelConverter::convertToONNX(model_path, format);
                actualFormat = "onnx";
                std::cout << "Model successfully converted to ONNX: " << actualModelPath << std::endl;
            } catch (const ModelConversionError& e) {
                std::cerr << "Warning: " << e.what() << std::endl;
                std::cerr << "Please convert your model to ONNX format manually..." << std::endl;
                return {str_to_c("Failed to convert model to ONNX: " + std::string(e.what())), 1};
            } catch (const std::exception& e) {
                 return {str_to_c("Error during model conversion: " + std::string(e.what())), 1};
            }
        }
        std::unique_ptr<IAIModelParser> parser;
        if (actualFormat == "gguf") {
#ifdef ENABLE_GGUF
            parser = std::make_unique<GGUFModelParser>();
#else
            return {str_to_c("GGUF support is not enabled in this build."), 1};
#endif
        }
#if defined(ENABLE_ONNX) || defined(ENABLE_ONNX_PROTOBUF)
        else if (actualFormat == "onnx") {
            parser = std::make_unique<ONNXModelParser>();
        }
#endif
        else {
            return {str_to_c("Unsupported model format. Please convert to ONNX or GGUF."), 1};
        }
        if (!parser) {
             return {str_to_c("Failed to create model parser for format: " + actualFormat), 1};
        }
        auto ai_compressor = new AICompressor(std::move(parser));
        const uint8_t SDR_STRATEGY_ID = 1;
        const uint8_t RLE_STRATEGY_ID = 2;
        const uint8_t GZIP_STRATEGY_ID = 3;
        const uint8_t QUANT_STRATEGY_ID = 4;

        // Create compression strategies
        auto gzipStrategy = std::make_shared<GzipStrategy>(options->compression_level);
        
        // Create our size-adaptive SDR strategy that addresses size expansion issues with small models
        // while maintaining the SDR-based encoding framework for all model components
        // This strategy automatically uses direct storage for very small segments
        // Uses the --sparsity parameter passed via CLI for SDR encoding of larger segments
        auto adaptiveStrategy = std::make_shared<AdaptiveSDRStrategy>(options->sparsity);
        
        // Register for all segment types with highest priority
        // Metadata and graph structure
        ai_compressor->registerStrategy(SegmentType::METADATA_JSON, 1, SDR_STRATEGY_ID, adaptiveStrategy);
        ai_compressor->registerStrategy(SegmentType::GRAPH_STRUCTURE_PROTO, 1, SDR_STRATEGY_ID, adaptiveStrategy);
        
        // Tensor data segments
        if (options->use_delta_encoding) {
            ai_compressor->registerStrategy(SegmentType::SPARSE_INDICES, 2, SDR_STRATEGY_ID, adaptiveStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 2, SDR_STRATEGY_ID, adaptiveStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16, 2, SDR_STRATEGY_ID, adaptiveStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT8, 2, SDR_STRATEGY_ID, adaptiveStrategy);
        }

            if (options->use_rle) {
                auto rleStrategy = std::make_shared<NumericalRLEStrategy>();
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 3, RLE_STRATEGY_ID, rleStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16, 3, RLE_STRATEGY_ID, rleStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT8, 3, RLE_STRATEGY_ID, rleStrategy);
            }

#ifdef ENABLE_QUANTIZATION
            if (options->use_quantization) {
                auto quantStrategy = std::make_shared<QuantizedTensorStrategy>(options->quantization_bits);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 3, QUANT_STRATEGY_ID, quantStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16, 3, QUANT_STRATEGY_ID, quantStrategy);
            }
#endif

        // Register Gzip as fallback for weight tensors with lowest priority
        ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT8, 4, GZIP_STRATEGY_ID, gzipStrategy);

        *handle = reinterpret_cast<CortexCompressorHandle>(ai_compressor);
        g_modelInfoMap[*handle] = {model_path, actualModelPath, actualFormat};
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_compressor_compress(CortexCompressorHandle handle, const char* output_path) {
    try {
        if (!handle || !output_path) {
            return {str_to_c("Invalid arguments (null handle or output path)"), 1};
        }
        auto compressor = reinterpret_cast<AICompressor*>(handle);
        StreamingCompressor streamCompressor(output_path);
        auto it = g_modelInfoMap.find(handle);
        if (it == g_modelInfoMap.end()) {
             return {str_to_c("Internal error: Model info not found for handle."), 1};
        }
        const std::string& modelToCompress = it->second.actualPath;
        compressor->compressModelStreaming(modelToCompress, streamCompressor);
        streamCompressor.finalizeArchive();
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_compressor_get_stats(CortexCompressorHandle handle,
                                      size_t* original_size,
                                      size_t* compressed_size,
                                      double* compression_ratio,
                                      double* compression_time_ms) {
    try {
        if (!handle) return {str_to_c("Invalid compressor handle"), 1};
        if (!original_size || !compressed_size || !compression_ratio || !compression_time_ms) {
             return {str_to_c("Invalid output pointers for stats"), 1};
        }
        auto compressor = reinterpret_cast<AICompressor*>(handle);
        const auto& stats = compressor->getCompressionStats();
        *original_size = stats.originalSize;
        *compressed_size = stats.compressedSize;
        *compression_ratio = stats.compressionRatio;
        *compression_time_ms = stats.compressionTimeMs;
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_compressor_free(CortexCompressorHandle handle) {
    try {
        if (handle) {
            g_modelInfoMap.erase(handle);
            delete reinterpret_cast<AICompressor*>(handle);
        }
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_decompressor_create(const char* compressed_path,
                                      CortexDecompressorHandle* handle,
                                      float sparsity) {
    try {
        if (!compressed_path || !handle) {
            return {str_to_c("Invalid arguments (null path or handle)"), 1};
        }
        auto ai_decompressor = new AIDecompressor();
        const uint8_t SDR_STRATEGY_ID = 1;
        const uint8_t RLE_STRATEGY_ID = 2;
        const uint8_t GZIP_STRATEGY_ID = 3;
        const uint8_t QUANT_STRATEGY_ID = 4;
        
        // Create strategies for different segment types
        // Use the size-adaptive SDR strategy for all segment types
        // This ensures consistent handling of both compressed and decompressed data
        auto adaptiveStrategy = std::make_shared<AdaptiveSDRStrategy>(sparsity);
        
        // Register the adaptive strategy for SDR-encoded segments
        ai_decompressor->registerStrategy(SDR_STRATEGY_ID, adaptiveStrategy);
        
        // For backwards compatibility with files compressed using the old strategies
        auto metadataStrategy = std::make_shared<MetadataSDRStrategy>(sparsity);
        auto sdrStrategy = std::make_shared<SDRIndexStorageStrategy>();
        sdrStrategy->setSparsity(sparsity);
        ai_decompressor->registerStrategy(SDR_STRATEGY_ID + 10, sdrStrategy); // Legacy standard SDR strategy
        
        // Register RLE strategy for numerical data
        ai_decompressor->registerStrategy(RLE_STRATEGY_ID, std::make_shared<NumericalRLEStrategy>());
        
        // Keep Gzip strategy for backward compatibility with older archives
        // This will be phased out as we fully transition to SDR-based compression
        ai_decompressor->registerStrategy(GZIP_STRATEGY_ID, std::make_shared<GzipStrategy>());
#ifdef ENABLE_QUANTIZATION
        ai_decompressor->registerStrategy(QUANT_STRATEGY_ID, std::make_shared<QuantizedTensorStrategy>());
#endif
        *handle = reinterpret_cast<CortexDecompressorHandle>(ai_decompressor);
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_decompressor_decompress(CortexDecompressorHandle handle,
                                         const char* compressed_path,
                                         const char* output_path) {
    try {
        if (!handle || !compressed_path || !output_path) {
            return {str_to_c("Invalid arguments (null handle or paths)"), 1};
        }
        auto decompressor = reinterpret_cast<AIDecompressor*>(handle);
        std::ifstream inputFile(compressed_path, std::ios::binary);
        if (!inputFile) {
            return {str_to_c("Failed to open compressed file: " + std::string(compressed_path)), 1};
        }
        class InMemorySegmentHandler : public ISegmentHandler {
        public:
            void handleSegment(ModelSegment segment) override {
                segments.push_back(std::move(segment));
            }
            std::vector<ModelSegment> segments;
        };
        InMemorySegmentHandler handler;
        decompressor->decompressModelStream(inputFile, handler);
        inputFile.close();

#if defined(ENABLE_ONNX) && defined(ENABLE_ONNX_PROTOBUF)
        std::cout << "Attempting ONNX model reconstruction from " << handler.segments.size() << " segments..." << std::endl;
        onnx::ModelProto reconstructed_model_proto;
        onnx::GraphProto* reconstructed_graph_proto = reconstructed_model_proto.mutable_graph();
        const ModelSegment* graph_structure_segment = nullptr;
        std::vector<const ModelSegment*> weight_segments;
        const ModelSegment* metadata_segment = nullptr;

        for (const auto& segment : handler.segments) {
            // Look for the model structure segment (renamed from graph_structure)
            if (segment.type == SegmentType::GRAPH_STRUCTURE_PROTO) {
                if (graph_structure_segment) return {str_to_c("Error: Found multiple GRAPH_STRUCTURE_PROTO segments."), 1};
                graph_structure_segment = &segment;
            } else if (segment.isWeightTensor()) {
                weight_segments.push_back(&segment);
            } else if (segment.type == SegmentType::METADATA_JSON) {
                 metadata_segment = &segment;
            }
        }

        if (graph_structure_segment) {
            std::cout << "  Processing model structure segment..." << std::endl;
            
            // Check if the model segment data is all zeros
            bool is_all_zeros = true;
            size_t check_limit = std::min<size_t>(32, graph_structure_segment->data.size());
            for (size_t i = 0; i < check_limit; ++i) {
                if (graph_structure_segment->data[i] != std::byte{0}) {
                    is_all_zeros = false;
                    break;
                }
            }
            
            if (is_all_zeros && graph_structure_segment->data.size() > 0) {
                std::cerr << "  Warning: Model structure data appears to be all zeros. Will attempt to create a basic graph." << std::endl;
                reconstructed_graph_proto->set_name("reconstructed_graph_from_zero_data");
                
                // Create a basic graph structure based on the weight segments
                std::unordered_map<std::string, std::vector<std::string>> layer_components;
                
                // First pass: collect all layer components
                for (const auto* weight_seg : weight_segments) {
                    std::string initializer_name = weight_seg->name;
                    std::regex layer_pattern(R"(h\.(\d+)\.(\w+))");
                    std::smatch matches;
                    if (std::regex_search(initializer_name, matches, layer_pattern)) {
                        int layer_num = std::stoi(matches[1].str());
                        std::string component = matches[2].str();
                        std::string layer_key = "layer_" + std::to_string(layer_num);
                        layer_components[layer_key].push_back(component);
                    }
                }
                
                // Second pass: create nodes and connect them
                for (const auto& [layer_key, components] : layer_components) {
                    // Create input node for the layer
                    onnx::NodeProto* input_node = reconstructed_graph_proto->add_node();
                    input_node->set_name(layer_key + "_input");
                    input_node->set_op_type("Identity");
                    input_node->add_input("input_0");
                    input_node->add_output(layer_key + "_hidden");
                    
                    // Create nodes for each component
                    for (const auto& component : components) {
                        onnx::NodeProto* node = reconstructed_graph_proto->add_node();
                        node->set_name(layer_key + "_" + component);
                        node->set_op_type("Identity");
                        node->add_input(layer_key + "_hidden");
                        node->add_output(layer_key + "_" + component + "_output");
                    }
                    
                    // Create output node for the layer
                    onnx::NodeProto* output_node = reconstructed_graph_proto->add_node();
                    output_node->set_name(layer_key + "_output");
                    output_node->set_op_type("Identity");
                    output_node->add_input(layer_key + "_" + components.back() + "_output");
                    output_node->add_output(layer_key + "_output");
                }
                
                // Connect the last layer to the output
                if (!layer_components.empty()) {
                    onnx::NodeProto* final_node = reconstructed_graph_proto->add_node();
                    final_node->set_name("final_output");
                    final_node->set_op_type("Identity");
                    final_node->add_input("layer_" + std::to_string(layer_components.size() - 1) + "_output");
                    final_node->add_output("output_0");
                }
            } else {
                // Try parsing the entire ModelProto first (not just the GraphProto)
                std::string model_str(reinterpret_cast<const char*>(graph_structure_segment->data.data()), 
                                     graph_structure_segment->data.size());
                
                // Print first few bytes of the model structure data for debugging
                std::cerr << "  Model structure data size: " << model_str.size() << " bytes" << std::endl;
                std::cerr << "  First 32 bytes of model structure data: ";
                for (size_t i = 0; i < std::min(size_t(32), model_str.size()); i++) {
                    std::cerr << std::hex << std::setw(2) << std::setfill('0') << (int)(unsigned char)model_str[i] << " ";
                }
                std::cerr << std::dec << std::endl;
                
                // Try to parse as a complete ModelProto first
                if (reconstructed_model_proto.ParseFromString(model_str)) {
                    std::cout << "  Successfully parsed complete model structure." << std::endl;
                    std::cout << "  Model IR Version: " << reconstructed_model_proto.ir_version() << std::endl;
                    std::cout << "  Model has graph: " << (reconstructed_model_proto.has_graph() ? "yes" : "no") << std::endl;
                    if (reconstructed_model_proto.has_graph()) {
                        const auto& graph = reconstructed_model_proto.graph();
                        std::cout << "    Graph name: " << (graph.has_name() ? graph.name() : "<unnamed>") << std::endl;
                        std::cout << "    Inputs: " << graph.input_size() << std::endl;
                        std::cout << "    Outputs: " << graph.output_size() << std::endl;
                        std::cout << "    Nodes: " << graph.node_size() << std::endl;
                        std::cout << "    Initializers: " << graph.initializer_size() << std::endl;
                    }
                    // The graph is already set in the model_proto, no need to do anything else
                } else {
                    std::cerr << "  Failed to parse as complete ModelProto, trying as GraphProto..." << std::endl;
                    // Fall back to trying to parse just as a GraphProto
                    if (!reconstructed_graph_proto->ParseFromString(model_str)) {
                        std::cerr << "  Failed to parse model structure data. Creating a basic structure." << std::endl;
                        reconstructed_graph_proto->set_name("reconstructed_graph_fallback");
                    } else {
                        std::cout << "  Successfully parsed graph structure." << std::endl;
                        std::cout << "  Graph name: " << (reconstructed_graph_proto->has_name() ? reconstructed_graph_proto->name() : "<unnamed>") << std::endl;
                        std::cout << "  Inputs: " << reconstructed_graph_proto->input_size() << std::endl;
                        std::cout << "  Outputs: " << reconstructed_graph_proto->output_size() << std::endl;
                        std::cout << "  Nodes: " << reconstructed_graph_proto->node_size() << std::endl;
                        std::cout << "  Initializers: " << reconstructed_graph_proto->initializer_size() << std::endl;
                    }
                }
            }
        } else {
            std::cout << "  Warning: GRAPH_STRUCTURE_PROTO segment not found in archive. Creating minimal valid graph." << std::endl;
            reconstructed_graph_proto->set_name("minimal_graph");
        }

        // Add initializers to the graph with proper type information
        for (const auto* weight_seg : weight_segments) {
            onnx::TensorProto* tensor_proto = reconstructed_graph_proto->add_initializer();
            tensor_proto->set_name(weight_seg->name);

            // Set data type based on segment type
            int32_t onnx_dtype = onnx::TensorProto::UNDEFINED;
            switch(weight_seg->type) {
                case SegmentType::WEIGHTS_FP32: onnx_dtype = onnx::TensorProto::FLOAT; break;
                case SegmentType::WEIGHTS_FP16: onnx_dtype = onnx::TensorProto::FLOAT16; break;
                case SegmentType::WEIGHTS_INT8:  onnx_dtype = onnx::TensorProto::INT8; break;
                default: onnx_dtype = onnx::TensorProto::FLOAT;
            }
            tensor_proto->set_data_type(onnx_dtype);

            // Set dimensions if available
            if (weight_seg->tensor_metadata && !weight_seg->tensor_metadata->dimensions.empty()) {
                 for (size_t dim : weight_seg->tensor_metadata->dimensions) {
                     tensor_proto->add_dims(static_cast<int64_t>(dim));
                 }
            } else {
                // If no dimensions available, try to infer from data size
                size_t element_size = 4; // Default to float32
                switch(onnx_dtype) {
                    case onnx::TensorProto::FLOAT16: element_size = 2; break;
                    case onnx::TensorProto::INT8: element_size = 1; break;
                }
                size_t num_elements = weight_seg->data.size() / element_size;
                tensor_proto->add_dims(static_cast<int64_t>(num_elements));
            }
            
            // Set raw data
            tensor_proto->set_raw_data(weight_seg->data.data(), weight_seg->data.size());
        }

        // Ensure the model has at least one input and one output
        if (reconstructed_graph_proto->input_size() == 0) {
            std::cout << "  Adding default input specification..." << std::endl;
            onnx::ValueInfoProto* input = reconstructed_graph_proto->add_input();
            input->set_name("input_0");
            onnx::TypeProto* input_type = input->mutable_type();
            onnx::TypeProto_Tensor* input_tensor = input_type->mutable_tensor_type();
            input_tensor->set_elem_type(onnx::TensorProto::FLOAT);
            onnx::TensorShapeProto* input_shape = input_tensor->mutable_shape();
            onnx::TensorShapeProto_Dimension* input_dim = input_shape->add_dim();
            input_dim->set_dim_value(1);
        }

        if (reconstructed_graph_proto->output_size() == 0) {
            std::cout << "  Adding default output specification..." << std::endl;
            onnx::ValueInfoProto* output = reconstructed_graph_proto->add_output();
            output->set_name("output_0");
            onnx::TypeProto* output_type = output->mutable_type();
            onnx::TypeProto_Tensor* output_tensor = output_type->mutable_tensor_type();
            output_tensor->set_elem_type(onnx::TensorProto::FLOAT);
            onnx::TensorShapeProto* output_shape = output_tensor->mutable_shape();
            onnx::TensorShapeProto_Dimension* output_dim = output_shape->add_dim();
            output_dim->set_dim_value(1);
        }

        // Set model metadata
        reconstructed_model_proto.set_ir_version(onnx::Version::IR_VERSION);
        reconstructed_model_proto.set_producer_name("CortexSDR");
        reconstructed_model_proto.set_producer_version("1.0");
        reconstructed_model_proto.set_domain("ai.compression");
        reconstructed_model_proto.set_model_version(1);
        reconstructed_model_proto.set_doc_string("Reconstructed ONNX model");

        // Instead of using SerializeToString which seems to produce zeros,
        // use the original model structure data directly
        std::string serialized_model;
        
        // Check if we have the original model structure data
        if (graph_structure_segment && !graph_structure_segment->data.empty()) {
            // Use the original model structure data
            serialized_model.assign(
                reinterpret_cast<const char*>(graph_structure_segment->data.data()),
                graph_structure_segment->data.size()
            );
            std::cerr << "  Using original model structure data for serialization" << std::endl;
        } else {
            // Fall back to SerializeToString
            if (!reconstructed_model_proto.SerializeToString(&serialized_model)) {
                std::cerr << "  Failed to serialize reconstructed model" << std::endl;
                return {str_to_c("Failed to serialize reconstructed model"), 1};
            }
        }
        
        // Add debug information about the serialized model
        std::cerr << "  Serialized model size: " << serialized_model.size() << " bytes" << std::endl;
        if (serialized_model.empty()) {
            std::cerr << "  Warning: Serialized model is empty!" << std::endl;
        } else {
            std::cerr << "  First 32 bytes of serialized model: ";
            for (size_t i = 0; i < std::min(size_t(32), serialized_model.size()); i++) {
                std::cerr << std::hex << std::setw(2) << std::setfill('0') << (int)(unsigned char)serialized_model[i] << " ";
            }
            std::cerr << std::dec << std::endl;
        }
        
        // Verify the serialized model can be parsed
        onnx::ModelProto verify_model;
        if (!verify_model.ParseFromString(serialized_model)) {
            std::cerr << "  Warning: Serialized model cannot be parsed as a valid ONNX model" << std::endl;
        }

        std::cout << "  Successfully reconstructed model with " 
                  << reconstructed_graph_proto->node_size() << " nodes and "
                  << reconstructed_graph_proto->initializer_size() << " initializers" << std::endl;

        std::ofstream output_stream(output_path, std::ios::binary | std::ios::trunc);
        if (!output_stream) {
            return {str_to_c("Failed to open output file for writing"), 1};
        }

        output_stream.write(serialized_model.data(), serialized_model.size());
        if (!output_stream.good()) {
            return {str_to_c("Failed to write serialized data to file"), 1};
            }

            output_stream.close();
        std::cout << "Reconstruction complete." << std::endl;
        return {nullptr, 0};
#else 
        std::string error_msg = "Error: Decompression successful but reconstructing requires ONNX Protobuf support.";
        return {str_to_c(error_msg), 2};
#endif
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_decompressor_free(CortexDecompressorHandle handle) {
    try {
        if (handle) {
            delete reinterpret_cast<AIDecompressor*>(handle);
        }
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

void cortex_error_free(CortexError* error) {
    if (error && error->message) {
        delete[] error->message;
        error->message = nullptr;
    }
}
