#include "c_api.hpp"
#include "../core/AICompressor.hpp"
#include "../core/AIDecompressor.hpp"
#include "../strategies/SDRIndexStorage.hpp"
#include "../strategies/GzipStrategy.hpp"
#include "../strategies/NumericalRLE.hpp"
#include "../strategies/QuantizedTensorStrategy.hpp"
#include "../parsers/ONNXModelParser.hpp"
#include "../parsers/GGUFModelParser.hpp"
#include "../utils/ModelConverter.hpp"
#include "../streaming/StreamingCompressor.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory> // For std::shared_ptr
#include <fstream> // For std::ofstream

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
        if (options) {
            if (options->use_delta_encoding) {
                auto sdrStrategy = std::make_shared<SDRIndexStorageStrategy>();
                sdrStrategy->setSparsity(options->sparsity);
                ai_compressor->registerStrategy(SegmentType::SPARSE_INDICES, 1, SDR_STRATEGY_ID, sdrStrategy);
                ai_compressor->registerStrategy(SegmentType::MODEL_INPUT,    1, SDR_STRATEGY_ID, sdrStrategy);
                ai_compressor->registerStrategy(SegmentType::MODEL_OUTPUT,   1, SDR_STRATEGY_ID, sdrStrategy);
                ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32,   1, SDR_STRATEGY_ID, sdrStrategy);
                ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16,   1, SDR_STRATEGY_ID, sdrStrategy);
                ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT8,   1, SDR_STRATEGY_ID, sdrStrategy);
            }
            if (options->use_rle) {
                auto rleStrategy = std::make_shared<NumericalRLEStrategy>();
                ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 2, RLE_STRATEGY_ID, rleStrategy);
                ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16, 2, RLE_STRATEGY_ID, rleStrategy);
                ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT8, 2, RLE_STRATEGY_ID, rleStrategy);
                ai_compressor->registerStrategy(SegmentType::MODEL_INPUT,  2, RLE_STRATEGY_ID, rleStrategy);
                ai_compressor->registerStrategy(SegmentType::MODEL_OUTPUT, 2, RLE_STRATEGY_ID, rleStrategy);
            }
#ifdef ENABLE_QUANTIZATION
            if (options->use_quantization) {
                auto quantStrategy = std::make_shared<QuantizedTensorStrategy>(options->quantization_bits);
                ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 2, QUANT_STRATEGY_ID, quantStrategy);
                ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16, 2, QUANT_STRATEGY_ID, quantStrategy);
            }
#endif
            auto gzipStrategy = std::make_shared<GzipStrategy>(options->compression_level);
            ai_compressor->registerStrategy(SegmentType::METADATA_JSON, 1, GZIP_STRATEGY_ID, gzipStrategy);
            ai_compressor->registerStrategy(SegmentType::GRAPH_STRUCTURE_PROTO, 1, GZIP_STRATEGY_ID, gzipStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 3, GZIP_STRATEGY_ID, gzipStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16, 3, GZIP_STRATEGY_ID, gzipStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT8, 3, GZIP_STRATEGY_ID, gzipStrategy);
        }
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
        auto sdrStrategy = std::make_shared<SDRIndexStorageStrategy>();
        sdrStrategy->setSparsity(sparsity);
        ai_decompressor->registerStrategy(SDR_STRATEGY_ID, sdrStrategy);
        ai_decompressor->registerStrategy(RLE_STRATEGY_ID, std::make_shared<NumericalRLEStrategy>());
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
            std::cout << "  Processing GraphProto segment..." << std::endl;
            if (!reconstructed_graph_proto->ParseFromArray(graph_structure_segment->data.data(), graph_structure_segment->data.size())) {
                std::string graph_str(reinterpret_cast<const char*>(graph_structure_segment->data.data()), graph_structure_segment->data.size());
                if (!reconstructed_graph_proto->ParseFromString(graph_str)) {
                   std::cerr << "  Failed to parse GraphProto from segment data." << std::endl;
                   return {str_to_c("Error: Failed to parse GraphProto segment data."), 1};
                }
            }
            std::cout << "  Successfully parsed graph structure." << std::endl;
        } else {
            std::cout << "  Warning: GRAPH_STRUCTURE_PROTO segment not found in archive. Creating default graph." << std::endl;
            reconstructed_graph_proto->set_name("reconstructed_graph");
        }

        std::cout << "  Graph state after processing segment (or default): Nodes=" << reconstructed_graph_proto->node_size()
                  << ", Inputs=" << reconstructed_graph_proto->input_size()
                  << ", Outputs=" << reconstructed_graph_proto->output_size() << std::endl;

        std::cout << "  Adding " << weight_segments.size() << " weight initializers and graph inputs..." << std::endl;
        std::string first_initializer_name;

        for (const auto* weight_seg : weight_segments) {
            onnx::TensorProto* tensor_proto = reconstructed_graph_proto->add_initializer();
            tensor_proto->set_name(weight_seg->name);
            if (first_initializer_name.empty()) first_initializer_name = weight_seg->name;

            int32_t onnx_dtype = onnx::TensorProto::UNDEFINED;
            switch(weight_seg->type) {
                case SegmentType::WEIGHTS_FP32: onnx_dtype = onnx::TensorProto::FLOAT; break;
                case SegmentType::WEIGHTS_FP16: onnx_dtype = onnx::TensorProto::FLOAT16; break;
                case SegmentType::WEIGHTS_INT8:  onnx_dtype = onnx::TensorProto::INT8; break;
                default:
                     std::cerr << "Warning: Unhandled segment type for initializer data type: " << weight_seg->name << std::endl;
                     onnx_dtype = onnx::TensorProto::FLOAT;
            }
            tensor_proto->set_data_type(onnx_dtype);

            if (weight_seg->tensor_metadata && !weight_seg->tensor_metadata->dimensions.empty()) {
                 for (size_t dim : weight_seg->tensor_metadata->dimensions) {
                     tensor_proto->add_dims(static_cast<int64_t>(dim));
                 }
            } else {
                 std::cerr << "Warning: Missing dimension metadata for weight segment: " << weight_seg->name << std::endl;
            }
            tensor_proto->set_raw_data(weight_seg->data.data(), weight_seg->data.size());

            if (!graph_structure_segment) {
                onnx::ValueInfoProto* input_val_info = reconstructed_graph_proto->add_input();
                input_val_info->set_name(weight_seg->name);
                onnx::TypeProto* type_proto = input_val_info->mutable_type();
                onnx::TypeProto_Tensor* tensor_type_proto = type_proto->mutable_tensor_type();
                tensor_type_proto->set_elem_type(onnx_dtype);
                onnx::TensorShapeProto* shape_proto = tensor_type_proto->mutable_shape();
                if (weight_seg->tensor_metadata && !weight_seg->tensor_metadata->dimensions.empty()) {
                    for (size_t dim_val : weight_seg->tensor_metadata->dimensions) {
                        shape_proto->add_dim()->set_dim_value(static_cast<int64_t>(dim_val));
                    }
                } else {
                     std::cerr << "Warning: Missing dimension metadata for graph input: " << weight_seg->name << std::endl;
                }
            }
        }

        if (!graph_structure_segment && !weight_segments.empty()) {
            const auto* first_weight_seg = weight_segments.front(); // For type and shape info
            
            // Add graph output
            onnx::ValueInfoProto* output_val_info = reconstructed_graph_proto->add_output();
            output_val_info->set_name("output_" + first_initializer_name); // Ensure unique output name
            
            onnx::TypeProto* type_proto_out = output_val_info->mutable_type();
            onnx::TypeProto_Tensor* tensor_type_proto_out = type_proto_out->mutable_tensor_type();
            
            int32_t output_onnx_dtype = onnx::TensorProto::FLOAT; // Default
            if (first_weight_seg->tensor_metadata) { // Check if tensor_metadata exists
                 switch(first_weight_seg->type) { // Determine type from first_weight_seg
                    case SegmentType::WEIGHTS_FP32: output_onnx_dtype = onnx::TensorProto::FLOAT; break;
                    case SegmentType::WEIGHTS_FP16: output_onnx_dtype = onnx::TensorProto::FLOAT16; break;
                    case SegmentType::WEIGHTS_INT8:  output_onnx_dtype = onnx::TensorProto::INT8; break;
                    default: output_onnx_dtype = onnx::TensorProto::FLOAT; // Fallback
                }
            }
            tensor_type_proto_out->set_elem_type(output_onnx_dtype);

            onnx::TensorShapeProto* shape_proto_out = tensor_type_proto_out->mutable_shape();
            if (first_weight_seg->tensor_metadata && !first_weight_seg->tensor_metadata->dimensions.empty()) {
                for (size_t dim_val : first_weight_seg->tensor_metadata->dimensions) {
                    shape_proto_out->add_dim()->set_dim_value(static_cast<int64_t>(dim_val));
                }
            }
            
            // Add Identity node
            onnx::NodeProto* identity_node = reconstructed_graph_proto->add_node();
            identity_node->add_input(first_initializer_name);
            identity_node->add_output("output_" + first_initializer_name);
            identity_node->set_op_type("Identity");
            identity_node->set_name("Identity_node_for_" + first_initializer_name);
        }

        reconstructed_model_proto.set_ir_version(onnx::Version::IR_VERSION);
        reconstructed_model_proto.set_producer_name("CortexSDR_Reconstructor");
        auto* opset_import = reconstructed_model_proto.add_opset_import();
        opset_import->set_domain(""); 
        opset_import->set_version(12); 

        std::cout << "  Serializing reconstructed model to: " << output_path << std::endl;
        std::ofstream output_stream(output_path, std::ios::binary | std::ios::trunc);
        if (!output_stream) {
             return {str_to_c("Error: Failed to open output file for writing: " + std::string(output_path)), 1};
        }
        if (!reconstructed_model_proto.SerializeToOstream(&output_stream)) {
             output_stream.close();
             return {str_to_c("Error: Failed to serialize reconstructed ONNX model to file."), 1};
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
