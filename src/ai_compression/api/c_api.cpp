#include "c_api.hpp"
#include "../core/AICompressor.hpp"
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

void cortex_error_free(CortexError* error) {
    if (error && error->message) {
        delete[] error->message;
        error->message = nullptr;
    }
}
