#include "c_api.hpp"
#include "../core/AICompressor.hpp"
#include "../core/AIDecompressor.hpp"
#include "../strategies/SDRIndexStorage.hpp"
#include "../strategies/GzipStrategy.hpp"
#include "../strategies/NumericalRLE.hpp"
#include "../strategies/QuantizedTensorStrategy.hpp"
#include "../strategies/HierarchicalSDRStrategy.hpp"
#include "../strategies/AdaptiveSDRStrategy.hpp"
#include "../parsers/ONNXModelParser.hpp"
#include "../parsers/GGUFModelParser.hpp"
#include "../parsers/ModelParserFactory.hpp"
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
#include <algorithm>
#include <cctype>
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

    // Automatically detect and find all parts of a multi-part model
    // Supports Hugging Face patterns like: model-00001-of-00002.gguf
    std::vector<std::string> findModelParts(const std::string& model_path) {
        std::vector<std::string> parts;
        
        std::filesystem::path path(model_path);
        if (!std::filesystem::exists(path)) {
            // File doesn't exist, return original path
            return {model_path};
        }
        
        std::filesystem::path parent = path.parent_path();
        if (parent.empty()) {
            parent = std::filesystem::current_path();
        }
        
        std::string filename = path.filename().string();
        
        // Pattern: model-00001-of-00005.gguf (Hugging Face standard)
        std::regex pattern(R"(^(.+)-(\d+)-of-(\d+)(\..+)$)");
        std::smatch match;
        
        if (std::regex_match(filename, match, pattern)) {
            std::string base_name = match[1].str();
            int current_part = std::stoi(match[2].str());
            int total_parts = std::stoi(match[3].str());
            std::string ext = match[4].str();
            
            std::cout << "[Multi-part detection] Found " << total_parts << "-part model: " << base_name << std::endl;
            
            // Find all parts
            for (int i = 1; i <= total_parts; i++) {
                // Format with leading zeros matching the original
                std::ostringstream part_name;
                part_name << base_name << "-";
                part_name << std::setfill('0') << std::setw(5) << i;
                part_name << "-of-";
                part_name << std::setfill('0') << std::setw(5) << total_parts;
                part_name << ext;
                
                std::filesystem::path part_path = parent / part_name.str();
                if (std::filesystem::exists(part_path)) {
                    parts.push_back(part_path.string());
                    std::cout << "  [Part " << i << "/" << total_parts << "] " << part_path.filename().string() << std::endl;
                } else {
                    std::cerr << "  [Warning] Missing part " << i << "/" << total_parts << ": " << part_path.filename().string() << std::endl;
                }
            }
            
            if (!parts.empty()) {
                return parts;
            }
        }
        
        // No multi-part pattern detected, return single file
        return {model_path};
    }

    std::string sanitizeFileName(const std::string& value) {
        std::string output;
        output.reserve(value.size());
        for (unsigned char ch : value) {
            if (std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.') {
                output.push_back(static_cast<char>(ch));
            } else {
                output.push_back('_');
            }
        }
        if (output.empty()) {
            output = "segment";
        }
        return output;
    }

    std::string segmentTypeToString(SegmentType type) {
        switch (type) {
            case SegmentType::UNKNOWN: return "UNKNOWN";
            case SegmentType::WEIGHTS_FP32: return "WEIGHTS_FP32";
            case SegmentType::WEIGHTS_FP16: return "WEIGHTS_FP16";
            case SegmentType::WEIGHTS_INT8: return "WEIGHTS_INT8";
            case SegmentType::WEIGHTS_INT4: return "WEIGHTS_INT4";
            case SegmentType::SPARSE_INDICES: return "SPARSE_INDICES";
            case SegmentType::METADATA_JSON: return "METADATA_JSON";
            case SegmentType::METADATA_TOML: return "METADATA_TOML";
            case SegmentType::TOKENIZER_VOCAB: return "TOKENIZER_VOCAB";
            case SegmentType::TOKENIZER_MODEL: return "TOKENIZER_MODEL";
            case SegmentType::CONFIG: return "CONFIG";
            case SegmentType::ATTENTION_WEIGHTS: return "ATTENTION_WEIGHTS";
            case SegmentType::FEED_FORWARD_WEIGHTS: return "FEED_FORWARD_WEIGHTS";
            case SegmentType::EMBEDDING_WEIGHTS: return "EMBEDDING_WEIGHTS";
            case SegmentType::LAYER_NORM_WEIGHTS: return "LAYER_NORM_WEIGHTS";
            case SegmentType::MODEL_INPUT: return "MODEL_INPUT";
            case SegmentType::MODEL_OUTPUT: return "MODEL_OUTPUT";
            case SegmentType::GRAPH_STRUCTURE_PROTO: return "GRAPH_STRUCTURE_PROTO";
            default: return "UNKNOWN";
        }
    }

    std::string escapeJsonString(const std::string& input) {
        std::ostringstream escaped;
        for (unsigned char ch : input) {
            switch (ch) {
                case '\\': escaped << "\\\\"; break;
                case '"': escaped << "\\\""; break;
                case '\b': escaped << "\\b"; break;
                case '\f': escaped << "\\f"; break;
                case '\n': escaped << "\\n"; break;
                case '\r': escaped << "\\r"; break;
                case '\t': escaped << "\\t"; break;
                default:
                    if (ch < 0x20) {
                        escaped << "\\u"
                                << std::hex << std::setw(4) << std::setfill('0')
                                << static_cast<int>(ch)
                                << std::dec << std::setfill(' ');
                    } else {
                        escaped << static_cast<char>(ch);
                    }
            }
        }
        return escaped.str();
    }

    bool isLikelyGGUFArchive(const std::vector<ModelSegment>& segments) {
        bool has_gguf_markers = false;
        for (const auto& segment : segments) {
            const std::string lower_name = [&segment]() {
                std::string value = segment.name;
                std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
                    return static_cast<char>(std::tolower(ch));
                });
                return value;
            }();

            if (lower_name == "gguf_metadata" || lower_name == "gguf_config" ||
                lower_name == "gguf_tokenizer_model" || lower_name == "gguf_tokenizer_vocab") {
                has_gguf_markers = true;
            }
            if (segment.name == "gguf_metadata" && segment.type == SegmentType::METADATA_JSON && !segment.data.empty()) {
                const std::string payload(
                    reinterpret_cast<const char*>(segment.data.data()),
                    segment.data.size()
                );
                if (payload.find("\"format\":\"GGUF\"") != std::string::npos) {
                    has_gguf_markers = true;
                }
            }
        }
        return has_gguf_markers;
    }

    void writeArchiveExtractionBundle(
        const std::vector<ModelSegment>& segments,
        const std::filesystem::path& output_root,
        bool gguf_friendly_names
    ) {
        std::filesystem::create_directories(output_root);
        std::filesystem::create_directories(output_root / "tensors");
        std::filesystem::create_directories(output_root / "segments");

        std::ofstream manifest(output_root / "manifest.json", std::ios::binary | std::ios::trunc);
        if (!manifest) {
            throw std::runtime_error("Failed to create archive extraction manifest.");
        }

        manifest << "{\n  \"format\": \"ARCHIVE_EXTRACTED\",\n  \"segments\": [\n";
        bool first_entry = true;

        for (size_t index = 0; index < segments.size(); ++index) {
            const auto& segment = segments[index];
            std::filesystem::path relative_output;
            const bool text_segment =
                segment.type == SegmentType::METADATA_JSON ||
                segment.type == SegmentType::CONFIG ||
                segment.type == SegmentType::TOKENIZER_VOCAB ||
                segment.type == SegmentType::TOKENIZER_MODEL;

            if (gguf_friendly_names && text_segment) {
                if (segment.name == "gguf_metadata") {
                    relative_output = "metadata.json";
                } else if (segment.name == "gguf_config") {
                    relative_output = "config.json";
                } else if (segment.name == "gguf_tokenizer_vocab") {
                    relative_output = "tokenizer_vocab.txt";
                } else if (segment.name == "gguf_tokenizer_model") {
                    relative_output = "tokenizer_model.txt";
                } else {
                    relative_output = sanitizeFileName(segment.name) + ".txt";
                }
            } else {
                const std::string base_name = std::to_string(index) + "_" + sanitizeFileName(segment.name);
                if (text_segment) {
                    relative_output = std::filesystem::path("segments") / (base_name + ".txt");
                } else {
                    relative_output = std::filesystem::path("tensors") / (base_name + ".bin");
                }
            }

            const std::filesystem::path full_output = output_root / relative_output;
            std::ofstream out(full_output, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("Failed to write extracted GGUF segment: " + segment.name);
            }
            if (!segment.data.empty()) {
                out.write(reinterpret_cast<const char*>(segment.data.data()), static_cast<std::streamsize>(segment.data.size()));
            }
            out.close();

            if (!first_entry) {
                manifest << ",\n";
            }
            first_entry = false;
            manifest << "    {\"name\":\"" << escapeJsonString(segment.name)
                     << "\",\"type\":\"" << escapeJsonString(segmentTypeToString(segment.type))
                     << "\",\"size\":" << segment.data.size()
                     << ",\"file\":\"" << escapeJsonString(relative_output.generic_string()) << "\"}";
        }

        manifest << "\n  ]\n}\n";
        manifest.close();
    }

    std::unique_ptr<AIDecompressor> createConfiguredDecompressor(float sparsity) {
        auto decompressor = std::make_unique<AIDecompressor>();
        const uint8_t SDR_STRATEGY_ID = 1;
        const uint8_t RLE_STRATEGY_ID = 2;
        const uint8_t GZIP_STRATEGY_ID = 3;
        const uint8_t QUANT_STRATEGY_ID = 4;
        const uint8_t HSDR_STRATEGY_ID = 5;  ///< V4b hierarchical binary SDR

        auto adaptiveStrategy = std::make_shared<AdaptiveSDRStrategy>(sparsity);
        decompressor->registerStrategy(SDR_STRATEGY_ID, adaptiveStrategy);

        auto sdrStrategy = std::make_shared<SDRIndexStorageStrategy>();
        sdrStrategy->setSparsity(sparsity);
        decompressor->registerStrategy(SDR_STRATEGY_ID + 10, sdrStrategy);
        decompressor->registerStrategy(RLE_STRATEGY_ID, std::make_shared<NumericalRLEStrategy>());
        decompressor->registerStrategy(GZIP_STRATEGY_ID, std::make_shared<GzipStrategy>());
        decompressor->registerStrategy(QUANT_STRATEGY_ID, std::make_shared<QuantizedTensorStrategy>());
        decompressor->registerStrategy(HSDR_STRATEGY_ID, std::make_shared<HierarchicalSDRStrategy>());

        return decompressor;
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
        options->use_quantization = 1;
        options->quantization_bits = 8;
        options->use_hsdr = 0;  // V4b: opt-in, fitting is slow at TinyLlama scale
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
    std::vector<std::string> additionalFiles;  // For multi-part models
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
        std::transform(actualFormat.begin(), actualFormat.end(), actualFormat.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        
        // Check if the format is supported by our parsers
        if (!ModelParserFactory::isFormatSupported(actualFormat)) {
            return {str_to_c("Unsupported model format: " + std::string(format)), 1};
        }
        // Use the user-requested format to select the parser explicitly.
        // This avoids accidental parser mismatches when file extension/content is ambiguous.
        std::unique_ptr<IAIModelParser> parser;
        try {
            parser = ModelParserFactory::createParserForFormat(actualFormat);
        } catch (const ParsingError& e) {
            return {str_to_c("Failed to create parser: " + std::string(e.what())), 1};
        } catch (const std::exception& e) {
            return {str_to_c("Unexpected error creating parser: " + std::string(e.what())), 1};
        }
        
        if (!parser) {
             return {str_to_c("Failed to create model parser for format: " + actualFormat), 1};
        }
        auto ai_compressor = new AICompressor(std::move(parser));
        const uint8_t SDR_STRATEGY_ID = 1;
        const uint8_t RLE_STRATEGY_ID = 2;
        const uint8_t GZIP_STRATEGY_ID = 3;
        const uint8_t QUANT_STRATEGY_ID = 4;
        const uint8_t HSDR_STRATEGY_ID = 5;  ///< V4b hierarchical binary SDR

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
        ai_compressor->registerStrategy(SegmentType::CONFIG, 1, SDR_STRATEGY_ID, adaptiveStrategy);
        ai_compressor->registerStrategy(SegmentType::TOKENIZER_VOCAB, 1, SDR_STRATEGY_ID, adaptiveStrategy);
        ai_compressor->registerStrategy(SegmentType::TOKENIZER_MODEL, 1, SDR_STRATEGY_ID, adaptiveStrategy);
        
        // Tensor data segments.
        // SDR is intentionally NOT registered for weight tensor types: magnitude
        // pruning + heuristic interpolation destroys LLM semantics. SDR is reserved
        // for naturally-sparse data (SPARSE_INDICES) and metadata.
        if (options->use_delta_encoding) {
            ai_compressor->registerStrategy(SegmentType::SPARSE_INDICES, 2, SDR_STRATEGY_ID, adaptiveStrategy);
        }

        // V4b hierarchical binary SDR is the new primary lossy compressor for FP32
        // weight tensors. It fails fast on non-FP32 dtypes and on shapes not divisible
        // by the tile size, so the chain falls through to Quant / Gzip cleanly.
        // OPT-IN: fitting is slow (~minutes per MLP layer on CPU). See
        // research/step0/results/STEP0_REPORT.md for the design rationale.
        if (options->use_hsdr) {
            auto hsdrStrategy = std::make_shared<HierarchicalSDRStrategy>();
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 1, HSDR_STRATEGY_ID, hsdrStrategy);
            ai_compressor->registerStrategy(SegmentType::ATTENTION_WEIGHTS, 1, HSDR_STRATEGY_ID, hsdrStrategy);
            ai_compressor->registerStrategy(SegmentType::FEED_FORWARD_WEIGHTS, 1, HSDR_STRATEGY_ID, hsdrStrategy);
            ai_compressor->registerStrategy(SegmentType::EMBEDDING_WEIGHTS, 1, HSDR_STRATEGY_ID, hsdrStrategy);
            ai_compressor->registerStrategy(SegmentType::LAYER_NORM_WEIGHTS, 1, HSDR_STRATEGY_ID, hsdrStrategy);
        }

        // Quantization is the broadly-applicable lossy compressor for weight tensors.
        // It throws on non-FP32 input (already-quantized GGUF, FP16, etc.) so those
        // segments fall through to the Gzip lossless fallback below.
        if (options->use_quantization) {
            auto quantStrategy = std::make_shared<QuantizedTensorStrategy>(options->quantization_bits);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 2, QUANT_STRATEGY_ID, quantStrategy);
            ai_compressor->registerStrategy(SegmentType::ATTENTION_WEIGHTS, 2, QUANT_STRATEGY_ID, quantStrategy);
            ai_compressor->registerStrategy(SegmentType::FEED_FORWARD_WEIGHTS, 2, QUANT_STRATEGY_ID, quantStrategy);
            ai_compressor->registerStrategy(SegmentType::EMBEDDING_WEIGHTS, 2, QUANT_STRATEGY_ID, quantStrategy);
            ai_compressor->registerStrategy(SegmentType::LAYER_NORM_WEIGHTS, 2, QUANT_STRATEGY_ID, quantStrategy);
        }

        if (options->use_rle) {
            auto rleStrategy = std::make_shared<NumericalRLEStrategy>();
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT8, 3, RLE_STRATEGY_ID, rleStrategy);
            ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT4, 3, RLE_STRATEGY_ID, rleStrategy);
        }

        // Register Gzip as fallback for weight tensors with lowest priority
        ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::WEIGHTS_FP16, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT8, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::WEIGHTS_INT4, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::ATTENTION_WEIGHTS, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::FEED_FORWARD_WEIGHTS, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::EMBEDDING_WEIGHTS, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::LAYER_NORM_WEIGHTS, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::CONFIG, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::TOKENIZER_VOCAB, 4, GZIP_STRATEGY_ID, gzipStrategy);
        ai_compressor->registerStrategy(SegmentType::TOKENIZER_MODEL, 4, GZIP_STRATEGY_ID, gzipStrategy);

        *handle = reinterpret_cast<CortexCompressorHandle>(ai_compressor);
        
        // Auto-detect multi-part models
        std::vector<std::string> model_parts = findModelParts(actualModelPath);
        std::vector<std::string> additional_parts;
        
        if (model_parts.size() > 1) {
            std::cout << "[API] Auto-detected " << model_parts.size() << "-part model" << std::endl;
            // First part is the primary, rest are additional
            actualModelPath = model_parts[0];
            for (size_t i = 1; i < model_parts.size(); i++) {
                additional_parts.push_back(model_parts[i]);
            }
            std::cout << "[API] Will compress all " << model_parts.size() << " parts into single archive" << std::endl;
        }
        
        g_modelInfoMap[*handle] = {model_path, actualModelPath, actualFormat, additional_parts};
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_compressor_add_file(CortexCompressorHandle handle, const char* model_path) {
    try {
        if (!handle || !model_path) {
            return {str_to_c("Invalid arguments (null handle or path)"), 1};
        }
        
        auto it = g_modelInfoMap.find(handle);
        if (it == g_modelInfoMap.end()) {
            return {str_to_c("Internal error: Model info not found for handle."), 1};
        }
        
        // Clear auto-detected files on first manual add (to prevent duplication)
        static std::unordered_set<CortexCompressorHandle> handles_with_manual_adds;
        if (handles_with_manual_adds.find(handle) == handles_with_manual_adds.end()) {
            // First manual add - clear auto-detected files
            if (!it->second.additionalFiles.empty()) {
                std::cout << "[API] Clearing auto-detected files (manual mode)" << std::endl;
                it->second.additionalFiles.clear();
            }
            handles_with_manual_adds.insert(handle);
        }
        
        // Add the file to the list of additional files
        it->second.additionalFiles.push_back(model_path);
        
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

int cortex_model_get_parts(const char* model_path, char*** parts_out, int* num_parts) {
    try {
        if (!model_path) {
            return 0;
        }
        
        std::vector<std::string> parts = findModelParts(model_path);
        
        if (num_parts) {
            *num_parts = static_cast<int>(parts.size());
        }
        
        if (parts_out) {
            *parts_out = new char*[parts.size()];
            for (size_t i = 0; i < parts.size(); i++) {
                (*parts_out)[i] = str_to_c(parts[i]);
            }
        }
        
        return static_cast<int>(parts.size());
    } catch (const std::exception& e) {
        std::cerr << "Error in cortex_model_get_parts: " << e.what() << std::endl;
        return 0;
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
        
        // Compress the primary file
        const std::string& modelToCompress = it->second.actualPath;
        compressor->compressModelStreaming(modelToCompress, streamCompressor);
        
        // Compress additional files if any (for multi-part models)
        for (const auto& additionalFile : it->second.additionalFiles) {
            compressor->compressModelStreaming(additionalFile, streamCompressor);
        }
        
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
        auto configured = createConfiguredDecompressor(sparsity);
        *handle = reinterpret_cast<CortexDecompressorHandle>(configured.release());
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

        // Runtime contract is one-way (.sdr is the inference artifact), so
        // decompression here performs archive extraction only.
        const std::filesystem::path out_path(output_path);
        if (std::filesystem::exists(out_path) && !std::filesystem::is_directory(out_path)) {
            return {str_to_c("Output path exists and is not a directory: " + out_path.string()), 1};
        }
        const bool gguf_friendly_names = isLikelyGGUFArchive(handler.segments);
        writeArchiveExtractionBundle(handler.segments, out_path, gguf_friendly_names);
        return {nullptr, 0};

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

CortexError cortex_archive_extract(const char* compressed_path, const char* output_dir, float sparsity) {
    try {
        if (!compressed_path || !output_dir) {
            return {str_to_c("Invalid arguments (null archive path or output dir)"), 1};
        }

        std::ifstream input_file(compressed_path, std::ios::binary);
        if (!input_file) {
            return {str_to_c("Failed to open compressed file: " + std::string(compressed_path)), 1};
        }

        auto decompressor = createConfiguredDecompressor(sparsity);

        class InMemorySegmentHandler : public ISegmentHandler {
        public:
            void handleSegment(ModelSegment segment) override {
                segments.push_back(std::move(segment));
            }
            std::vector<ModelSegment> segments;
        };

        InMemorySegmentHandler handler;
        decompressor->decompressModelStream(input_file, handler);
        input_file.close();

        const std::filesystem::path output_root(output_dir);
        if (std::filesystem::exists(output_root) && !std::filesystem::is_directory(output_root)) {
            return {str_to_c("Output path exists and is not a directory: " + output_root.string()), 1};
        }

        const bool gguf_friendly_names = isLikelyGGUFArchive(handler.segments);
        writeArchiveExtractionBundle(handler.segments, output_root, gguf_friendly_names);
        return {nullptr, 0};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

const char* cortex_error_string(int code) {
    switch (code) {
        case 0: return "Success";
        case 1: return "Invalid argument";
        case 2: return "File not found";
        case 3: return "Unsupported format";
        case 100: return "Internal error";
        default: return "Unknown error";
    }
}

void cortex_error_free(CortexError* error) {
    if (error && error->message) {
        delete[] error->message;
        error->message = nullptr;
    }
}
