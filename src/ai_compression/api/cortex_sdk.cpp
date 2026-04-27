/**
 * @file cortex_sdk.cpp
 * @brief Implementation of the CortexSDR C++ SDK API
 * 
 * This file provides the C++ implementation of the CortexSDR software development kit,
 * offering high-level interfaces for neural network compression, decompression, and
 * inference operations with sparse distributed representations.
 * 
 * Key Features:
 * - Neural network model compression with various strategies
 * - On-demand layer loading for memory-efficient inference
 * - Multiple compression formats (SDR, RLE, Gzip, Quantization)
 * - C-compatible API for cross-language integration
 * - Comprehensive error handling and resource management
 */

#include "cortex_sdk.h"
#include "c_api.hpp"
#include "../SparseInferenceEngine.hpp"
#include "../core/AICompressor.hpp"
#include "../core/AIDecompressor.hpp"
#include <string>
#include <cstring>
#include "../parsers/ModelParserFactory.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h> // mkstemp, close, unlink
#include <fcntl.h>
#endif
 #include <sstream>
 #include <filesystem>

using namespace CortexAICompression;

// SDK version information
#define CORTEX_SDK_VERSION "1.0.0"

// Error code definitions for comprehensive error handling
#define CORTEX_SUCCESS 0                        ///< Operation completed successfully
#define CORTEX_ERROR_INVALID_ARGUMENT -1        ///< Invalid input parameter provided
#define CORTEX_ERROR_FILE_IO -2                 ///< File input/output operation failed
#define CORTEX_ERROR_MEMORY -3                  ///< Memory allocation or management error
#define CORTEX_ERROR_UNSUPPORTED_FORMAT -4      ///< Unsupported file or data format
#define CORTEX_ERROR_COMPRESSION -5             ///< Compression operation failed
#define CORTEX_ERROR_DECOMPRESSION -6           ///< Decompression operation failed
#define CORTEX_ERROR_INFERENCE -7               ///< Neural network inference failed
#define CORTEX_ERROR_UNKNOWN -99                ///< Unknown or unexpected error

/**
 * @brief Internal structure for inference engine handle management
 * 
 * Encapsulates the inference engine components with proper resource management
 * and supports both on-demand and legacy inference modes.
 */
struct CortexInferenceEngine {
    std::unique_ptr<SDRModelLoader> model_loader;
    std::unique_ptr<SDRInferenceEngine> inference_engine;
};

// Global storage for handles
static std::unordered_map<CortexInferenceEngineHandle, CortexInferenceEngine*> g_inferenceEngines;

// Helper functions
namespace {
    CortexError convert_exception(const std::exception& e) {
        char* msg_copy = new char[strlen(e.what()) + 1];
        strcpy(msg_copy, e.what());
        return {msg_copy, CORTEX_ERROR_UNKNOWN}; 
    }

    char* str_to_c(const std::string& str) {
        char* cstr = new char[str.length() + 1];
        strcpy(cstr, str.c_str());
        return cstr;
    }
}

// Error handling and compression options initialization functions are defined in c_api.cpp
// We don't redefine them here to avoid duplicate symbol errors when building shared libraries

// Compressor functions - these are already implemented in c_api.cpp, so we'll just forward them
extern "C" {
    // These are defined in c_api.cpp
    extern CortexError cortex_compressor_create(const char* model_path, const char* format,
                                         const CortexCompressionOptions* options,
                                         CortexCompressorHandle* handle);
    
    extern CortexError cortex_compressor_compress(CortexCompressorHandle handle, const char* output_path);
    
    extern CortexError cortex_compressor_get_stats(CortexCompressorHandle handle,
                                            size_t* original_size,
                                            size_t* compressed_size,
                                            double* compression_ratio,
                                            double* compression_time_ms);
    
    extern CortexError cortex_compressor_free(CortexCompressorHandle handle);
    
    // Forward declaration - implementation is in c_api.cpp
CORTEXSDR_API CortexError cortex_decompressor_create(const char* compressed_path,
                                          CortexDecompressorHandle* handle,
                                          float sparsity);
    
    // Forward declaration - implementation is in c_api.cpp
CORTEXSDR_API CortexError cortex_decompressor_decompress(CortexDecompressorHandle handle,
                                               const char* compressed_path,
                                               const char* output_path);
    
    // Forward declaration - implementation is in c_api.cpp
CORTEXSDR_API CortexError cortex_decompressor_free(CortexDecompressorHandle handle);
}

// Inference Engine functions
CortexError cortex_inference_engine_create(
    const char* compressed_model_path,
    CortexInferenceEngineHandle* handle)
{
    try {
        if (!compressed_model_path || !handle) {
            return {"Invalid arguments (null pointers)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        // Create a new inference engine instance
        auto engine = new CortexInferenceEngine();
        
        // Create the model loader
        engine->model_loader = std::make_unique<SDRModelLoader>(compressed_model_path);
        
        // Create the inference engine
        engine->inference_engine = std::make_unique<SDRInferenceEngine>(*engine->model_loader);
        
        // Store the handle
        *handle = static_cast<CortexInferenceEngineHandle>(engine);
        g_inferenceEngines[*handle] = engine;
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_set_batch_size(
    CortexInferenceEngineHandle handle,
    size_t batch_size)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->setBatchSize(batch_size);
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_enable_dropout(
    CortexInferenceEngineHandle handle,
    int enable)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->enableDropout(enable != 0);
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_set_mode(
    CortexInferenceEngineHandle handle,
    int training_mode)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->setInferenceMode(training_mode != 0);
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_run(
    CortexInferenceEngineHandle handle,
    const float* input_data,
    size_t input_size,
    float* output_data,
    size_t output_size,
    size_t* actual_output_size)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        if (!input_data || !output_data || !actual_output_size) {
            return {"Invalid arguments (null pointers)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        
        // Convert input data to vector
        std::vector<float> input(input_data, input_data + input_size);
        
        // Run inference
        std::vector<float> output = engine->inference_engine->run(input);
        
        // Check output buffer size
        if (output.size() > output_size) {
            *actual_output_size = output.size();
            return {"Output buffer too small", CORTEX_ERROR_MEMORY};
        }
        
        // Copy output data
        std::copy(output.begin(), output.end(), output_data);
        *actual_output_size = output.size();
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_run_layer(
    CortexInferenceEngineHandle handle,
    const char* layer_name,
    const float* input_data,
    size_t input_size,
    float* output_data,
    size_t output_size,
    size_t* actual_output_size)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        if (!layer_name || !input_data || !output_data || !actual_output_size) {
            return {"Invalid arguments (null pointers)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        
        // Convert input data to vector
        std::vector<float> input(input_data, input_data + input_size);
        
        // Load the layer
        LayerInfo layer = engine->model_loader->loadLayerByName(layer_name);
        
        // Run inference on the specific layer
        std::vector<float> output = engine->inference_engine->runLayer(layer, input);
        
        // Check output buffer size
        if (output.size() > output_size) {
            *actual_output_size = output.size();
            return {"Output buffer too small", CORTEX_ERROR_MEMORY};
        }
        
        // Copy output data
        std::copy(output.begin(), output.end(), output_data);
        *actual_output_size = output.size();
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_free(
    CortexInferenceEngineHandle handle)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        
        auto engine = g_inferenceEngines[handle];
        delete engine;
        g_inferenceEngines.erase(handle);
        
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

const char* cortex_sdk_version() {
    return CORTEX_SDK_VERSION;
}
CortexError cortex_inference_engine_get_last_run_stats_json(
    CortexInferenceEngineHandle handle,
    char** out_json)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end() || !out_json) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        const auto& stats = engine->inference_engine->getLastRunStats();
        std::ostringstream os;
        os << "{\"total_ms\": " << stats.total_ms << ", \"layers\": [";
        for (size_t i = 0; i < stats.layers.size(); ++i) {
            const auto& l = stats.layers[i];
            os << "{\"name\":\"" << l.name << "\",\"load_ms\":" << l.load_ms
               << ",\"exec_ms\":" << l.exec_ms
               << ",\"output_size\":" << l.output_size
               << ",\"used_compressed\":" << (l.used_compressed ? "true" : "false")
               << "}";
            if (i + 1 < stats.layers.size()) os << ",";
        }
        os << "]}";
        std::string s = os.str();
        char* buf = (char*)malloc(s.size() + 1);
        if (!buf) return {"Allocation failure", CORTEX_ERROR_MEMORY};
        std::memcpy(buf, s.c_str(), s.size() + 1);
        *out_json = buf;
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

void cortex_free_string(char* s) {
    if (s) free(s);
}

// Inspect a compressed archive for tokenizer assets
CortexError cortex_archive_get_tokenizer_info(
    const char* archive_path,
    int* out_has_tokenizer,
    char** out_tokenizer_type)
{
    try {
        if (!archive_path || !out_has_tokenizer) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        *out_has_tokenizer = 0;
        if (out_tokenizer_type) *out_tokenizer_type = nullptr;

        // Create a temporary loader to read the archive index
        SDRModelLoader loader(archive_path);
        const auto& segments = loader.getSegmentIndex();
        bool has_vocab = false, has_merges = false, has_spm = false;
        for (const auto& seg : segments) {
            std::string n = seg.name;
            for (auto& ch : n) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
            if (n.find("tokenizer.model") != std::string::npos ||
                n.find("sentencepiece") != std::string::npos ||
                n.find(".spm") != std::string::npos ||
                n.find("gguf_tokenizer_model") != std::string::npos) {
                has_spm = true;
            }
            if (n.find("vocab.json") != std::string::npos ||
                n.find("vocab") != std::string::npos ||
                n.find("gguf_tokenizer_vocab") != std::string::npos) {
                has_vocab = true;
            }
            if (n.find("merges.txt") != std::string::npos || n.find("merges") != std::string::npos) {
                has_merges = true;
            }
        }
        if (has_spm || (has_vocab && has_merges)) {
            *out_has_tokenizer = 1;
            if (out_tokenizer_type) {
                std::string t = has_spm ? std::string("sentencepiece") : std::string("gpt2-bpe");
                *out_tokenizer_type = str_to_c(t);
            }
        }
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}
CortexError cortex_compress_from_url(
    const char* url_or_path,
    const char* format,
    const char* output_path,
    float sparsity)
{
    try {
        if (!url_or_path || !output_path) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }

        std::string src(url_or_path);
        std::string local_path = src;
        bool is_remote = (src.rfind("http://", 0) == 0) || (src.rfind("https://", 0) == 0);

        // Download to temp file if remote (or entire repo when applicable)
        std::string tmp_file;
        std::string tmp_dir; // used when downloading full Hugging Face repositories
        if (is_remote) {
            // Create a temporary file path
#ifdef _WIN32
            char tmpPath[MAX_PATH];
            DWORD len = GetTempPathA(MAX_PATH, tmpPath);
            if (len == 0 || len > MAX_PATH) {
                return {"Failed to get temp path", CORTEX_ERROR_FILE_IO};
            }
            char tmpFile[MAX_PATH];
            if (GetTempFileNameA(tmpPath, "cxsdr", 0, tmpFile) == 0) {
                return {"Failed to create temp file", CORTEX_ERROR_FILE_IO};
            }
            tmp_file = std::string(tmpFile);
#else
            char tmpname[] = "/tmp/cortexsdr_dl_XXXXXX";
            int fd = mkstemp(tmpname);
            if (fd == -1) {
                return {"Failed to create temp file", CORTEX_ERROR_FILE_IO};
            }
            close(fd);
            tmp_file = std::string(tmpname);
#endif

            // Build optional auth header for Hugging Face
            std::string authHeader;
            const char* env_hf1 = std::getenv("HUGGING_FACE_HUB_TOKEN");
            const char* env_hf2 = std::getenv("HUGGINGFACE_TOKEN");
            const char* env_hf3 = std::getenv("HF_TOKEN");
            const char* token = env_hf1 ? env_hf1 : (env_hf2 ? env_hf2 : env_hf3);
            bool is_hf = (src.find("huggingface.co/") != std::string::npos);
#ifndef _WIN32
            // Detect Hugging Face repository (folder) URL: no /resolve/ or /blob/
            auto strip_query = [](const std::string& u) {
                std::string p = u;
                auto qpos = p.find('?'); if (qpos != std::string::npos) p = p.substr(0, qpos);
                auto hpos = p.find('#'); if (hpos != std::string::npos) p = p.substr(0, hpos);
                return p;
            };
            std::string clean_src = strip_query(src);
            bool looks_like_repo = is_hf && (clean_src.find("/resolve/") == std::string::npos) && (clean_src.find("/blob/") == std::string::npos);
#else
            // On Windows, still attempt repo detection
            auto strip_query = [](const std::string& u) {
                std::string p = u;
                auto qpos = p.find('?'); if (qpos != std::string::npos) p = p.substr(0, qpos);
                auto hpos = p.find('#'); if (hpos != std::string::npos) p = p.substr(0, hpos);
                return p;
            };
            std::string clean_src = strip_query(src);
            bool looks_like_repo = is_hf && (clean_src.find("/resolve/") == std::string::npos) && (clean_src.find("/blob/") == std::string::npos);
#endif

            int rc = -1;
            bool handled_download = false;
            if (looks_like_repo) {
                // Create a temporary directory for the repository
#ifdef _WIN32
                char tmpPath[MAX_PATH];
                DWORD lenp = GetTempPathA(MAX_PATH, tmpPath);
                if (lenp == 0 || lenp > MAX_PATH) {
                    return {"Failed to get temp path", CORTEX_ERROR_FILE_IO};
                }
                char dirSeed[MAX_PATH];
                if (GetTempFileNameA(tmpPath, "cxsdr", 0, dirSeed) == 0) {
                    return {"Failed to create temp seed", CORTEX_ERROR_FILE_IO};
                }
                _unlink(dirSeed);
                std::filesystem::create_directory(dirSeed);
                tmp_dir = std::string(dirSeed);
#else
                char dirTemplate[] = "/tmp/cortexsdr_repo_XXXXXX";
                char* made = mkdtemp(dirTemplate);
                if (!made) {
                    return {"Failed to create temp directory", CORTEX_ERROR_FILE_IO};
                }
                tmp_dir = std::string(made);
#endif
                // Parse org/repo and optional revision from URL
                std::string after = clean_src;
                size_t hostPos = after.find("huggingface.co/");
                if (hostPos != std::string::npos) after = after.substr(hostPos + std::string("huggingface.co/").size());
                while (!after.empty() && after[0] == '/') after.erase(after.begin());
                std::string org, repo, revision;
                size_t s1 = after.find('/');
                if (s1 != std::string::npos) {
                    org = after.substr(0, s1);
                    std::string rest = after.substr(s1 + 1);
                    size_t s2 = rest.find('/');
                    if (s2 == std::string::npos) {
                        repo = rest;
                    } else {
                        repo = rest.substr(0, s2);
                        std::string t = rest.substr(s2 + 1);
                        if (t.rfind("tree/", 0) == 0) {
                            revision = t.substr(std::string("tree/").size());
                        }
                    }
                }
                if (!org.empty() && !repo.empty()) {
                    std::string fullrepo = org + "/" + repo;
#ifdef _WIN32
                    {
                        std::string cmd = std::string("huggingface-cli.exe download --repo-type model ") + fullrepo +
                            " --local-dir \"" + tmp_dir + "\"";
                        if (!revision.empty()) cmd += std::string(" --revision \"") + revision + "\"";
                        rc = system(cmd.c_str());
                    }
#else
                    {
                        std::string cmd = std::string("huggingface-cli download --repo-type model ") + fullrepo +
                            " --local-dir \"" + tmp_dir + "\"";
                        if (!revision.empty()) cmd += std::string(" --revision \"") + revision + "\"";
                        rc = system(cmd.c_str());
                    }
#endif
                    if (rc != 0) {
                        std::error_code ec; std::filesystem::remove_all(tmp_dir, ec);
                        return {"Failed to download Hugging Face repository (requires huggingface-cli)", CORTEX_ERROR_FILE_IO};
                    }
                    // Pick a primary model file by extension priority
                    std::vector<std::string> exts = {".gguf", ".onnx", ".pt", ".pth", ".pb"};
                    std::string chosen;
                    for (const auto& ext : exts) {
                        for (auto it = std::filesystem::recursive_directory_iterator(tmp_dir);
                             it != std::filesystem::recursive_directory_iterator(); ++it) {
                            if (!it->is_regular_file()) continue;
                            const auto& p = it->path();
                            std::string s = p.string();
                            if (s.size() >= ext.size() && s.rfind(ext) == s.size() - ext.size()) { chosen = s; break; }
                        }
                        if (!chosen.empty()) break;
                    }
                    if (chosen.empty()) {
                        std::error_code ec; std::filesystem::remove_all(tmp_dir, ec);
                        return {"Could not find a primary model file in repository", CORTEX_ERROR_FILE_IO};
                    }
                    local_path = chosen;
                    handled_download = true;
                } else {
                    // Fallback to single-file path if parsing failed
                    tmp_dir.clear();
                }
            }

            // Try curl first (curl.exe on Windows), then wget on Unix for single files
            if (!handled_download) {
#ifdef _WIN32
                {
                    std::string cmd = std::string("curl.exe -L -f -s --retry 3 --connect-timeout 10 ");
                    if (is_hf && token) {
                        cmd += std::string("-H \"Authorization: Bearer ") + token + "\" ";
                    }
                    cmd += std::string("-o \"") + tmp_file + "\" \"" + src + "\"";
                    rc = system(cmd.c_str());
                }
#else
                {
                    std::string cmd = std::string("curl -L -f -s --retry 3 --connect-timeout 10 ");
                    if (is_hf && token) {
                        cmd += std::string("-H \"Authorization: Bearer ") + token + "\" ";
                    }
                    cmd += std::string("-o ") + tmp_file + " " + src;
                    rc = system(cmd.c_str());
                    if (rc != 0) {
                        cmd = std::string("wget -q ");
                        if (is_hf && token) {
                            cmd += std::string("--header=\"Authorization: Bearer ") + token + "\" ";
                        }
                        cmd += std::string("-O ") + tmp_file + " " + src;
                        rc = system(cmd.c_str());
                    }
                }
#endif
                if (rc != 0) {
#ifdef _WIN32
                    _unlink(tmp_file.c_str());
#else
                    unlink(tmp_file.c_str());
#endif
                    return {"Failed to download remote model", CORTEX_ERROR_FILE_IO};
                }
                local_path = tmp_file;
            }
#ifdef _WIN32
            {
                std::string cmd = std::string("curl.exe -L -f -s --retry 3 --connect-timeout 10 ");
                if (is_hf && token) {
                    cmd += std::string("-H \"Authorization: Bearer ") + token + "\" ";
                }
                cmd += std::string("-o \"") + tmp_file + "\" \"" + src + "\"";
                rc = system(cmd.c_str());
            }
#else
            {
                std::string cmd = std::string("curl -L -f -s --retry 3 --connect-timeout 10 ");
                if (is_hf && token) {
                    cmd += std::string("-H \"Authorization: Bearer ") + token + "\" ";
                }
                cmd += std::string("-o ") + tmp_file + " " + src;
                rc = system(cmd.c_str());
                if (rc != 0) {
                    cmd = std::string("wget -q ");
                    if (is_hf && token) {
                        // wget supports --header to pass Authorization
                        cmd += std::string("--header=\"Authorization: Bearer ") + token + "\" ";
                    }
                    cmd += std::string("-O ") + tmp_file + " " + src;
                    rc = system(cmd.c_str());
                }
            }
#endif
            if (rc != 0) {
                #ifdef _WIN32
                _unlink(tmp_file.c_str());
                #else
                unlink(tmp_file.c_str());
                #endif
                return {"Failed to download remote model", CORTEX_ERROR_FILE_IO};
            }
            local_path = tmp_file;
        }

        // If local path is a directory (e.g., ~/.llama), try to detect a primary model file inside
        if (!is_remote) {
            std::error_code ec;
            if (std::filesystem::is_directory(local_path, ec)) {
                std::vector<std::string> exts = {".gguf", ".onnx", ".pt", ".pth", ".pb"};
                std::string chosen;
                for (const auto& ext : exts) {
                    for (auto it = std::filesystem::recursive_directory_iterator(local_path, ec);
                         it != std::filesystem::recursive_directory_iterator(); ++it) {
                        if (ec) break;
                        if (!it->is_regular_file()) continue;
                        const auto& p = it->path();
                        std::string s = p.string();
                        if (s.size() >= ext.size() && s.rfind(ext) == s.size() - ext.size()) { chosen = s; break; }
                    }
                    if (!chosen.empty()) break;
                }
                if (chosen.empty()) {
                    return {"Local directory provided but no primary model file (.gguf/.onnx/.pt/.pth/.pb) found", CORTEX_ERROR_FILE_IO};
                }
                local_path = chosen;
            }
        }

        // Prepare compression options
        CortexCompressionOptions options;
        CortexError err = cortex_compression_options_init(&options);
        if (err.code != CORTEX_SUCCESS) return err;
        options.sparsity = sparsity;
        options.verbose = 1;
        options.show_stats = 1;

        // Determine format: accept explicit, or auto-detect if empty/"auto"
        std::string format_str = (format ? std::string(format) : std::string());
        if (format_str.empty() || format_str == "auto") {
            try {
                format_str = CortexAICompression::ModelParserFactory::detectFormat(local_path);
            } catch (const std::exception& e) {
                // As a fallback, try extension-based guess or default to onnx
                std::cerr << "[SDK] Format auto-detection failed: " << e.what() << ". Assuming ONNX." << std::endl;
                format_str = "onnx";
            }
        }

        CortexCompressorHandle compressor;
        err = cortex_compressor_create(local_path.c_str(), format_str.c_str(), &options, &compressor);
        if (err.code != CORTEX_SUCCESS) {
            if (is_remote) {
#ifdef _WIN32
                if (tmp_dir.empty()) _unlink(local_path.c_str());
#else
                if (tmp_dir.empty()) unlink(local_path.c_str());
#endif
                if (!tmp_dir.empty()) { std::error_code ec; std::filesystem::remove_all(tmp_dir, ec); }
            }
            return err;
        }

        err = cortex_compressor_compress(compressor, output_path);
        cortex_compressor_free(compressor);

        if (is_remote) {
#ifdef _WIN32
            if (tmp_dir.empty()) _unlink(local_path.c_str());
#else
            if (tmp_dir.empty()) unlink(local_path.c_str());
#endif
            if (!tmp_dir.empty()) { std::error_code ec; std::filesystem::remove_all(tmp_dir, ec); }
        }
        return err;
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

// Memory management helpers exposed via SDK
CortexError cortex_inference_engine_init_memory_pool(
    CortexInferenceEngineHandle handle,
    size_t max_memory_mb)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->initializeMemoryPool(max_memory_mb);
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_enable_aggressive_memory(
    CortexInferenceEngineHandle handle,
    int enable)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end()) {
            return {"Invalid handle", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        engine->inference_engine->enableAggressiveMemoryManagement(enable != 0);
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

CortexError cortex_inference_engine_get_memory_usage(
    CortexInferenceEngineHandle handle,
    size_t* bytes)
{
    try {
        if (!handle || g_inferenceEngines.find(handle) == g_inferenceEngines.end() || !bytes) {
            return {"Invalid argument(s)", CORTEX_ERROR_INVALID_ARGUMENT};
        }
        auto engine = g_inferenceEngines[handle];
        *bytes = engine->inference_engine->getCurrentMemoryUsage();
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}
