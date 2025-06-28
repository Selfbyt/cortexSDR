#include "cortex_sdk.h"
#include "c_api.hpp"
#include "../SparseInferenceEngine.hpp"
#include "../core/AICompressor.hpp"
#include "../core/AIDecompressor.hpp"
#include <string>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace CortexAICompression;

// Version information
#define CORTEX_SDK_VERSION "1.0.0"

// Internal structures for opaque handles
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

// Error handling
void cortex_error_free(CortexError* error) {
    if (error && error->message) {
        delete[] error->message;
        error->message = nullptr;
    }
}

// Compression options initialization
CortexError cortex_compression_options_init(CortexCompressionOptions* options) {
    try {
        if (!options) {
            return {"Invalid options pointer", CORTEX_ERROR_INVALID_ARGUMENT};
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
        return {nullptr, CORTEX_SUCCESS};
    } catch (const std::exception& e) {
        return convert_exception(e);
    }
}

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
    
    extern CortexError cortex_decompressor_create(const char* compressed_path,
                                          CortexDecompressorHandle* handle,
                                          float sparsity);
    
    extern CortexError cortex_decompressor_decompress(CortexDecompressorHandle handle,
                                               const char* compressed_path,
                                               const char* output_path);
    
    extern CortexError cortex_decompressor_free(CortexDecompressorHandle handle);
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
