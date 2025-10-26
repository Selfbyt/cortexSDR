#ifndef CORTEX_SDK_H
#define CORTEX_SDK_H

#include "c_api.hpp"
#include "../../../include/cortex_sdk_export.h"
#include <stddef.h>
#include <stdint.h>

// Include C API definitions to avoid duplication
#ifdef __cplusplus
#include "c_api.hpp"
extern "C" {
#else
#include "c_api.h"
#endif

/**
 * @file cortex_sdk.h
 * @brief Unified SDK for CortexSDR AI compression and inference
 * 
 * This SDK combines the compression/decompression capabilities with
 * the sparse inference engine to provide a complete solution for
 * AI model deployment across different platforms.
 */

/* ===== Error Handling ===== */

// Error codes are defined in c_api.hpp

/* ===== Opaque Handle Types ===== */

// Compressor and Decompressor handles are defined in c_api.hpp

/**
 * Opaque handle for inference engine
 */
typedef struct CortexInferenceEngine* CortexInferenceEngineHandle;

/**
 * Initialize compression options with defaults
 */
CORTEXSDR_API CortexError cortex_compression_options_init(CortexCompressionOptions* options);

/* ===== Compression Functions ===== */

/**
 * Create a compressor instance
 */
CORTEXSDR_API CortexError cortex_compressor_create(
    const char* model_path,
    const char* format,
    const CortexCompressionOptions* options,
    CortexCompressorHandle* handle
);

/**
 * Compress a model
 */
CORTEXSDR_API CortexError cortex_compressor_compress(
    CortexCompressorHandle handle,
    const char* output_path
);

/**
 * Get compression statistics
 */
CORTEXSDR_API CortexError cortex_compressor_get_stats(
    CortexCompressorHandle handle,
    size_t* original_size,
    size_t* compressed_size,
    double* compression_ratio,
    double* compression_time_ms
);

/**
 * Free a compressor instance
 */
CORTEXSDR_API CortexError cortex_compressor_free(CortexCompressorHandle handle);

/* ===== Decompression Functions ===== */

/**
 * Create a decompressor instance
 */
CORTEXSDR_API CortexError cortex_decompressor_create(
    const char* compressed_path,
    CortexDecompressorHandle* handle,
    float sparsity
);

/**
 * Decompress a model
 */
CORTEXSDR_API CortexError cortex_decompressor_decompress(
    CortexDecompressorHandle handle,
    const char* compressed_path,
    const char* output_path
);

/**
 * Free a decompressor instance
 */
CORTEXSDR_API CortexError cortex_decompressor_free(CortexDecompressorHandle handle);

/* ===== Inference Engine Functions ===== */

/**
 * Create an inference engine from a compressed model
 */
CORTEXSDR_API CortexError cortex_inference_engine_create(
    const char* compressed_model_path,
    CortexInferenceEngineHandle* handle
);

/**
 * Set batch size for inference
 */
CORTEXSDR_API CortexError cortex_inference_engine_set_batch_size(
    CortexInferenceEngineHandle handle,
    size_t batch_size
);

/**
 * Enable/disable dropout for inference
 */
CORTEXSDR_API CortexError cortex_inference_engine_enable_dropout(
    CortexInferenceEngineHandle handle,
    int enable
);

/**
 * Set inference mode (training=1, inference=0)
 */
CORTEXSDR_API CortexError cortex_inference_engine_set_mode(
    CortexInferenceEngineHandle handle,
    int training_mode
);

/**
 * Initialize an internal memory pool for large-model inference
 */
CORTEXSDR_API CortexError cortex_inference_engine_init_memory_pool(
    CortexInferenceEngineHandle handle,
    size_t max_memory_mb
);

/**
 * Enable or disable aggressive memory management (cache eviction between layers)
 */
CORTEXSDR_API CortexError cortex_inference_engine_enable_aggressive_memory(
    CortexInferenceEngineHandle handle,
    int enable
);

/**
 * Get current memory usage in bytes (if memory pool is enabled)
 */
CORTEXSDR_API CortexError cortex_inference_engine_get_memory_usage(
    CortexInferenceEngineHandle handle,
    size_t* bytes
);

/**
 * Get last run benchmark stats as JSON (caller must free with cortex_free_string)
 */
CORTEXSDR_API CortexError cortex_inference_engine_get_last_run_stats_json(
    CortexInferenceEngineHandle handle,
    char** out_json
);

/** Free strings returned by SDK functions */
CORTEXSDR_API void cortex_free_string(char* s);

/**
 * Run inference on input data
 * 
 * @param handle The inference engine handle
 * @param input_data Pointer to input data (float array)
 * @param input_size Size of input data array
 * @param output_data Pointer to output buffer (must be pre-allocated)
 * @param output_size Size of output buffer
 * @param actual_output_size Actual size of output data written
 */
CORTEXSDR_API CortexError cortex_inference_engine_run(
    CortexInferenceEngineHandle handle,
    const float* input_data,
    size_t input_size,
    float* output_data,
    size_t output_size,
    size_t* actual_output_size
);

/**
 * Run inference on a specific layer
 */
CORTEXSDR_API CortexError cortex_inference_engine_run_layer(
    CortexInferenceEngineHandle handle,
    const char* layer_name,
    const float* input_data,
    size_t input_size,
    float* output_data,
    size_t output_size,
    size_t* actual_output_size
);

/**
 * Free an inference engine instance
 */
CORTEXSDR_API CortexError cortex_inference_engine_free(
    CortexInferenceEngineHandle handle
);

/**
 * Get version information
 */
CORTEXSDR_API const char* cortex_sdk_version();

/* ===== Convenience: Compress While Downloading ===== */

/**
 * Download a remote model and compress to .sdr in one call.
 * The URL can be http(s) or a local file path. For http(s), this function
 * attempts to use an available downloader (curl or wget). On success, the
 * temporary file is removed.
 *
 * @param url_or_path Remote URL (http/https) or local file path to model
 * @param format Model format hint (e.g., "onnx", "gguf", "tensorflow")
 * @param output_path Destination .sdr path
 * @param sparsity SDR sparsity (e.g., 0.02 for 2%)
 */
CORTEXSDR_API CortexError cortex_compress_from_url(
    const char* url_or_path,
    const char* format,
    const char* output_path,
    float sparsity
);

/* ===== Model Introspection ===== */

/**
 * Inspect a compressed .sdr archive for tokenizer assets.
 * If present, sets out_has_tokenizer to 1 and returns a best-effort
 * tokenizer type string via out_tokenizer_type (owned by caller via cortex_free_string),
 * such as "sentencepiece", "gpt2-bpe", or "unknown".
 */
CORTEXSDR_API CortexError cortex_archive_get_tokenizer_info(
    const char* archive_path,
    int* out_has_tokenizer,
    char** out_tokenizer_type
);

#ifdef __cplusplus
}
#endif

#endif /* CORTEX_SDK_H */
