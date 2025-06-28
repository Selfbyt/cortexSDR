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
CortexError cortex_compression_options_init(CortexCompressionOptions* options);

/* ===== Compression Functions ===== */

/**
 * Create a compressor instance
 */
CortexError cortex_compressor_create(
    const char* model_path,
    const char* format,
    const CortexCompressionOptions* options,
    CortexCompressorHandle* handle
);

/**
 * Compress a model
 */
CortexError cortex_compressor_compress(
    CortexCompressorHandle handle,
    const char* output_path
);

/**
 * Get compression statistics
 */
CortexError cortex_compressor_get_stats(
    CortexCompressorHandle handle,
    size_t* original_size,
    size_t* compressed_size,
    double* compression_ratio,
    double* compression_time_ms
);

/**
 * Free a compressor instance
 */
CortexError cortex_compressor_free(CortexCompressorHandle handle);

/* ===== Decompression Functions ===== */

/**
 * Create a decompressor instance
 */
CortexError cortex_decompressor_create(
    const char* compressed_path,
    CortexDecompressorHandle* handle,
    float sparsity
);

/**
 * Decompress a model
 */
CortexError cortex_decompressor_decompress(
    CortexDecompressorHandle handle,
    const char* compressed_path,
    const char* output_path
);

/**
 * Free a decompressor instance
 */
CortexError cortex_decompressor_free(CortexDecompressorHandle handle);

/* ===== Inference Engine Functions ===== */

/**
 * Create an inference engine from a compressed model
 */
CortexError cortex_inference_engine_create(
    const char* compressed_model_path,
    CortexInferenceEngineHandle* handle
);

/**
 * Set batch size for inference
 */
CortexError cortex_inference_engine_set_batch_size(
    CortexInferenceEngineHandle handle,
    size_t batch_size
);

/**
 * Enable/disable dropout for inference
 */
CortexError cortex_inference_engine_enable_dropout(
    CortexInferenceEngineHandle handle,
    int enable
);

/**
 * Set inference mode (training=1, inference=0)
 */
CortexError cortex_inference_engine_set_mode(
    CortexInferenceEngineHandle handle,
    int training_mode
);

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
CortexError cortex_inference_engine_run(
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
CortexError cortex_inference_engine_run_layer(
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
CortexError cortex_inference_engine_free(
    CortexInferenceEngineHandle handle
);

/**
 * Get version information
 */
const char* cortex_sdk_version();

#ifdef __cplusplus
}
#endif

#endif /* CORTEX_SDK_H */
