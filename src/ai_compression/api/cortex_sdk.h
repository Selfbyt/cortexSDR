#ifndef CORTEX_SDK_H
#define CORTEX_SDK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
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

/**
 * Error codes for SDK functions
 */
typedef enum {
    CORTEX_SUCCESS = 0,
    CORTEX_ERROR_INVALID_ARGUMENT = -1,
    CORTEX_ERROR_FILE_IO = -2,
    CORTEX_ERROR_MEMORY = -3,
    CORTEX_ERROR_UNSUPPORTED_FORMAT = -4,
    CORTEX_ERROR_COMPRESSION = -5,
    CORTEX_ERROR_DECOMPRESSION = -6,
    CORTEX_ERROR_INFERENCE = -7,
    CORTEX_ERROR_UNKNOWN = -99
} CortexStatus;

/**
 * Error information structure
 */
typedef struct {
    const char* message;
    int code;
} CortexError;

/**
 * Free error message
 */
void cortex_error_free(CortexError* error);

/* ===== Opaque Handle Types ===== */

/**
 * Opaque handle for compressor
 */
typedef struct CortexCompressor* CortexCompressorHandle;

/**
 * Opaque handle for decompressor
 */
typedef void* CortexDecompressorHandle;

/**
 * Opaque handle for inference engine
 */
typedef struct CortexInferenceEngine* CortexInferenceEngineHandle;

/* ===== Compression Options ===== */

/**
 * Compression options structure
 */
typedef struct {
    size_t num_threads;
    int verbose;
    int show_stats;
    int use_delta_encoding;
    int use_rle;
    int compression_level;
    int use_quantization;      // Whether to use quantization for floating-point tensors
    int quantization_bits;     // Number of bits for quantization (e.g., 8 for int8)
    float sparsity;           // Fraction of active bits in SDR encoding (default 0.02 = 2%)
} CortexCompressionOptions;

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
