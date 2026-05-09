#ifndef C_API_HPP
#define C_API_HPP

#include "../core/AICompressor.hpp"
#include <cstddef>
#include <cstdint>
#include "../../../include/cortex_sdk_export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef struct CortexCompressor* CortexCompressorHandle;
typedef void* CortexDecompressorHandle;

// Error handling
typedef enum {
    CORTEX_OK = 0,
    CORTEX_ERR_INVALID_ARG = 1,
    CORTEX_ERR_FILE_NOT_FOUND = 2,
    CORTEX_ERR_UNSUPPORTED_FORMAT = 3,
    CORTEX_ERR_INTERNAL = 100
} CortexErrorCode;

typedef struct {
    const char* message;
    int code;
} CortexError;

// Compression options
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

// Initialize compression options with defaults
CORTEXSDR_API CortexError cortex_compression_options_init(CortexCompressionOptions* options);

// Create a compressor instance
CORTEXSDR_API CortexError cortex_compressor_create(
    const char* model_path,
    const char* format,
    const CortexCompressionOptions* options,
    CortexCompressorHandle* handle
);

// Check if a model path is a multi-part model and get all parts
// Returns the number of parts found (1 for single file, >1 for multi-part)
// If parts_out is not NULL, it will be filled with the paths (caller must free each string and the array)
CORTEXSDR_API int cortex_model_get_parts(
    const char* model_path,
    char*** parts_out,
    int* num_parts
);

// Add additional file to compressor (for multi-part models)
CORTEXSDR_API CortexError cortex_compressor_add_file(
    CortexCompressorHandle handle,
    const char* model_path
);

// Compress a model
CORTEXSDR_API CortexError cortex_compressor_compress(
    CortexCompressorHandle handle,
    const char* output_path
);

// Get compression statistics
CORTEXSDR_API CortexError cortex_compressor_get_stats(
    CortexCompressorHandle handle,
    size_t* original_size,
    size_t* compressed_size,
    double* compression_ratio,
    double* compression_time_ms
);

// Free a compressor instance
CORTEXSDR_API CortexError cortex_compressor_free(CortexCompressorHandle handle);

// Create a decompressor instance
CORTEXSDR_API CortexError cortex_decompressor_create(
    const char* compressed_path,
    CortexDecompressorHandle* handle,
    float sparsity
);

// Extract archive contents to a directory bundle
CORTEXSDR_API CortexError cortex_decompressor_decompress(
    CortexDecompressorHandle handle,
    const char* compressed_path,
    const char* output_path
);

// Extract archive segments to a directory bundle
CORTEXSDR_API CortexError cortex_archive_extract(
    const char* compressed_path,
    const char* output_dir,
    float sparsity
);

// Free a decompressor instance
CORTEXSDR_API CortexError cortex_decompressor_free(CortexDecompressorHandle handle);

// Free error message
CORTEXSDR_API void cortex_error_free(CortexError* error);

// Get a human-readable error string for a given error code
CORTEXSDR_API const char* cortex_error_string(int code);

#ifdef __cplusplus
}
#endif

#endif // C_API_HPP
