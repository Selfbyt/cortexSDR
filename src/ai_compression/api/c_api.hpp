#ifndef C_API_HPP
#define C_API_HPP

#include "../core/AICompressor.hpp"
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef struct CortexCompressor* CortexCompressorHandle;

// Error handling
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
CortexError cortex_compression_options_init(CortexCompressionOptions* options);

// Create a compressor instance
CortexError cortex_compressor_create(
    const char* model_path,
    const char* format,
    const CortexCompressionOptions* options,
    CortexCompressorHandle* handle
);

// Compress a model
CortexError cortex_compressor_compress(
    CortexCompressorHandle handle,
    const char* output_path
);

// Get compression statistics
CortexError cortex_compressor_get_stats(
    CortexCompressorHandle handle,
    size_t* original_size,
    size_t* compressed_size,
    double* compression_ratio,
    double* compression_time_ms
);

// Free a compressor instance
CortexError cortex_compressor_free(CortexCompressorHandle handle);

// Free error message
void cortex_error_free(CortexError* error);

#ifdef __cplusplus
}
#endif

#endif // C_API_HPP
