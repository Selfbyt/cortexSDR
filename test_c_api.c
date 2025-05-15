#include "src/ai_compression/api/c_api.hpp"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <model_path> <format> <output_path>\n", argv[0]);
        printf("Example: %s mobilenetv2-7.onnx onnx compressed_model.sdr\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* format = argv[2];
    const char* output_path = argv[3];
    const char* decompressed_path = argc > 4 ? argv[4] : "decompressed_model.onnx";

    // Initialize compression options
    CortexCompressionOptions options;
    CortexError error = cortex_compression_options_init(&options);
    if (error.code != 0) {
        printf("Error initializing options: %s (code: %d)\n", error.message, error.code);
        cortex_error_free(&error);
        return 1;
    }

    // Set options
    options.verbose = 1;
    options.show_stats = 1;
    options.sparsity = 0.02f; // 2% sparsity

    // Create compressor
    printf("Creating compressor for model: %s (format: %s)\n", model_path, format);
    CortexCompressorHandle compressor;
    error = cortex_compressor_create(model_path, format, &options, &compressor);
    if (error.code != 0) {
        printf("Error creating compressor: %s (code: %d)\n", error.message, error.code);
        cortex_error_free(&error);
        return 1;
    }

    // Compress the model
    printf("Compressing model to: %s\n", output_path);
    error = cortex_compressor_compress(compressor, output_path);
    if (error.code != 0) {
        printf("Error compressing model: %s (code: %d)\n", error.message, error.code);
        cortex_error_free(&error);
        cortex_compressor_free(compressor);
        return 1;
    }

    // Get compression stats
    size_t original_size, compressed_size;
    double compression_ratio, compression_time_ms;
    error = cortex_compressor_get_stats(compressor, &original_size, &compressed_size, 
                                        &compression_ratio, &compression_time_ms);
    if (error.code != 0) {
        printf("Error getting stats: %s (code: %d)\n", error.message, error.code);
        cortex_error_free(&error);
    } else {
        printf("Compression Stats:\n");
        printf("  Original size: %zu bytes\n", original_size);
        printf("  Compressed size: %zu bytes\n", compressed_size);
        printf("  Compression ratio: %.2f:1\n", compression_ratio);
        printf("  Compression time: %.2f ms\n", compression_time_ms);
    }

    // Free the compressor
    cortex_compressor_free(compressor);

    // Now decompress the model
    printf("\nDecompressing model from: %s to: %s\n", output_path, decompressed_path);
    CortexDecompressorHandle decompressor;
    error = cortex_decompressor_create(output_path, &decompressor);
    if (error.code != 0) {
        printf("Error creating decompressor: %s (code: %d)\n", error.message, error.code);
        cortex_error_free(&error);
        return 1;
    }

    // Decompress
    error = cortex_decompressor_decompress(decompressor, output_path);
    if (error.code != 0) {
        printf("Error decompressing model: %s (code: %d)\n", error.message, error.code);
        cortex_error_free(&error);
        cortex_decompressor_free(decompressor);
        return 1;
    }

    printf("Decompression complete!\n");

    // Free the decompressor
    cortex_decompressor_free(decompressor);

    printf("Test completed successfully!\n");
    return 0;
}
