#include "ai_compression/api/c_api.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <model_path> <format> <output_path> [decompressed_path]\n";
        std::cout << "Example: " << argv[0] << " mobilenetv2-7.onnx onnx compressed_model.sdr\n";
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
        std::cout << "Error initializing options: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Set options
    options.verbose = 1;
    options.show_stats = 1;
    options.sparsity = 0.02f; // 2% sparsity

    // Create compressor
    std::cout << "Creating compressor for model: " << model_path << " (format: " << format << ")\n";
    CortexCompressorHandle compressor;
    error = cortex_compressor_create(model_path, format, &options, &compressor);
    if (error.code != 0) {
        std::cout << "Error creating compressor: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Compress the model
    std::cout << "Compressing model to: " << output_path << "\n";
    error = cortex_compressor_compress(compressor, output_path);
    if (error.code != 0) {
        std::cout << "Error compressing model: " << error.message << " (code: " << error.code << ")\n";
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
        std::cout << "Error getting stats: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
    } else {
        std::cout << "Compression Stats:\n";
        std::cout << "  Original size: " << original_size << " bytes\n";
        std::cout << "  Compressed size: " << compressed_size << " bytes\n";
        std::cout << "  Compression ratio: " << compression_ratio << ":1\n";
        std::cout << "  Compression time: " << compression_time_ms << " ms\n";
    }

    // Free the compressor
    cortex_compressor_free(compressor);

    // Now decompress the model
    std::cout << "\nDecompressing model from: " << output_path << " to: " << decompressed_path << "\n";
    CortexDecompressorHandle decompressor;
    error = cortex_decompressor_create(output_path, &decompressor);
    if (error.code != 0) {
        std::cout << "Error creating decompressor: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Decompress - Pass the compressed file path (output_path in this script) and the target decompressed path
    error = cortex_decompressor_decompress(decompressor, output_path, decompressed_path);
    if (error.code != 0) {
        std::cout << "Error decompressing model: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        cortex_decompressor_free(decompressor);
        return 1;
    }

    std::cout << "Decompression complete!\n";

    // Free the decompressor
    cortex_decompressor_free(decompressor);

    std::cout << "Test completed successfully!\n";
    return 0;
}
