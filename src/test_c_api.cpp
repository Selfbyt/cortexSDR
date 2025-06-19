#include "ai_compression/api/c_api.hpp"
#include <iostream>
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <format> <output_path> [sparsity]\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* format = argv[2];
    const char* output_path = argv[3];
    float sparsity = 0.02f; // Default sparsity (2%)

    if (argc > 4) {
        try {
            sparsity = std::stof(argv[4]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid sparsity value.\n";
            return 1;
        }
    }

    // Initialize compression options
    CortexCompressionOptions options;
    CortexError error = cortex_compression_options_init(&options);
    if (error.code != 0) {
        std::cerr << "Error initializing options: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Set options
    options.verbose = 1;
    options.show_stats = 1;
    options.sparsity = sparsity;

    // Create compressor
    std::cout << "Creating compressor for model: " << model_path << " (format: " << format << ")\n";
    CortexCompressorHandle compressor;
    error = cortex_compressor_create(model_path, format, &options, &compressor);
    if (error.code != 0) {
        std::cerr << "Error creating compressor: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Compress the model
    std::cout << "Compressing model to: " << output_path << "\n";
    error = cortex_compressor_compress(compressor, output_path);
    if (error.code != 0) {
        std::cerr << "Error compressing model: " << error.message << " (code: " << error.code << ")\n";
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
        std::cerr << "Error getting stats: " << error.message << " (code: " << error.code << ")\n";
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
    std::cout << "Compression complete.\n";

    return 0;
}
