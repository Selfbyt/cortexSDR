/**
 * @file multipart_compression_example.cpp
 * @brief Example showing how to compress multi-part models using the CortexSDR API
 * 
 * This example demonstrates:
 * 1. Automatic detection of multi-part models (e.g., from Hugging Face)
 * 2. Compressing all parts into a single unified .sdr archive
 * 3. Using the C API for compression
 */

#include "ai_compression/api/c_api.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <output.sdr> [sparsity]\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " model-00001-of-00002.gguf model.sdr 0.02\n";
        std::cout << "\nNote: If the model is multi-part, all parts will be automatically detected and compressed.\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* output_path = argv[2];
    float sparsity = 0.02f;
    
    if (argc > 3) {
        sparsity = std::stof(argv[3]);
    }

    std::cout << "===========================================\n";
    std::cout << "CortexSDR Multi-Part Model Compression\n";
    std::cout << "===========================================\n\n";

    // Step 1: Check if this is a multi-part model
    char** parts = nullptr;
    int num_parts = 0;
    int detected_parts = cortex_model_get_parts(model_path, &parts, &num_parts);
    
    if (detected_parts > 1) {
        std::cout << "✓ Detected multi-part model with " << num_parts << " parts:\n";
        for (int i = 0; i < num_parts; i++) {
            std::cout << "  Part " << (i + 1) << ": " << parts[i] << "\n";
        }
        std::cout << "\n";
        
        // Free the parts array
        for (int i = 0; i < num_parts; i++) {
            delete[] parts[i];
        }
        delete[] parts;
    } else {
        std::cout << "✓ Single-file model detected\n\n";
    }

    // Step 2: Initialize compression options
    CortexCompressionOptions options;
    CortexError error = cortex_compression_options_init(&options);
    if (error.code != 0) {
        std::cerr << "Error initializing options: " << error.message << "\n";
        cortex_error_free(&error);
        return 1;
    }

    options.verbose = 1;
    options.show_stats = 1;
    options.sparsity = sparsity;

    std::cout << "Compression settings:\n";
    std::cout << "  Sparsity: " << (sparsity * 100) << "%\n";
    std::cout << "  Format: gguf\n\n";

    // Step 3: Create compressor (automatically detects and handles multi-part models)
    std::cout << "Creating compressor...\n";
    CortexCompressorHandle compressor;
    error = cortex_compressor_create(model_path, "gguf", &options, &compressor);
    if (error.code != 0) {
        std::cerr << "Error creating compressor: " << error.message << "\n";
        cortex_error_free(&error);
        return 1;
    }

    // Step 4: Compress (all parts are automatically included)
    std::cout << "\nCompressing to: " << output_path << "\n";
    error = cortex_compressor_compress(compressor, output_path);
    if (error.code != 0) {
        std::cerr << "Error compressing model: " << error.message << "\n";
        cortex_error_free(&error);
        cortex_compressor_free(compressor);
        return 1;
    }

    // Step 5: Get compression statistics
    size_t original_size, compressed_size;
    double compression_ratio, compression_time_ms;
    error = cortex_compressor_get_stats(compressor, &original_size, &compressed_size,
                                       &compression_ratio, &compression_time_ms);
    
    if (error.code == 0) {
        std::cout << "\n===========================================\n";
        std::cout << "Compression Complete!\n";
        std::cout << "===========================================\n";
        std::cout << "Original size:     " << (original_size / 1024.0 / 1024.0) << " MB\n";
        std::cout << "Compressed size:   " << (compressed_size / 1024.0 / 1024.0) << " MB\n";
        std::cout << "Compression ratio: " << compression_ratio << ":1\n";
        std::cout << "Time taken:        " << compression_time_ms << " ms\n";
        std::cout << "===========================================\n";
    }

    // Step 6: Cleanup
    cortex_compressor_free(compressor);

    return 0;
}
