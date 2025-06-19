#include "ai_compression/api/c_api.hpp"
#include "ai_compression/SparseInferenceEngine.hpp"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>

void printUsage(const char* programName) {
    std::cout << "CortexSDR AI Model Compression CLI\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " -c <model_path> <format> <output_path> [sparsity]   (Compress)\n";
    std::cout << "  " << programName << " -i <archive_path> <input_indices_file>              (Inference)\n";
    std::cout << "\nSupported formats:\n";
    std::cout << "  - onnx: ONNX models\n";
    std::cout << "  - tensorflow: TensorFlow models (will be converted to ONNX)\n";
    std::cout << "  - pytorch: PyTorch models (will be converted to ONNX)\n";
    std::cout << "  - gguf: GGUF models\n";
    std::cout << "\nOptions:\n";
    std::cout << "  sparsity: Fraction of active bits in SDR encoding (default: 0.02 = 2%)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " -c model.onnx onnx compressed_model.sdr\n";
    std::cout << "  " << programName << " -c model.onnx onnx compressed_model.sdr 0.01\n";
    std::cout << "  " << programName << " -i compressed_model.sdr input_indices.txt\n";
}

std::vector<size_t> loadInputIndices(const char* input_path) {
    std::vector<size_t> indices;
    std::ifstream file(input_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input indices file");
    }
    size_t idx;
    while (file >> idx) {
        indices.push_back(idx);
    }
    return indices;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "-c") {
        // Compression mode
        if (argc < 5) {
            std::cerr << "Error: Compression mode requires at least 3 arguments.\n";
            printUsage(argv[0]);
            return 1;
        }

        const char* model_path = argv[2];
        const char* format = argv[3];
        const char* output_path = argv[4];
        float sparsity = 0.02f; // Default sparsity (2%)

        if (argc > 5) {
            try {
                sparsity = std::stof(argv[5]);
                if (sparsity <= 0.0f || sparsity >= 1.0f) {
                    std::cerr << "Error: Sparsity must be between 0 and 1.\n";
                    return 1;
                }
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
        std::cout << "Using sparsity: " << sparsity << " (" << (sparsity * 100) << "%)\n";
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

    } else if (mode == "-i") {
        // Inference mode
        if (argc < 4) {
            std::cerr << "Error: Inference mode requires 2 arguments.\n";
            printUsage(argv[0]);
            return 1;
        }
        const char* archive_path = argv[2];
        const char* input_path = argv[3];

        try {
            // Load input indices from file
            std::vector<size_t> input_indices = loadInputIndices(input_path);
            
            // Load model and run inference
            std::cout << "Loading compressed model from: " << archive_path << "\n";
            CortexAICompression::SDRModelLoader loader(archive_path);
            CortexAICompression::SDRInferenceEngine engine(loader);
            
            std::cout << "Running inference with " << input_indices.size() << " input indices...\n";
            auto output_indices = engine.run(input_indices);
            
            std::cout << "Inference complete. Output active indices: ";
            for (auto idx : output_indices) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: Invalid mode '" << mode << "'.\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}
