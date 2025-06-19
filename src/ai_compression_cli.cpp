#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "ai_compression/api/c_api.hpp"
#include "ai_compression/SparseInferenceEngine.hpp"
#include <cstring>

// Function prototypes
void printUsage(const char* programName);
int compressModel(const char* model_path, const char* format, const char* output_path, float sparsity);
int decompressModel(const char* compressed_path, const char* output_path, float sparsity);
int runInference(const char* archive_path, const char* input_indices_file);
std::vector<size_t> loadInputIndices(const char* input_path);

void printUsage(const char* programName) {
    std::cout << "CortexSDR AI Model Compression CLI\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " -c <model_path> <format> <output_path> [sparsity]   (Compress)\n";
    std::cout << "  " << programName << " -d <compressed_path> <output_path> [sparsity]       (Decompress)\n";
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
    std::cout << "  " << programName << " -d compressed_model.sdr decompressed_model.onnx\n";
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
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "-c") {
        if (argc < 5) {
            printUsage(argv[0]);
            return 1;
        }
        const char* model_path = argv[2];
        const char* format = argv[3];
        const char* output_path = argv[4];
        float sparsity = 0.02f;
        if (argc > 5) {
            sparsity = std::stof(argv[5]);
        }
        return compressModel(model_path, format, output_path, sparsity);
    } else if (mode == "-d") {
        if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        const char* compressed_path = argv[2];
        const char* output_path = argv[3];
        float sparsity = 0.02f;
        if (argc > 4) {
            sparsity = std::stof(argv[4]);
        }
        return decompressModel(compressed_path, output_path, sparsity);
    } else if (mode == "-i") {
        if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        const char* archive_path = argv[2];
        const char* input_indices_file = argv[3];
        return runInference(archive_path, input_indices_file);
    } else {
        printUsage(argv[0]);
        return 1;
    }
}

int compressModel(const char* model_path, const char* format, const char* output_path, float sparsity) {
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
    return 0;
}

int decompressModel(const char* compressed_path, const char* output_path, float sparsity) {
    std::cout << "Decompressing model from: " << compressed_path << " to: " << output_path << std::endl;
    std::cout << "Using sparsity: " << sparsity << " (" << (sparsity * 100) << "%)" << std::endl;

    CortexDecompressorHandle handle;
    CortexError error = cortex_decompressor_create(compressed_path, &handle, sparsity);
    if (error.code != 0) {
        std::cerr << "Error creating decompressor: " << error.message << std::endl;
        cortex_error_free(&error);
        return 1;
    }

    error = cortex_decompressor_decompress(handle, compressed_path, output_path);
    if (error.code != 0) {
        std::cerr << "Error decompressing model: " << error.message << std::endl;
        cortex_error_free(&error);
        cortex_decompressor_free(handle);
        return 1;
    }

    cortex_decompressor_free(handle);
    std::cout << "Decompression complete." << std::endl;
    return 0;
}

int runInference(const char* archive_path, const char* input_indices_file) {
    try {
        // Load input indices from file
        std::vector<size_t> input_indices = loadInputIndices(input_indices_file);
        
        // Load model and run inference
        std::cout << "Loading compressed model from: " << archive_path << "\n";
        CortexAICompression::SDRModelLoader loader(archive_path);
        CortexAICompression::SDRInferenceEngine engine(loader);
        
        // Convert input indices to a float tensor (1.0 for active indices, 0.0 for others)
        std::vector<float> input_tensor(1000, 0.0f);  // Assuming 1000-dimensional input
        for (auto idx : input_indices) {
            if (idx < input_tensor.size()) {
                input_tensor[idx] = 1.0f;
            }
        }
        
        std::cout << "Running inference with " << input_indices.size() << " active indices...\n";
        auto output_tensor = engine.run(input_tensor);
        
        std::cout << "Inference complete. Output tensor size: " << output_tensor.size() << std::endl;
        std::cout << "First few output values: ";
        for (size_t i = 0; i < std::min(size_t(5), output_tensor.size()); ++i) {
            std::cout << output_tensor[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
