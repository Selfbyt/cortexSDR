/**
 * @file calculate_error.cpp
 * @brief Calculate reconstruction error between original and compressed model
 * 
 * This tool:
 * 1. Loads original GGUF model
 * 2. Loads compressed .sdr model
 * 3. Compares weight tensors
 * 4. Calculates MSE, MAE, and relative error
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include "src/ai_compression/core/AIDecompressor.hpp"
#include "src/ai_compression/parsers/GGUFModelParser.hpp"
#include "src/ai_compression/strategies/AdaptiveSDRStrategy.hpp"

using namespace CortexAICompression;

struct ErrorMetrics {
    double mse = 0.0;           // Mean Squared Error
    double mae = 0.0;           // Mean Absolute Error
    double max_error = 0.0;     // Maximum absolute error
    double relative_error = 0.0; // Relative error (normalized by original magnitude)
    size_t total_elements = 0;
    size_t zero_elements_original = 0;
    size_t zero_elements_reconstructed = 0;
};

ErrorMetrics calculateError(const std::vector<std::byte>& original, 
                            const std::vector<std::byte>& reconstructed) {
    ErrorMetrics metrics;
    
    if (original.size() != reconstructed.size()) {
        std::cerr << "Size mismatch: " << original.size() << " vs " << reconstructed.size() << std::endl;
        return metrics;
    }
    
    // Assume float32 data
    size_t num_elements = original.size() / sizeof(float);
    metrics.total_elements = num_elements;
    
    const float* orig_data = reinterpret_cast<const float*>(original.data());
    const float* recon_data = reinterpret_cast<const float*>(reconstructed.data());
    
    double sum_squared_error = 0.0;
    double sum_absolute_error = 0.0;
    double sum_original_squared = 0.0;
    
    for (size_t i = 0; i < num_elements; i++) {
        float orig = orig_data[i];
        float recon = recon_data[i];
        
        if (orig == 0.0f) metrics.zero_elements_original++;
        if (recon == 0.0f) metrics.zero_elements_reconstructed++;
        
        double error = std::abs(orig - recon);
        double squared_error = error * error;
        
        sum_squared_error += squared_error;
        sum_absolute_error += error;
        sum_original_squared += orig * orig;
        
        if (error > metrics.max_error) {
            metrics.max_error = error;
        }
    }
    
    metrics.mse = sum_squared_error / num_elements;
    metrics.mae = sum_absolute_error / num_elements;
    
    // Relative error (RMSE / RMS of original)
    double rmse = std::sqrt(metrics.mse);
    double rms_original = std::sqrt(sum_original_squared / num_elements);
    metrics.relative_error = (rms_original > 0) ? (rmse / rms_original) : 0.0;
    
    return metrics;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <original.gguf> <compressed.sdr> <sparsity>\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " model.gguf model.sdr 0.5\n";
        return 1;
    }
    
    std::string original_path = argv[1];
    std::string compressed_path = argv[2];
    float sparsity = std::stof(argv[3]);
    
    std::cout << "===========================================\n";
    std::cout << "Model Reconstruction Error Analysis\n";
    std::cout << "===========================================\n";
    std::cout << "Original model: " << original_path << "\n";
    std::cout << "Compressed model: " << compressed_path << "\n";
    std::cout << "Sparsity: " << (sparsity * 100) << "%\n\n";
    
    // Parse original model
    std::cout << "Loading original model...\n";
    GGUFModelParser parser;
    std::vector<ModelSegment> original_segments;
    try {
        original_segments = parser.parse(original_path);
        std::cout << "  Loaded " << original_segments.size() << " segments\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading original model: " << e.what() << "\n";
        return 1;
    }
    
    // Decompress compressed model
    std::cout << "\nLoading compressed model...\n";
    auto decompressor = std::make_unique<AIDecompressor>();
    auto adaptiveStrategy = std::make_shared<AdaptiveSDRStrategy>(sparsity);
    decompressor->registerStrategy(1, adaptiveStrategy);
    
    std::ifstream compressed_file(compressed_path, std::ios::binary);
    if (!compressed_file) {
        std::cerr << "Error opening compressed file\n";
        return 1;
    }
    
    class SegmentCollector : public ISegmentHandler {
    public:
        void handleSegment(ModelSegment segment) override {
            segments.push_back(std::move(segment));
        }
        std::vector<ModelSegment> segments;
    };
    
    SegmentCollector collector;
    try {
        decompressor->decompressModelStream(compressed_file, collector);
        std::cout << "  Decompressed " << collector.segments.size() << " segments\n";
    } catch (const std::exception& e) {
        std::cerr << "Error decompressing model: " << e.what() << "\n";
        return 1;
    }
    
    // Compare weight tensors
    std::cout << "\n===========================================\n";
    std::cout << "Weight Tensor Comparison\n";
    std::cout << "===========================================\n\n";
    
    size_t compared_tensors = 0;
    double total_relative_error = 0.0;
    
    for (const auto& orig_seg : original_segments) {
        // Only compare weight tensors
        if (!orig_seg.isWeightTensor()) continue;
        
        // Find matching segment in decompressed model
        auto it = std::find_if(collector.segments.begin(), collector.segments.end(),
            [&](const ModelSegment& seg) { return seg.name == orig_seg.name; });
        
        if (it == collector.segments.end()) {
            std::cout << "Warning: Segment '" << orig_seg.name << "' not found in compressed model\n";
            continue;
        }
        
        ErrorMetrics metrics = calculateError(orig_seg.data, it->data);
        
        std::cout << "Tensor: " << orig_seg.name << "\n";
        std::cout << "  Elements: " << metrics.total_elements << "\n";
        std::cout << "  MSE: " << std::scientific << metrics.mse << "\n";
        std::cout << "  MAE: " << metrics.mae << "\n";
        std::cout << "  Max Error: " << metrics.max_error << "\n";
        std::cout << "  Relative Error: " << std::fixed << std::setprecision(2) 
                  << (metrics.relative_error * 100) << "%\n";
        std::cout << "  Zeros (original): " << metrics.zero_elements_original 
                  << " (" << (100.0 * metrics.zero_elements_original / metrics.total_elements) << "%)\n";
        std::cout << "  Zeros (reconstructed): " << metrics.zero_elements_reconstructed 
                  << " (" << (100.0 * metrics.zero_elements_reconstructed / metrics.total_elements) << "%)\n";
        std::cout << "\n";
        
        compared_tensors++;
        total_relative_error += metrics.relative_error;
    }
    
    if (compared_tensors > 0) {
        double avg_relative_error = total_relative_error / compared_tensors;
        std::cout << "===========================================\n";
        std::cout << "Summary\n";
        std::cout << "===========================================\n";
        std::cout << "Compared tensors: " << compared_tensors << "\n";
        std::cout << "Average relative error: " << std::fixed << std::setprecision(2) 
                  << (avg_relative_error * 100) << "%\n";
        std::cout << "Sparsity: " << (sparsity * 100) << "%\n";
        std::cout << "===========================================\n";
    }
    
    return 0;
}
