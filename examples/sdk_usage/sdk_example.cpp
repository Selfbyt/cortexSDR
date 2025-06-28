#include "ai_compression/api/cortex_sdk.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>

// Helper function to print error information
void print_error(const CortexError& error) {
    if (error.code != CORTEX_SUCCESS) {
        std::cerr << "Error " << error.code << ": " << (error.message ? error.message : "Unknown error") << std::endl;
        cortex_error_free(const_cast<CortexError*>(&error));
    }
}

// Helper function to print a vector of floats
void print_vector(const std::vector<float>& vec, const std::string& name, int max_items = 10) {
    std::cout << name << " (size: " << vec.size() << "): ";
    int count = std::min(static_cast<int>(vec.size()), max_items);
    for (int i = 0; i < count; ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i] << " ";
    }
    if (vec.size() > max_items) {
        std::cout << "...";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "CortexSDR SDK Example - Version: " << cortex_sdk_version() << std::endl;
    std::cout << "======================================================" << std::endl;
    
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <output_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " model.onnx compressed_model.cortex" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* output_path = argv[2];
    
    std::cout << "Input model: " << model_path << std::endl;
    std::cout << "Output path: " << output_path << std::endl;
    
    // Step 1: Initialize compression options
    CortexCompressionOptions options;
    CortexError error = cortex_compression_options_init(&options);
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
        return 1;
    }
    
    // Configure options
    options.verbose = 1;
    options.show_stats = 1;
    options.sparsity = 0.02f; // 2% sparsity
    
    std::cout << "\n1. Compressing model..." << std::endl;
    
    // Step 2: Create compressor
    CortexCompressorHandle compressor = nullptr;
    error = cortex_compressor_create(model_path, "onnx", &options, &compressor);
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
        return 1;
    }
    
    // Step 3: Compress model
    error = cortex_compressor_compress(compressor, output_path);
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
        cortex_compressor_free(compressor);
        return 1;
    }
    
    // Step 4: Get compression statistics
    size_t original_size = 0;
    size_t compressed_size = 0;
    double compression_ratio = 0.0;
    double compression_time_ms = 0.0;
    
    error = cortex_compressor_get_stats(compressor, &original_size, &compressed_size, 
                                       &compression_ratio, &compression_time_ms);
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
    } else {
        std::cout << "Compression stats:" << std::endl;
        std::cout << "  Original size: " << original_size << " bytes" << std::endl;
        std::cout << "  Compressed size: " << compressed_size << " bytes" << std::endl;
        std::cout << "  Compression ratio: " << std::fixed << std::setprecision(2) 
                  << compression_ratio << "x" << std::endl;
        std::cout << "  Compression time: " << compression_time_ms << " ms" << std::endl;
    }
    
    // Free compressor
    cortex_compressor_free(compressor);
    
    std::cout << "\n2. Creating inference engine..." << std::endl;
    
    // Step 5: Create inference engine from compressed model
    CortexInferenceEngineHandle engine = nullptr;
    error = cortex_inference_engine_create(output_path, &engine);
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
        return 1;
    }
    
    // Step 6: Configure inference engine
    error = cortex_inference_engine_set_batch_size(engine, 1);
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
    }
    
    error = cortex_inference_engine_enable_dropout(engine, 0); // Disable dropout for inference
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
    }
    
    error = cortex_inference_engine_set_mode(engine, 0); // Set to inference mode (not training)
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
    }
    
    std::cout << "\n3. Running inference..." << std::endl;
    
    // Step 7: Create sample input data (this would normally come from your application)
    // For this example, we'll create a simple vector of 10 values
    std::vector<float> input_data(10, 1.0f); // 10 elements, all set to 1.0
    
    // Prepare output buffer
    const size_t max_output_size = 1000; // Adjust based on your model's expected output size
    std::vector<float> output_data(max_output_size);
    size_t actual_output_size = 0;
    
    // Step 8: Run inference
    error = cortex_inference_engine_run(
        engine,
        input_data.data(),
        input_data.size(),
        output_data.data(),
        output_data.size(),
        &actual_output_size
    );
    
    if (error.code != CORTEX_SUCCESS) {
        print_error(error);
    } else {
        // Resize output vector to actual size
        output_data.resize(actual_output_size);
        
        // Print input and output
        print_vector(input_data, "Input");
        print_vector(output_data, "Output");
    }
    
    // Step 9: Free inference engine
    cortex_inference_engine_free(engine);
    
    std::cout << "\nSDK example completed successfully!" << std::endl;
    return 0;
}
