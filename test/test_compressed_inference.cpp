#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
#include "../src/ai_compression/api/c_api.hpp"
#include "../src/ai_compression/SparseInferenceEngine.hpp"
#include <iostream>
#include <random>
#include <algorithm>

using namespace CortexAICompression;
using json = nlohmann::json;

class CompressedInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator for test data
        std::random_device rd;
        gen = std::mt19937(rd());
        dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }

    void TearDown() override {
        // Clean up test files
        remove("compressed_cnn.sdr");
        // remove("compressed_cnn.json"); // Commented out to preserve the JSON output
    }

    // Helper to generate random input tensor
    std::vector<float> generateInputTensor(size_t size) {
        std::vector<float> input(size);
        std::generate(input.begin(), input.end(), [this]() { return dist(gen); });
        return input;
    }

    // Helper to save model data to JSON
    void saveModelToJson(const std::string& filename, 
                        const std::vector<size_t>& input_indices,
                        const std::vector<float>& input_values,
                        const std::vector<size_t>& output_indices) {
        json j;
        
        // Store input data
        json input_data;
        for (size_t i = 0; i < input_indices.size(); ++i) {
            input_data.push_back({
                {"index", input_indices[i]},
                {"value", input_values[input_indices[i]]}
            });
        }
        j["input"] = input_data;

        // Store output indices
        j["output_indices"] = output_indices;

        // Write to file with pretty printing
        std::ofstream file(filename);
        file << j.dump(4);
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_F(CompressedInferenceTest, CompressAndInfer) {
    // 1. Compress the model using the C API
    const char* input_model = "/home/mbishu/Desktop/cortexSDR/cnn_classifier.onnx";
    const char* output_compressed = "compressed_cnn.sdr";
    const char* output_json = "compressed_cnn.json";

    std::cout << "Starting compression of model: " << input_model << std::endl;

    CortexCompressionOptions options;
    CortexError err = cortex_compression_options_init(&options);
    ASSERT_EQ(err.code, 0) << (err.message ? err.message : "Failed to init options");
    options.sparsity = 0.02f;

    CortexCompressorHandle handle = nullptr;
    err = cortex_compressor_create(input_model, "onnx", &options, &handle);
    ASSERT_EQ(err.code, 0) << (err.message ? err.message : "Failed to create compressor");

    err = cortex_compressor_compress(handle, output_compressed);
    ASSERT_EQ(err.code, 0) << (err.message ? err.message : "Compression failed");

    cortex_compressor_free(handle);

    std::cout << "Model compression completed successfully" << std::endl;

    // 2. Load the compressed model for inference
    SDRModelLoader loader(output_compressed);
    SDRInferenceEngine engine(loader);

    // 3. Generate input tensor (assuming 3x32x32 input for CNN classifier)
    const size_t input_size = 3 * 32 * 32;  // Channels * Height * Width
    std::vector<float> input_tensor = generateInputTensor(input_size);
    
    std::cout << "Generated input tensor of size: " << input_size << std::endl;
    std::cout << "First few input values: ";
    for (size_t i = 0; i < std::min(size_t(5), input_size); ++i) {
        std::cout << input_tensor[i] << " ";
    }
    std::cout << std::endl;

    // 4. Run inference
    std::vector<size_t> input_indices;
    for (size_t i = 0; i < input_size; i++) {
        if (std::abs(input_tensor[i]) > 0.1f) {  // Only use significant values
            input_indices.push_back(i);
        }
    }

    std::cout << "Running inference with " << input_indices.size() << " active input indices" << std::endl;
    auto output_indices = engine.run(input_indices);

    // 5. Save the model data to JSON
    saveModelToJson(output_json, input_indices, input_tensor, output_indices);
    std::cout << "Saved model data to " << output_json << std::endl;

    // 6. Verify output
    EXPECT_FALSE(output_indices.empty()) << "Inference produced no output";
    std::cout << "Inference produced " << output_indices.size() << " output indices" << std::endl;
    
    // Print top 5 output indices and their values
    std::cout << "Top 5 output indices: ";
    for (size_t i = 0; i < std::min(size_t(5), output_indices.size()); ++i) {
        std::cout << output_indices[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 