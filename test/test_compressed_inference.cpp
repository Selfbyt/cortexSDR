#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>
#include "../src/ai_compression/api/c_api.hpp"
#include "../src/ai_compression/SparseInferenceEngine.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>

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
        remove("compressed_cnn.json");
        remove("decompressed_model.onnx");
    }

    // Helper to generate random input tensor
    std::vector<float> generateInputTensor(size_t size) {
        std::vector<float> input(size);
        std::generate(input.begin(), input.end(), [this]() { return dist(gen); });
        return input;
    }

    // Helper to convert binary data to hex string
    std::string bytesToHex(const std::vector<std::byte>& data) {
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (const auto& byte : data) {
            ss << std::setw(2) << static_cast<int>(static_cast<unsigned char>(byte)) << " ";
        }
        return ss.str();
    }

    // Helper to save SDR file contents to JSON
    void saveSDRToJson(const std::string& sdr_path, const std::string& json_path) {
        std::ifstream infile(sdr_path, std::ios::binary);
        if (!infile) {
            throw std::runtime_error("Failed to open SDR file: " + sdr_path);
        }

        json j;
        json segments = json::array();
        size_t segment_count = 0;

        while (infile) {
            json segment;
            
            // Read segment header
            uint16_t name_len = 0;
            infile.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            if (!infile || name_len == 0 || name_len > 1024) break;

            // Read segment name
            std::string name(name_len, '\0');
            infile.read(&name[0], name_len);
            if (!infile) break;
            segment["name"] = name;

            // Read data size
            uint32_t data_size = 0;
            infile.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
            if (!infile || data_size == 0 || data_size > (1 << 26)) break;
            segment["data_size"] = data_size;

            // Read segment data
            std::vector<std::byte> data(data_size);
            infile.read(reinterpret_cast<char*>(data.data()), data_size);
            if (!infile) break;

            // Store data as hex string
            segment["data_hex"] = bytesToHex(data);

            // Try to interpret as text if it looks like text
            bool is_text = true;
            for (const auto& byte : data) {
                unsigned char c = static_cast<unsigned char>(byte);
                if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
                    is_text = false;
                    break;
                }
            }
            if (is_text) {
                segment["data_text"] = std::string(reinterpret_cast<char*>(data.data()), data.size());
            }

            // Try to interpret as float array if size is multiple of 4
            if (data_size % sizeof(float) == 0) {
                const float* floats = reinterpret_cast<const float*>(data.data());
                size_t num_floats = data_size / sizeof(float);
                json float_array = json::array();
                for (size_t i = 0; i < num_floats; ++i) {
                    float_array.push_back(floats[i]);
                }
                segment["data_floats"] = float_array;
            }

            segments.push_back(segment);
            segment_count++;
        }

        j["total_segments"] = segment_count;
        j["segments"] = segments;

        // Write to file with pretty printing
        std::ofstream file(json_path);
        file << j.dump(4);
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_F(CompressedInferenceTest, CompressAndInfer) {
    // Load and compress the model
    std::string model_path = "/home/mbishu/Desktop/cortexSDR/cnn_classifier.onnx";
    std::string compressed_path = "compressed_cnn.sdr";
    float sparsity = 0.02f;

    // Compress the model using the C API
    CortexCompressionOptions options;
    CortexError err = cortex_compression_options_init(&options);
    ASSERT_EQ(err.code, 0) << (err.message ? err.message : "Failed to init options");
    options.sparsity = sparsity;

    CortexCompressorHandle compressor = nullptr;
    err = cortex_compressor_create(model_path.c_str(), "onnx", &options, &compressor);
    ASSERT_EQ(err.code, 0) << (err.message ? err.message : "Failed to create compressor");

    err = cortex_compressor_compress(compressor, compressed_path.c_str());
    ASSERT_EQ(err.code, 0) << (err.message ? err.message : "Compression failed");

    cortex_compressor_free(compressor);

    std::cout << "About to load compressed model from: " << compressed_path << std::endl;
    
    // Create inference engine with compressed model
    try {
        SDRModelLoader loader(compressed_path);
        std::cout << "Model loaded successfully, checking layers..." << std::endl;
        // On-demand: print available segment names
        const auto& segments = loader.getSegmentIndex();
        std::cout << "Available segments in archive:" << std::endl;
        for (const auto& seg : segments) {
            std::cout << "  " << seg.name << std::endl;
        }
        // On-demand: load a single layer by name (first weight segment)
        auto it = std::find_if(segments.begin(), segments.end(), [](const SegmentInfo& seg) {
            return seg.type == SegmentType::WEIGHTS_FP32 || seg.type == SegmentType::WEIGHTS_FP16 || seg.type == SegmentType::WEIGHTS_INT8;
        });
        if (it != segments.end()) {
            std::cout << "\nLoading layer on-demand: " << it->name << std::endl;
            LayerInfo layer = loader.loadLayerByName(it->name);
            std::cout << "Loaded layer: " << layer.name << ", raw_data size: " << layer.raw_data.size() << std::endl;
        } else {
            std::cout << "No weight segment found for on-demand loading test." << std::endl;
        }
        // Legacy: load all layers (not memory efficient)
        ASSERT_FALSE(loader.getLayers().empty()) << "No layers loaded from compressed model";
        std::cout << "Number of layers loaded: " << loader.getLayers().size() << std::endl;
        
        // Print layer information for debugging
        std::cout << "\nLoaded layers:" << std::endl;
        for (const auto& layer : loader.getLayers()) {
            std::cout << "Layer: " << layer.name << std::endl;
            std::cout << "  Type: " << (layer.layer_type == "CONV2D" ? "Convolutional" : layer.layer_type) << std::endl;
            std::cout << "  Input shape: [";
            for (size_t i = 0; i < layer.input_shape.size(); ++i) {
                std::cout << layer.input_shape[i] << (i < layer.input_shape.size() - 1 ? ", " : "");
            }
            std::cout << "]" << std::endl;
            std::cout << "  Output shape: [";
            for (size_t i = 0; i < layer.output_shape.size(); ++i) {
                std::cout << layer.output_shape[i] << (i < layer.output_shape.size() - 1 ? ", " : "");
            }
            std::cout << "]" << std::endl;
            std::cout << "  Activation: " << layer.properties.activation_type << std::endl;
            std::cout << "  Batch norm: " << (layer.properties.use_batch_norm ? "Yes" : "No") << std::endl;
            std::cout << "  Dropout rate: " << layer.properties.dropout_rate << std::endl;
        }
        
        SDRInferenceEngine engine(loader);

        // Model-agnostic: check output size matches last layer's output_shape
        const auto& layers = loader.getLayers();
        ASSERT_FALSE(layers.empty()) << "No layers loaded from compressed model";
        const auto& first_layer = layers.front();
        size_t input_size = 1;
        for (size_t d : first_layer.input_shape) input_size *= d;
        std::vector<float> input_tensor_single(input_size, 0.0f);
        for (size_t i = 0; i < input_tensor_single.size(); ++i) {
            input_tensor_single[i] = static_cast<float>(i) / input_tensor_single.size();
        }
        std::vector<float> output_single = engine.run(input_tensor_single);
        const auto& last_layer = layers.back();
        size_t expected_output_size = 1;
        for (size_t d : last_layer.output_shape) expected_output_size *= d;
        std::cout << "Expected output size: " << expected_output_size << ", actual: " << output_single.size() << std::endl;
        ASSERT_EQ(output_single.size(), expected_output_size) << "Single batch output tensor has wrong size";

        engine.setBatchSize(4);
        std::vector<float> input_tensor_batch(4 * 3 * 32 * 32, 0.0f);
        for (size_t i = 0; i < input_tensor_batch.size(); ++i) {
            input_tensor_batch[i] = static_cast<float>(i) / input_tensor_batch.size();
        }
        std::vector<float> output_batch = engine.run(input_tensor_batch);
        // For batch, expected output size is batch_size * (product of last_layer.output_shape dims except batch)
        size_t per_sample_output = 1;
        if (!last_layer.output_shape.empty()) {
            for (size_t i = 1; i < last_layer.output_shape.size(); ++i) per_sample_output *= last_layer.output_shape[i];
        }
        size_t expected_batch_output_size = 4 * per_sample_output;
        std::cout << "Expected batch output size: " << expected_batch_output_size << ", actual: " << output_batch.size() << std::endl;
        ASSERT_EQ(output_batch.size(), expected_batch_output_size) << "Batch output tensor has wrong size";

        // Test training mode
        engine.setInferenceMode(true);
        engine.enableDropout(true);
        std::vector<float> output_training = engine.run(input_tensor_single);
        ASSERT_EQ(output_training.size(), 10) << "Training mode output tensor has wrong size";

        // Test inference mode
        engine.setInferenceMode(false);
        engine.enableDropout(false);
        std::vector<float> output_inference = engine.run(input_tensor_single);
        ASSERT_EQ(output_inference.size(), 10) << "Inference mode output tensor has wrong size";

        // Verify output values are reasonable
        for (float val : output_inference) {
            ASSERT_GE(val, 0.0f) << "Output value should be non-negative";
            ASSERT_LE(val, 1.0f) << "Output value should be <= 1";
        }

        // Print output probabilities
        std::cout << "\nOutput probabilities (inference mode):" << std::endl;
        for (size_t i = 0; i < output_inference.size(); ++i) {
            std::cout << "Class " << i << ": " << output_inference[i] << std::endl;
        }

        // Save model data to JSON for inspection
        saveSDRToJson(compressed_path, "compressed_cnn.json");
    } catch (const std::exception& e) {
        std::cerr << "Error loading compressed model: " << e.what() << std::endl;
        ASSERT_FALSE(true) << "Error loading compressed model";
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 