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

// Forward declaration for the utility function
void print_possible_layer_chains(const std::vector<CortexAICompression::LayerInfo>& layers);

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
        // remove("compressed_model.sdr");
        // remove("compressed_model.json");
        // remove("decompressed_model.onnx");
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

TEST_F(CompressedInferenceTest, SelectiveInference) {
    // Restore: Compress the model before inference
    std::string model_path = "/home/mbishu/Desktop/cortexSDR/gpt2-10.onnx";
    std::string compressed_path = "compressed_model.sdr";
    // Raise sparsity to retain 20% of weights so embeddings aren't overly zeroed
    float sparsity = 0.20f;

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
    // End restore

    SDRModelLoader loader(compressed_path);
    SDRInferenceEngine engine(loader);

    const auto& segments = loader.getSegmentIndex();
    if (segments.empty()) {
        GTEST_SKIP() << "No segments found in compressed model, skipping test.";
    }

    // Heuristically find a sequence of layers to run.
    std::vector<std::string> selected_layers;
    for (const auto& seg1 : segments) {
        if (seg1.original_type != SegmentType::WEIGHTS_FP32) continue;
        try {
            LayerInfo layer1 = loader.loadLayerByName(seg1.name);
            for (const auto& seg2 : segments) {
                if (seg1.name == seg2.name || seg2.original_type != SegmentType::WEIGHTS_FP32) continue;
                try {
                    LayerInfo layer2 = loader.loadLayerByName(seg2.name);
                    if (!layer1.output_shape.empty() && layer1.output_shape == layer2.input_shape) {
                        selected_layers.push_back(layer1.name);
                        selected_layers.push_back(layer2.name);
                        goto found_pair;
                    }
                } catch (const std::exception&) {
                    // Skip if a layer fails to load, it might be metadata
                }
            }
        } catch (const std::exception&) {
            // Skip if a layer fails to load
        }
    }

found_pair:
    if (selected_layers.empty()) {
        std::cout << "[SelectiveInference] No compatible layer pair found. Trying single layer." << std::endl;
        // Fallback: just find the first valid layer
        for (const auto& seg : segments) {
             if (seg.original_type == SegmentType::WEIGHTS_FP32) {
                 selected_layers.push_back(seg.name);
                 break;
             }
        }
    }
    
    if (selected_layers.empty()) {
        GTEST_SKIP() << "No runnable layers found, skipping test.";
    }


    std::cout << "\n[SelectiveInference] Running layers:";
    for (const auto& name : selected_layers) std::cout << " " << name;
    std::cout << std::endl;

    LayerInfo first_layer = loader.loadLayerByName(selected_layers.front());
    size_t input_size = 1;
    for (size_t d : first_layer.input_shape) input_size *= d;
    if (input_size == 0) {
        GTEST_SKIP() << "First layer has zero-sized input, cannot proceed.";
    }
    auto input_tensor = generateInputTensor(input_size);

    std::vector<float> current = input_tensor;
    for (const auto& name : selected_layers) {
        LayerInfo layer = loader.loadLayerByName(name);
        std::cout << "[SelectiveInference] Running layer: " << name << " (type: " << layer.layer_type << ")\n";
        
        current = engine.runLayer(layer, current);
        if (current.empty()) {
             FAIL() << "runLayer for " << name << " returned an empty tensor.";
        }
    }
    std::cout << "[SelectiveInference] Output size after running selected layers: " << current.size() << std::endl;
    ASSERT_FALSE(current.empty());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 