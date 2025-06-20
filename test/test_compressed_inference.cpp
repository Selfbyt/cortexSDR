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
    float sparsity = 0.02f;

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

    // Print all possible layer chains based on shape matching
    CortexAICompression::print_possible_layer_chains(loader.getLayers());

    const auto& layer_map = loader.getLayerMap();
    std::cout << "\n[SelectiveInference] Layer shapes:" << std::endl;
    for (const auto& kv : layer_map) {
        const auto& layer = kv.second;
        std::cout << "  " << kv.first << " (type: " << layer.layer_type << ")\n";
        std::cout << "    Input shape: [";
        for (size_t i = 0; i < layer.input_shape.size(); ++i) {
            std::cout << layer.input_shape[i] << (i < layer.input_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]\n    Output shape: [";
        for (size_t i = 0; i < layer.output_shape.size(); ++i) {
            std::cout << layer.output_shape[i] << (i < layer.output_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]\n";
    }

    // Find a pair of layers with matching output/input shapes
    std::vector<std::string> selected_layers;
    for (auto it1 = layer_map.begin(); it1 != layer_map.end(); ++it1) {
        for (auto it2 = layer_map.begin(); it2 != layer_map.end(); ++it2) {
            if (it1 == it2) continue;
            const auto& out_shape = it1->second.output_shape;
            const auto& in_shape = it2->second.input_shape;
            if (!out_shape.empty() && out_shape == in_shape) {
                selected_layers.push_back(it1->first);
                selected_layers.push_back(it2->first);
                goto found_pair;
            }
        }
    }
found_pair:
    if (selected_layers.size() < 2) {
        std::cout << "[SelectiveInference] No compatible layer pair found. Skipping test." << std::endl;
        GTEST_SKIP();
    }
    std::cout << "\n[SelectiveInference] Running layers:";
    for (const auto& name : selected_layers) std::cout << " " << name;
    std::cout << std::endl;

    const auto& first_layer = layer_map.at(selected_layers.front());
    size_t input_size = 1;
    for (size_t d : first_layer.input_shape) input_size *= d;
    auto input_tensor = generateInputTensor(input_size);

    std::vector<float> current = input_tensor;
    for (const auto& name : selected_layers) {
        const auto& layer = layer_map.at(name);
        std::cout << "[SelectiveInference] Running layer: " << name << " (type: " << layer.layer_type << ")\n";
        std::cout << "  Input size: " << current.size() << ", expected: ";
        size_t expected_in = 1;
        for (size_t d : layer.input_shape) expected_in *= d;
        std::cout << expected_in << "\n";
        std::cout << "  Input shape: [";
        for (size_t i = 0; i < layer.input_shape.size(); ++i) {
            std::cout << layer.input_shape[i] << (i < layer.input_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]\n";
        current = engine.runLayer(name, current);
        std::cout << "  Output size: " << current.size() << ", expected: ";
        size_t expected_out = 1;
        for (size_t d : layer.output_shape) expected_out *= d;
        std::cout << expected_out << "\n";
        std::cout << "  Output shape: [";
        for (size_t i = 0; i < layer.output_shape.size(); ++i) {
            std::cout << layer.output_shape[i] << (i < layer.output_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]\n";
        std::cout << "  Output preview: ";
        for (size_t i = 0; i < std::min<size_t>(5, current.size()); ++i) {
            std::cout << current[i] << " ";
        }
        std::cout << (current.size() > 5 ? "..." : "") << std::endl;
    }
    std::cout << "[SelectiveInference] Output size after running selected layers: " << current.size() << std::endl;
    ASSERT_FALSE(current.empty());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 