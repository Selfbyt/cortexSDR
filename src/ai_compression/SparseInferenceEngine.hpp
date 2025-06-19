#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstddef>

namespace CortexAICompression {

// LayerInfo holds SDR indices and metadata for a layer
struct LayerInfo {
    std::string name;
    std::vector<size_t> active_indices; // Decoded indices of active bits
    std::vector<std::byte> raw_data;    // Raw SDR data (for debugging)
    std::vector<size_t> input_shape;    // Optional: input shape
    std::vector<size_t> output_shape;   // Optional: output shape
    // Add more metadata as needed
};

class SDRModelLoader {
public:
    explicit SDRModelLoader(const std::string& archive_path);
    const std::vector<LayerInfo>& getLayers() const;
    // Helper to decode varint-encoded indices from SDR data
    static std::vector<size_t> decode_varint_indices(const std::vector<std::byte>& data);
private:
    std::vector<LayerInfo> layers;
    void loadFromArchive(const std::string& archive_path);
};

class SDRInferenceEngine {
public:
    explicit SDRInferenceEngine(const SDRModelLoader& model);
    // Given input active indices, run sparse inference and return output active indices
    std::vector<size_t> run(const std::vector<size_t>& input_active_indices);
private:
    std::vector<LayerInfo> layers;
};

} // namespace CortexAICompression 