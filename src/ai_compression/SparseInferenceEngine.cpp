#include "SparseInferenceEngine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>

namespace CortexAICompression {

// Helper: decode varint-encoded indices from SDR data
std::vector<size_t> SDRModelLoader::decode_varint_indices(const std::vector<std::byte>& data) {
    std::vector<size_t> indices;
    size_t pos = 0;
    while (pos < data.size()) {
        uint32_t index = 0;
        uint32_t shift = 0;
        while (pos < data.size()) {
            uint8_t byte = static_cast<uint8_t>(data[pos++]);
            index |= (byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) break;
            shift += 7;
        }
        indices.push_back(index);
    }
    return indices;
}

SDRModelLoader::SDRModelLoader(const std::string& archive_path) {
    loadFromArchive(archive_path);
}

// Example SDR archive format:
// [uint16_t name_len][char name[]][uint32_t data_size][byte data[]] ...
void SDRModelLoader::loadFromArchive(const std::string& archive_path) {
    std::ifstream infile(archive_path, std::ios::binary);
    if (!infile) {
        std::cerr << "[SDRModelLoader] Failed to open archive: " << archive_path << std::endl;
        return;
    }
    while (infile) {
        uint16_t name_len = 0;
        infile.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        if (!infile) break;
        if (name_len == 0 || name_len > 1024) break;
        std::string name(name_len, '\0');
        infile.read(&name[0], name_len);
        if (!infile) break;
        uint32_t data_size = 0;
        infile.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        if (!infile) break;
        if (data_size == 0 || data_size > (1 << 26)) break;
        std::vector<std::byte> data(data_size);
        infile.read(reinterpret_cast<char*>(data.data()), data_size);
        if (!infile) break;
        // Only process weight segments (skip metadata/graph for now)
        if (name == "model_metadata" || name == "model_structure") continue;
        LayerInfo layer;
        layer.name = name;
        layer.raw_data = data;
        layer.active_indices = decode_varint_indices(data);
        layers.push_back(layer);
    }
    std::cout << "[SDRModelLoader] Loaded " << layers.size() << " layers from archive." << std::endl;
}

const std::vector<LayerInfo>& SDRModelLoader::getLayers() const {
    return layers;
}

SDRInferenceEngine::SDRInferenceEngine(const SDRModelLoader& model)
    : layers(model.getLayers()) {
    std::cout << "[SDRInferenceEngine] Initialized with " << layers.size() << " layers." << std::endl;
}

std::vector<size_t> SDRInferenceEngine::run(const std::vector<size_t>& input_active_indices) {
    std::vector<size_t> current = input_active_indices;
    for (const auto& layer : layers) {
        std::vector<size_t> next;
        // Propagate: for each input index, if it matches an active index in this layer, activate it
        std::set_intersection(
            current.begin(), current.end(),
            layer.active_indices.begin(), layer.active_indices.end(),
            std::back_inserter(next)
        );
        std::cout << "[SDRInferenceEngine] Layer '" << layer.name << "' input: " << current.size() << ", output: " << next.size() << std::endl;
        current = next;
    }
    return current;
}

} // namespace CortexAICompression 