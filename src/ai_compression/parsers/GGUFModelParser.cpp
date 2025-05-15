#include "GGUFModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>

namespace CortexAICompression {

GGUFModelParser::GGUFModelParser() {}

bool GGUFModelParser::readHeader(std::ifstream& file) const {
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != GGUF_MAGIC) {
        throw ParsingError("Invalid GGUF magic number");
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != GGUF_VERSION) {
        throw ParsingError("Unsupported GGUF version: " + std::to_string(version));
    }

    return true;
}

std::vector<GGUFModelParser::GGUFTensorInfo> GGUFModelParser::readTensorInfo(std::ifstream& file) const {
    std::vector<GGUFTensorInfo> tensors;
    uint32_t numTensors;
    file.read(reinterpret_cast<char*>(&numTensors), sizeof(numTensors));

    for (uint32_t i = 0; i < numTensors; ++i) {
        GGUFTensorInfo info;
        
        // Read name length and name
        uint32_t nameLen;
        file.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        info.name.resize(nameLen);
        file.read(&info.name[0], nameLen);

        // Read dimensions
        uint32_t numDims;
        file.read(reinterpret_cast<char*>(&numDims), sizeof(numDims));
        info.dimensions.resize(numDims);
        file.read(reinterpret_cast<char*>(info.dimensions.data()), numDims * sizeof(size_t));

        // Read data type
        uint32_t typeLen;
        file.read(reinterpret_cast<char*>(&typeLen), sizeof(typeLen));
        info.data_type.resize(typeLen);
        file.read(&info.data_type[0], typeLen);

        // Read offset and size
        file.read(reinterpret_cast<char*>(&info.offset), sizeof(info.offset));
        file.read(reinterpret_cast<char*>(&info.size), sizeof(info.size));

        tensors.push_back(std::move(info));
    }

    return tensors;
}

SegmentType GGUFModelParser::determineSegmentType(const std::string& tensorName, const std::string& dataType) const {
    // Determine segment type based on tensor name and data type
    if (tensorName.find("attention") != std::string::npos) {
        return SegmentType::ATTENTION_WEIGHTS;
    } else if (tensorName.find("feed_forward") != std::string::npos) {
        return SegmentType::FEED_FORWARD_WEIGHTS;
    } else if (tensorName.find("embedding") != std::string::npos) {
        return SegmentType::EMBEDDING_WEIGHTS;
    } else if (tensorName.find("layer_norm") != std::string::npos) {
        return SegmentType::LAYER_NORM_WEIGHTS;
    }

    // Determine type based on data type
    if (dataType == "float32") {
        return SegmentType::WEIGHTS_FP32;
    } else if (dataType == "float16") {
        return SegmentType::WEIGHTS_FP16;
    } else if (dataType == "int8") {
        return SegmentType::WEIGHTS_INT8;
    } else if (dataType == "int4") {
        return SegmentType::WEIGHTS_INT4;
    }

    return SegmentType::UNKNOWN;
}

TensorMetadata GGUFModelParser::extractTensorMetadata(const GGUFTensorInfo& info) const {
    TensorMetadata metadata;
    metadata.dimensions = info.dimensions;
    
    // Calculate sparsity (placeholder - would need to analyze actual data)
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = false;

    // Extract quantization info if available
    if (info.data_type.find("q") == 0) {  // Quantized type
        metadata.scale = 1.0f;  // Would need to read from actual quantization data
        metadata.zero_point = 0.0f;
    }

    return metadata;
}

std::string GGUFModelParser::extractLayerName(const std::string& tensorName) const {
    std::regex layer_pattern(R"(layers\.(\d+)\.([\w\.]+))");
    std::smatch matches;
    if (std::regex_search(tensorName, matches, layer_pattern)) {
        return matches[2].str();
    }
    return "";
}

size_t GGUFModelParser::extractLayerIndex(const std::string& tensorName) const {
    std::regex layer_pattern(R"(layers\.(\d+))");
    std::smatch matches;
    if (std::regex_search(tensorName, matches, layer_pattern)) {
        return std::stoul(matches[1].str());
    }
    return 0;
}

ModelSegment GGUFModelParser::readTensor(std::ifstream& file, const GGUFTensorInfo& info) const {
    ModelSegment segment;
    segment.name = info.name;
    segment.type = determineSegmentType(info.name, info.data_type);
    segment.tensor_metadata = extractTensorMetadata(info);
    segment.layer_name = extractLayerName(info.name);
    segment.layer_index = extractLayerIndex(info.name);
    
    // Read tensor data
    file.seekg(info.offset);
    segment.data.resize(info.size);
    file.read(reinterpret_cast<char*>(segment.data.data()), info.size);
    segment.original_size = info.size;

    return segment;
}

std::vector<ModelSegment> GGUFModelParser::parse(const std::string& modelPath) const {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file) {
        throw ParsingError("Failed to open model file: " + modelPath);
    }

    if (!readHeader(file)) {
        throw ParsingError("Failed to read GGUF header");
    }

    auto tensorInfos = readTensorInfo(file);
    std::vector<ModelSegment> segments;
    segments.reserve(tensorInfos.size());

    for (const auto& info : tensorInfos) {
        segments.push_back(readTensor(file, info));
    }

    return segments;
}

std::vector<ModelSegment> GGUFModelParser::parseWithChunking(const std::string& modelPath) const {
    auto segments = parse(modelPath);
    
    // Group segments by layer for better compression
    std::map<size_t, std::vector<ModelSegment*>> layerGroups;
    for (auto& segment : segments) {
        layerGroups[segment.layer_index].push_back(&segment);
    }

    // Sort segments within each layer by type
    for (auto& [layer, group] : layerGroups) {
        std::sort(group.begin(), group.end(),
                 [](const ModelSegment* a, const ModelSegment* b) {
                     return static_cast<int>(a->type) < static_cast<int>(b->type);
                 });
    }

    // Reorder segments for optimal compression
    std::vector<ModelSegment> reorderedSegments;
    reorderedSegments.reserve(segments.size());

    // First, add non-layer segments (embeddings, global params, etc.)
    for (const auto& segment : segments) {
        if (segment.layer_index == 0 && segment.layer_name.empty()) {
            reorderedSegments.push_back(segment);
        }
    }

    // Then add layer segments in order
    for (const auto& [layer, group] : layerGroups) {
        if (layer > 0 || !group.front()->layer_name.empty()) {
            for (const auto* segment : group) {
                reorderedSegments.push_back(*segment);
            }
        }
    }

    return reorderedSegments;
}

} // namespace CortexAICompression 