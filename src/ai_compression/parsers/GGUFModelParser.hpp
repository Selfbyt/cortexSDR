#ifndef GGUF_MODEL_PARSER_HPP
#define GGUF_MODEL_PARSER_HPP

#include "../core/AIModelParser.hpp"
#include "../core/ModelSegment.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>

namespace CortexAICompression {

// Parser for GGUF format models (used by llama.cpp and others)
class GGUFModelParser : public IAIModelParser {
public:
    GGUFModelParser();
    ~GGUFModelParser() override = default;

    // Parse the model into segments without special chunking
    std::vector<ModelSegment> parse(const std::string& modelPath) const override;

    // Parse with model-aware chunking for better compression
    std::vector<ModelSegment> parseWithChunking(const std::string& modelPath) const override;

private:
    // GGUF format constants
    static constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in ASCII
    static constexpr uint32_t GGUF_VERSION = 2;         // Current GGUF version

    // Helper struct for GGUF tensor info
    struct GGUFTensorInfo {
        std::string name;
        std::vector<size_t> dimensions;
        std::string data_type;
        size_t offset;
        size_t size;
    };

    // Helper methods
    bool readHeader(std::ifstream& file) const;
    std::vector<GGUFTensorInfo> readTensorInfo(std::ifstream& file) const;
    ModelSegment readTensor(std::ifstream& file, const GGUFTensorInfo& info) const;
    SegmentType determineSegmentType(const std::string& tensorName, const std::string& dataType) const;
    TensorMetadata extractTensorMetadata(const GGUFTensorInfo& info) const;
    std::string extractLayerName(const std::string& tensorName) const;
    size_t extractLayerIndex(const std::string& tensorName) const;
};

} // namespace CortexAICompression

#endif // GGUF_MODEL_PARSER_HPP 