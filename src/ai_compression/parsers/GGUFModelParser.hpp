/**
 * @file GGUFModelParser.hpp
 * @brief Parser for GGUF-format models (e.g., llama.cpp artifacts).
 */
#ifndef GGUF_MODEL_PARSER_HPP
#define GGUF_MODEL_PARSER_HPP

#include "../core/AIModelParser.hpp"
#include "../core/ModelSegment.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>
#include <cstdint>

namespace CortexAICompression {

/**
 * @brief Parse GGUF models into compression-ready segments.
 */
class GGUFModelParser : public IAIModelParser {
public:
    GGUFModelParser();
    ~GGUFModelParser() override = default;

    /** Parse the model into segments without special chunking. */
    std::vector<ModelSegment> parse(const std::string& modelPath) const override;

    /** Parse with model-aware chunking for better compression. */
    std::vector<ModelSegment> parseWithChunking(const std::string& modelPath) const override;

private:
    // GGUF format constants
    static constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in ASCII
    static constexpr uint32_t GGUF_VERSION_MIN = 2;
    static constexpr uint32_t GGUF_VERSION_MAX = 3;

    struct GGUFHeaderInfo {
        uint32_t version = 0;
        uint64_t tensor_count = 0;
        uint64_t metadata_count = 0;
        uint32_t alignment = 32;
        uint64_t tensor_data_offset = 0;
        std::string architecture;
        std::string model_name;
        std::string tokenizer_model;
        std::vector<std::string> tokenizer_tokens;
        std::map<std::string, std::string> metadata;
    };

    struct GGUFTensorInfo {
        std::string name;
        std::vector<size_t> dimensions;
        uint32_t ggml_type = 0;
        std::string data_type;
        uint64_t offset = 0;
        uint64_t size = 0;
        size_t shard_index = 0;
    };

    // Helper methods
    GGUFHeaderInfo readHeader(std::ifstream& file) const;
    std::vector<GGUFTensorInfo> readTensorInfo(std::ifstream& file, GGUFHeaderInfo& header, size_t shard_index = 0) const;
    ModelSegment readTensor(std::ifstream& file, const GGUFHeaderInfo& header, const GGUFTensorInfo& info) const;
    ModelSegment createMetadataSegment(const GGUFHeaderInfo& header) const;
    ModelSegment createConfigSegment(const GGUFHeaderInfo& header) const;
    ModelSegment createTokenizerVocabSegment(const GGUFHeaderInfo& header) const;
    ModelSegment createTokenizerModelSegment(const GGUFHeaderInfo& header) const;
    SegmentType determineSegmentType(const std::string& tensorName, const std::string& dataType) const;
    TensorMetadata extractTensorMetadata(const GGUFTensorInfo& info) const;
    std::string extractLayerName(const std::string& tensorName) const;
    size_t extractLayerIndex(const std::string& tensorName) const;
};

} // namespace CortexAICompression

#endif // GGUF_MODEL_PARSER_HPP
