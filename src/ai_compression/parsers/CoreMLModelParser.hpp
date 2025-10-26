/**
 * @file CoreMLModelParser.hpp
 * @brief Parser for CoreML model format (.mlmodel files)
 */
#ifndef COREML_MODEL_PARSER_HPP
#define COREML_MODEL_PARSER_HPP

#include "../core/AIModelParser.hpp"
#include "../core/ModelSegment.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>

#ifdef ENABLE_COREML
// CoreML headers would go here if available
// Note: CoreML is primarily an Apple framework, so cross-platform support is limited
#endif

namespace CortexAICompression {

/**
 * @brief Parse CoreML model format into compression-ready segments.
 * 
 * Note: CoreML is primarily an Apple framework. This parser provides
 * basic support for extracting model information from .mlmodel files
 * which are essentially protobuf archives.
 */
class CoreMLModelParser : public IAIModelParser {
public:
    CoreMLModelParser();
    ~CoreMLModelParser() override;

    std::vector<ModelSegment> parse(const std::string& modelPath) const override;
    std::vector<ModelSegment> parseWithChunking(const std::string& modelPath) const override;

private:
    // Helper struct for CoreML layer info
    struct CoreMLLayerInfo {
        std::string name;
        std::string layer_type;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<std::byte> weights_data;
        std::vector<std::byte> bias_data;
        std::string data_type;
    };

    // Helper methods
    std::vector<CoreMLLayerInfo> extractLayerInfo(const std::string& modelPath) const;
    ModelSegment createSegmentFromLayer(const CoreMLLayerInfo& layerInfo, const std::string& segmentName, const std::vector<std::byte>& data) const;
    SegmentType coremlDataTypeToSegmentType(const std::string& dataType) const;
    TensorMetadata extractTensorMetadata(const CoreMLLayerInfo& layerInfo, const std::vector<int64_t>& shape) const;
    std::string extractLayerName(const std::string& layerName) const;
    size_t extractLayerIndex(const std::string& layerName) const;
    std::vector<std::byte> readModelData(const std::string& modelPath) const;
};

} // namespace CortexAICompression

#endif // COREML_MODEL_PARSER_HPP
