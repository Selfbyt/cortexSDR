/**
 * @file TensorFlowModelParser.hpp
 * @brief Parser for TensorFlow SavedModel format (.pb files)
 */
#ifndef TENSORFLOW_MODEL_PARSER_HPP
#define TENSORFLOW_MODEL_PARSER_HPP

#include "../core/AIModelParser.hpp"
#include "../core/ModelSegment.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>

#ifdef ENABLE_TENSORFLOW
#include <tensorflow/c/c_api.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.pb.h>
#endif

namespace CortexAICompression {

/**
 * @brief Parse TensorFlow SavedModel format into compression-ready segments.
 */
class TensorFlowModelParser : public IAIModelParser {
public:
    TensorFlowModelParser();
    ~TensorFlowModelParser() override;

    std::vector<ModelSegment> parse(const std::string& modelPath) const override;
    std::vector<ModelSegment> parseWithChunking(const std::string& modelPath) const override;

private:
#ifdef ENABLE_TENSORFLOW
    // Helper struct for TensorFlow variable info
    struct TFVariableInfo {
        std::string name;
        std::vector<int64_t> shape;
        tensorflow::DataType data_type;
        size_t size_bytes;
    };

    // Helper methods
    std::vector<TFVariableInfo> extractVariableInfo(const std::string& modelPath) const;
    ModelSegment createSegmentFromVariable(const TFVariableInfo& varInfo, const std::vector<std::byte>& data) const;
    SegmentType tensorflowDataTypeToSegmentType(tensorflow::DataType tf_type) const;
    TensorMetadata extractTensorMetadata(const TFVariableInfo& varInfo) const;
    std::string extractLayerName(const std::string& variableName) const;
    size_t extractLayerIndex(const std::string& variableName) const;
    std::vector<std::byte> readVariableData(const std::string& modelPath, const TFVariableInfo& varInfo) const;
#endif
};

} // namespace CortexAICompression

#endif // TENSORFLOW_MODEL_PARSER_HPP
