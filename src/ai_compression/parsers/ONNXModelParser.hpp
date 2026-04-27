/**
 * @file ONNXModelParser.hpp
 * @brief Parser that converts ONNX models into archive segments.
 */
#ifndef ONNX_MODEL_PARSER_HPP
#define ONNX_MODEL_PARSER_HPP

#include "../core/AIModelParser.hpp"
#include "../core/ModelSegment.hpp"
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <thread>
#include <iostream>
#include <array>

#ifdef ENABLE_ONNX
// ONNX Runtime headers can be in different locations depending on installation
#include <onnxruntime_cxx_api.h>
#endif

#ifdef ENABLE_ONNX_PROTOBUF
// Include the generated protobuf header
#include <../onnx_proto/onnx.pb.h>
#endif

namespace CortexAICompression {

/**
 * @brief Parse ONNX models into compression-ready segments.
 */
class ONNXModelParser : public IAIModelParser {
public:
    ONNXModelParser();
    ~ONNXModelParser() override;

    std::vector<ModelSegment> parse(const std::string& modelPath) const override;
    // Override the parseWithChunking method from IAIModelParser
    std::vector<ModelSegment> parseWithChunking(const std::string& modelPath) const override;

private:
    // Helper method to print segment metadata
    void printSegmentMetadata(const ModelSegment& segment) const {
        // Debug output disabled
    }

#ifdef ENABLE_ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;

    ModelSegment createSegmentFromTensor(const std::string& name, const Ort::Value& tensor) const;
    SegmentType determineSegmentType(const std::string& tensorName, const Ort::Value& tensor) const;
    TensorMetadata extractTensorMetadata(const Ort::Value& tensor) const;
    std::vector<std::byte> tensorToBytes(const Ort::Value& tensor) const;
#endif

    // Used by both runtime-backed and protobuf-backed ONNX parsing flows.
    SegmentType determineSegmentType(const std::string& tensorName) const;
    std::string extractLayerName(const std::string& tensorName) const;
    size_t extractLayerIndex(const std::string& tensorName) const;

#ifdef ENABLE_ONNX_PROTOBUF
    // Helpers for protobuf-backed parsing.
    TensorMetadata extractTensorMetadataProto(const onnx::TensorProto& tensor) const;
    SegmentType onnxTensorTypeToSegmentType(int32_t onnx_type) const;
    std::vector<std::byte> tensorProtoToBytes(const onnx::TensorProto& tensor) const;

    // New optimized parsing methods
    void processMetadata(const onnx::ModelProto& model_proto, std::vector<ModelSegment>& segments) const;
    void processGraphStructure(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const;
    void processInitializers(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const;
    void processInitializersParallel(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const;
#endif
};

} // namespace CortexAICompression

#endif // ONNX_MODEL_PARSER_HPP
