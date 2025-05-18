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

#ifdef ENABLE_ONNX
// ONNX Runtime headers can be in different locations depending on installation
#include <onnxruntime_cxx_api.h>
#endif

#ifdef ENABLE_ONNX_PROTOBUF
// Include the generated protobuf header
#include <../onnx_proto/onnx.pb.h>
#endif

namespace CortexAICompression {

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
        std::cout << "Segment: " << segment.name << std::endl;
        std::cout << "  Type: " << static_cast<int>(segment.type) << std::endl;
        std::cout << "  Original Size: " << segment.original_size << " bytes" << std::endl;
        
        if (segment.tensor_metadata) {
            const auto& metadata = segment.tensor_metadata.value();
            std::cout << "  Tensor Metadata:" << std::endl;
            std::cout << "    Dimensions: [";
            for (size_t i = 0; i < metadata.dimensions.size(); ++i) {
                std::cout << metadata.dimensions[i];
                if (i < metadata.dimensions.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "    Sparsity Ratio: " << metadata.sparsity_ratio << std::endl;
            std::cout << "    Is Sorted: " << (metadata.is_sorted ? "true" : "false") << std::endl;
            if (metadata.scale) {
                std::cout << "    Scale: " << metadata.scale.value() << std::endl;
            }
            if (metadata.zero_point) {
                std::cout << "    Zero Point: " << metadata.zero_point.value() << std::endl;
            }
        } else {
            std::cout << "  No tensor metadata available" << std::endl;
        }
        
        if (!segment.layer_name.empty()) {
            std::cout << "  Layer Name: " << segment.layer_name << std::endl;
            std::cout << "  Layer Index: " << segment.layer_index << std::endl;
        }
        
        if (!segment.data_format.empty()) {
            std::cout << "  Data Format: " << segment.data_format << std::endl;
        }
        
        std::cout << std::endl;
    }

#ifdef ENABLE_ONNX
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;

    ModelSegment createSegmentFromTensor(const std::string& name, const Ort::Value& tensor) const;
    SegmentType determineSegmentType(const std::string& tensorName, const Ort::Value& tensor) const;
    // Overload matching the definition in the .cpp file (likely needs refactoring)
    SegmentType determineSegmentType(const std::string& tensorName) const;
    TensorMetadata extractTensorMetadata(const Ort::Value& tensor) const;
    std::string extractLayerName(const std::string& tensorName) const;
    size_t extractLayerIndex(const std::string& tensorName) const;
    std::vector<std::byte> tensorToBytes(const Ort::Value& tensor) const;
    
#ifdef ENABLE_ONNX_PROTOBUF
    // Helper matching the definition in the .cpp file (likely needs refactoring)
    TensorMetadata extractTensorMetadataProto(const onnx::TensorProto& tensor) const;
    SegmentType onnxTensorTypeToSegmentType(int32_t onnx_type) const; // Added declaration
    // Helper matching the definition in the .cpp file (likely needs refactoring)
    std::vector<std::byte> tensorProtoToBytes(const onnx::TensorProto& tensor) const;
#endif
#endif

    // New optimized parsing methods
    void processMetadata(const onnx::ModelProto& model_proto, std::vector<ModelSegment>& segments) const;
    void processGraphStructure(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const;
    void processInitializers(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const;
    void processInitializersParallel(const onnx::GraphProto& graph_proto, std::vector<ModelSegment>& segments) const;
};

} // namespace CortexAICompression

#endif // ONNX_MODEL_PARSER_HPP
