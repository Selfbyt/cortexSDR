/**
 * @file PyTorchModelParser.hpp
 * @brief Parser for PyTorch model format (.pt/.pth files) with ZIP archive support
 */
#ifndef PYTORCH_MODEL_PARSER_HPP
#define PYTORCH_MODEL_PARSER_HPP

#include "../core/AIModelParser.hpp"
#include "../core/ModelSegment.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>

#ifdef ENABLE_PYTORCH
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/serialization/pickle.h>
#endif

namespace CortexAICompression {

/**
 * @brief Parse PyTorch model format into compression-ready segments.
 * Supports: TorchScript, state dicts, and LLaMA-style ZIP archives (consolidated.pth)
 */
class PyTorchModelParser : public IAIModelParser {
public:
    PyTorchModelParser();
    ~PyTorchModelParser() override;
    
    std::vector<ModelSegment> parse(const std::string& modelPath) const override;
    std::vector<ModelSegment> parseWithChunking(const std::string& modelPath) const override;

private:
#ifdef ENABLE_PYTORCH
    // Helper struct for PyTorch tensor info
    struct PyTorchTensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        torch::ScalarType scalar_type;
        size_t size_bytes;
        std::vector<std::byte> data;
    };
    
    // Core extraction methods
    std::vector<PyTorchTensorInfo> extractTensorInfo(const std::string& modelPath) const;
    ModelSegment createSegmentFromTensor(const PyTorchTensorInfo& tensorInfo) const;
    
    // ZIP archive support (for LLaMA consolidated.pth format)
    bool isZipFile(const std::string& modelPath) const;
    std::vector<PyTorchTensorInfo> extractFromZipArchive(const std::string& modelPath) const;
    
    // Type conversion and metadata extraction
    SegmentType pytorchScalarTypeToSegmentType(torch::ScalarType scalar_type) const;
    TensorMetadata extractTensorMetadata(const PyTorchTensorInfo& tensorInfo) const;
    
    // Layer name/index extraction
    std::string extractLayerName(const std::string& tensorName) const;
    size_t extractLayerIndex(const std::string& tensorName) const;
    
    // Tensor data conversion
    std::vector<std::byte> tensorToBytes(const torch::Tensor& tensor) const;
#endif
};

} // namespace CortexAICompression

#endif // PYTORCH_MODEL_PARSER_HPP