#ifndef MODEL_CONVERTER_HPP
#define MODEL_CONVERTER_HPP

#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>

// Include ONNX Runtime headers if enabled
#ifdef ENABLE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

// Include TensorFlow headers if enabled
#ifdef ENABLE_TENSORFLOW
#include <tensorflow/c/c_api.h>
#endif

// Include PyTorch headers if enabled
#ifdef ENABLE_PYTORCH
#include <torch/script.h>
#endif

namespace CortexAICompression {

class ModelConversionError : public std::runtime_error {
public:
    explicit ModelConversionError(const std::string& message) : std::runtime_error(message) {}
};

class ModelConverter {
public:
    /**
     * Convert a model from its original format to ONNX format
     * 
     * @param modelPath Path to the original model file
     * @param format Original format of the model (e.g., "pytorch", "tensorflow")
     * @param outputPath Optional path for the output ONNX model. If empty, a default path will be generated.
     * @return Path to the converted ONNX model
     * @throws ModelConversionError if conversion fails
     */
    static std::string convertToONNX(const std::string& modelPath, const std::string& format, const std::string& outputPath = "");

private:
    /**
     * Generate a default output path for the ONNX model
     * 
     * @param inputPath Path to the input model
     * @return Generated output path
     */
    static std::string generateOutputPath(const std::string& inputPath);
};

} // namespace CortexAICompression

#endif // MODEL_CONVERTER_HPP
