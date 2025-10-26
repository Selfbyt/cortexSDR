/**
 * @file ModelConverter.hpp
 * @brief Utilities to convert external model formats into ONNX.
 */
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

/** Error indicating a model conversion failure. */
class ModelConversionError : public std::runtime_error {
public:
    explicit ModelConversionError(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Convert models from frameworks (PyTorch, TF) to ONNX.
 */
class ModelConverter {
public:
    /** Convert a model from its original format to ONNX format. */
    static std::string convertToONNX(const std::string& modelPath, const std::string& format, const std::string& outputPath = "");

private:
    /** Generate a default output path for the ONNX model. */
    static std::string generateOutputPath(const std::string& inputPath);
};

} // namespace CortexAICompression

#endif // MODEL_CONVERTER_HPP
