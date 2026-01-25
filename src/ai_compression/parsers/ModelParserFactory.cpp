/**
 * @file ModelParserFactory.cpp
 * @brief Implementation of model parser factory for automatic format detection
 */
#include "ModelParserFactory.hpp"
#include "ONNXModelParser.hpp"
#include "GGUFModelParser.hpp"
#include "TensorFlowModelParser.hpp"
#include "PyTorchModelParser.hpp"
#include "CoreMLModelParser.hpp"
#include "HDF5ModelParser.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace CortexAICompression {

// Static member initialization
std::map<std::string, std::function<std::unique_ptr<IAIModelParser>()>> ModelParserFactory::parserRegistry;

std::unique_ptr<IAIModelParser> ModelParserFactory::createParser(const std::string& modelPath) {
    // Initialize registry if not already done
    if (parserRegistry.empty()) {
        initializeRegistry();
    }
    
    // Detect the format
    std::string format = detectFormat(modelPath);
    
    // Find the appropriate parser
    auto it = parserRegistry.find(format);
    if (it != parserRegistry.end()) {
        return it->second();
    }
    
    throw ParsingError("No parser available for format: " + format + " (file: " + modelPath + ")");
}

std::string ModelParserFactory::detectFormat(const std::string& modelPath) {
    // First try to detect from file extension
    std::string format = detectFromExtension(modelPath);
    if (!format.empty()) {
        return format;
    }
    
    // If extension detection fails, try content-based detection
    format = detectFromContent(modelPath);
    if (!format.empty()) {
        return format;
    }
    
    throw ParsingError("Unable to detect model format for file: " + modelPath);
}

bool ModelParserFactory::isFormatSupported(const std::string& format) {
    if (parserRegistry.empty()) {
        initializeRegistry();
    }
    
    return parserRegistry.find(format) != parserRegistry.end();
}

std::vector<std::string> ModelParserFactory::getSupportedFormats() {
    if (parserRegistry.empty()) {
        initializeRegistry();
    }
    
    std::vector<std::string> formats;
    for (const auto& pair : parserRegistry) {
        formats.push_back(pair.first);
    }
    
    return formats;
}

std::string ModelParserFactory::detectFromExtension(const std::string& modelPath) {
    // Find the last dot in the path
    size_t lastDot = modelPath.find_last_of('.');
    if (lastDot == std::string::npos) {
        return "";
    }
    
    // Extract extension and convert to lowercase
    std::string extension = modelPath.substr(lastDot + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    // Map extensions to formats
    if (extension == "onnx") {
        return "onnx";
    } else if (extension == "gguf") {
        return "gguf";
    } else if (extension == "pb") {
        return "tensorflow";
    } else if (extension == "pt" || extension == "pth") {
        return "pytorch";
    } else if (extension == "mlmodel") {
        return "coreml";
    } else if (extension == "h5" || extension == "hdf5") {
        return "hdf5";
    }
    
    return "";
}

std::string ModelParserFactory::detectFromContent(const std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file) {
        return "";
    }
    
    // Read first few bytes to check magic numbers
    char magic[16];
    file.read(magic, sizeof(magic));
    size_t bytesRead = file.gcount();
    
    if (bytesRead < 4) {
        return "";
    }
    
    // Check for known magic numbers
    if (bytesRead >= 4) {
        // ONNX models start with a protobuf header
        // GGUF models start with "GGUF" magic
        if (magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F') {
            return "gguf";
        }
        
        // HDF5 files start with specific signature
        if (magic[0] == '\x89' && magic[1] == 'H' && magic[2] == 'D' && magic[3] == 'F') {
            return "hdf5";
        }
        
        // ZIP files (CoreML models are often ZIP archives)
        if (magic[0] == 'P' && magic[1] == 'K' && magic[2] == '\x03' && magic[3] == '\x04') {
            // Could be CoreML, but we need more sophisticated detection
            return "coreml";
        }
        
        // TensorFlow SavedModel files are protobuf
        // PyTorch files have specific structure
        // For now, we'll rely on extension detection for these
    }
    
    return "";
}

void ModelParserFactory::initializeRegistry() {
    // Register ONNX parser
    parserRegistry["onnx"] = []() -> std::unique_ptr<IAIModelParser> {
        return std::make_unique<ONNXModelParser>();
    };
    
    // Register GGUF parser
    parserRegistry["gguf"] = []() -> std::unique_ptr<IAIModelParser> {
        return std::make_unique<GGUFModelParser>();
    };
    
#ifdef ENABLE_TENSORFLOW
    // Register TensorFlow parser
    parserRegistry["tensorflow"] = []() -> std::unique_ptr<IAIModelParser> {
        return std::make_unique<TensorFlowModelParser>();
    };
#endif

#ifdef ENABLE_PYTORCH
    // Register PyTorch parser
    parserRegistry["pytorch"] = []() -> std::unique_ptr<IAIModelParser> {
        return std::make_unique<PyTorchModelParser>();
    };
#endif

    // Register CoreML parser (always available as it doesn't require external dependencies)
    parserRegistry["coreml"] = []() -> std::unique_ptr<IAIModelParser> {
        return std::make_unique<CoreMLModelParser>();
    };

#ifdef ENABLE_HDF5
    // Register HDF5 parser
    parserRegistry["hdf5"] = []() -> std::unique_ptr<IAIModelParser> {
        return std::make_unique<HDF5ModelParser>();
    };
#endif

}

} // namespace CortexAICompression
