#include "ModelConverter.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <filesystem>
#include <cstdlib>
#include <array>
#include <string>
#include <stdexcept>

namespace CortexAICompression {

std::string ModelConverter::convertToONNX(const std::string& modelPath, const std::string& format, const std::string& outputPath) {
    // Check if the input file exists
    if (!std::filesystem::exists(modelPath)) {
        throw ModelConversionError("Input model file does not exist: " + modelPath);
    }
    
    // If already in ONNX format, just return the path
    if (format == "onnx") {
        return modelPath;
    }
    
    // Generate output path if not provided
    std::string actualOutputPath = outputPath.empty() ? generateOutputPath(modelPath) : outputPath;
    
    // Provide information about model conversion
    std::cout << "Model conversion from " << format << " to ONNX format" << std::endl;
    std::cout << "Input model: " << modelPath << std::endl;
    std::cout << "Output model: " << actualOutputPath << std::endl;
    
#ifdef ENABLE_PYTORCH
    if (format == "pytorch") {
        std::cout << "Using PyTorch to ONNX conversion..." << std::endl;
        
        // Create a Python script to convert the model
        std::string scriptPath = std::filesystem::path(modelPath).parent_path().string() + "/torch_to_onnx_converter.py";
        std::ofstream scriptFile(scriptPath);
        
        if (!scriptFile) {
            throw ModelConversionError("Failed to create conversion script");
        }
        
        // Write a Python script that uses torch.onnx.export to convert the model
        scriptFile << "import torch\n"
                  << "import sys\n\n"
                  << "try:\n"
                  << "    # Load the model\n"
                  << "    model = torch.jit.load('" << modelPath << "')\n"
                  << "    model.eval()\n\n"
                  << "    # Create a dummy input tensor (adjust shape as needed for your model)\n"
                  << "    dummy_input = torch.ones(1, 3, 224, 224)\n\n"
                  << "    # Export the model to ONNX format\n"
                  << "    torch.onnx.export(model, dummy_input, '" << actualOutputPath << "', \n"
                  << "                      export_params=True, opset_version=12, \n"
                  << "                      do_constant_folding=True, input_names=['input'], \n"
                  << "                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, \n"
                  << "                                                            'output': {0: 'batch_size'}})\n"
                  << "    print('Model successfully converted to ONNX format')\n"
                  << "    sys.exit(0)\n"
                  << "except Exception as e:\n"
                  << "    print(f'Error: {str(e)}', file=sys.stderr)\n"
                  << "    sys.exit(1)\n";
        
        scriptFile.close();
        
        // Execute the Python script
        std::string command = "python " + scriptPath;
        int result = std::system(command.c_str());
        
        // Clean up the temporary script
        std::filesystem::remove(scriptPath);
        
        if (result != 0) {
            throw ModelConversionError("PyTorch to ONNX conversion failed. Check if PyTorch is installed with ONNX support.");
        }
        
        return actualOutputPath;
    }
#endif

#ifdef ENABLE_TENSORFLOW
    if (format == "tensorflow") {
        std::cout << "Using TensorFlow to ONNX conversion..." << std::endl;
        
        // Create a Python script to convert the model
        std::string scriptPath = std::filesystem::path(modelPath).parent_path().string() + "/tf_to_onnx_converter.py";
        std::ofstream scriptFile(scriptPath);
        
        if (!scriptFile) {
            throw ModelConversionError("Failed to create conversion script");
        }
        
        // Write a Python script that uses tf2onnx to convert the model
        scriptFile << "import tensorflow as tf\n"
                  << "import tf2onnx\n"
                  << "import sys\n\n"
                  << "try:\n"
                  << "    # Load the TensorFlow model\n"
                  << "    model = tf.saved_model.load('" << modelPath << "')\n\n"
                  << "    # Convert the model to ONNX\n"
                  << "    onnx_model, _ = tf2onnx.convert.from_keras(model)\n\n"
                  << "    # Save the ONNX model\n"
                  << "    with open('" << actualOutputPath << "', 'wb') as f:\n"
                  << "        f.write(onnx_model.SerializeToString())\n\n"
                  << "    print('Model successfully converted to ONNX format')\n"
                  << "    sys.exit(0)\n"
                  << "except Exception as e:\n"
                  << "    print(f'Error: {str(e)}', file=sys.stderr)\n"
                  << "    sys.exit(1)\n";
        
        scriptFile.close();
        
        // Execute the Python script
        std::string command = "python " + scriptPath;
        int result = std::system(command.c_str());
        
        // Clean up the temporary script
        std::filesystem::remove(scriptPath);
        
        if (result != 0) {
            throw ModelConversionError("TensorFlow to ONNX conversion failed. Check if TensorFlow and tf2onnx are installed.");
        }
        
        return actualOutputPath;
    }
#endif

    // If we get here, the format is not supported
    std::string errorMsg = "Unsupported model format: " + format;
    
    // Provide more specific error messages based on the requested format
    if (format == "tensorflow") {
#ifndef ENABLE_TENSORFLOW
        errorMsg += ". TensorFlow support is not enabled in this build";
#endif
    } else if (format == "pytorch") {
#ifndef ENABLE_PYTORCH
        errorMsg += ". PyTorch support is not enabled in this build";
#endif
    }
    
    errorMsg += ". Please convert your model to ONNX format manually.";
    throw ModelConversionError(errorMsg);
}



std::string ModelConverter::generateOutputPath(const std::string& inputPath) {
    std::filesystem::path path(inputPath);
    std::filesystem::path outputPath = path.parent_path() / (path.stem().string() + "_converted.onnx");
    return outputPath.string();
}

} // namespace CortexAICompression
