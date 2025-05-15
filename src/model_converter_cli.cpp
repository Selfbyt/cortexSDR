#include "ai_compression/utils/ModelConverter.hpp"
#include <iostream>
#include <string>

using namespace CortexAICompression;

void printUsage(const char* programName) {
    std::cout << "CortexSDR Model Converter - Convert ML models to ONNX format\n";
    std::cout << "Usage: " << programName << " <input_model> <output_model.onnx> <input_format>\n";
    std::cout << "Supported formats:\n";
    
#ifdef ENABLE_PYTORCH
    std::cout << "  - pytorch: PyTorch models (.pt, .pth)\n";
#endif
    
#ifdef ENABLE_TENSORFLOW
    std::cout << "  - tensorflow: TensorFlow models (SavedModel directory)\n";
#endif
    
    std::cout << "  - onnx: ONNX models (already in ONNX format, will be copied)\n";
    
    if (
#if !defined(ENABLE_PYTORCH) && !defined(ENABLE_TENSORFLOW)
        true
#else
        false
#endif
    ) {
        std::cout << "\nNOTE: This build does not include PyTorch or TensorFlow conversion support.\n";
        std::cout << "To enable conversion support, rebuild with ENABLE_PYTORCH=ON and/or ENABLE_TENSORFLOW=ON.\n";
    }
    
    std::cout << "\nExample:\n";
    std::cout << "  " << programName << " model.pt model.onnx pytorch\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputModel = argv[1];
    std::string outputModel = argv[2];
    std::string inputFormat = argv[3];

    try {
        std::cout << "Converting " << inputFormat << " model to ONNX format...\n";
        std::string convertedPath = ModelConverter::convertToONNX(inputModel, inputFormat, outputModel);
        std::cout << "Successfully converted model to ONNX format: " << convertedPath << std::endl;
        return 0;
    } catch (const ModelConversionError& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        
        // Provide helpful information based on the format
        if (inputFormat == "pytorch") {
#ifdef ENABLE_PYTORCH
            std::cerr << "For PyTorch models, make sure:\n";
            std::cerr << "- The model is saved using torch.save() or torch.jit.save()\n";
            std::cerr << "- LibTorch is properly installed\n";
#else
            std::cerr << "PyTorch conversion support is not enabled in this build.\n";
            std::cerr << "Rebuild with ENABLE_PYTORCH=ON to enable PyTorch conversion.\n";
#endif
        } else if (inputFormat == "tensorflow") {
#ifdef ENABLE_TENSORFLOW
            std::cerr << "For TensorFlow models, make sure:\n";
            std::cerr << "- The model is saved as a SavedModel directory\n";
            std::cerr << "- TensorFlow and tf2onnx are properly installed\n";
#else
            std::cerr << "TensorFlow conversion support is not enabled in this build.\n";
            std::cerr << "Rebuild with ENABLE_TENSORFLOW=ON to enable TensorFlow conversion.\n";
#endif
        }
        
        std::cerr << "\nAlternatively, you can convert your model manually using the appropriate tools:\n";
        std::cerr << "- For PyTorch models: torch.onnx.export()\n";
        std::cerr << "- For TensorFlow models: tf2onnx.convert\n";
        std::cerr << "For more information, see: https://github.com/onnx/tutorials\n";
        
        return 1;
    }
}
