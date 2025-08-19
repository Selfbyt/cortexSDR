#include "ai_compression/api/cortex_sdk.h"
#include <iostream>
#include <string>
#include <vector>
#include <signal.h>
#include <chrono>

// Global variables for signal handling
volatile bool g_running = true;
CortexInferenceEngineHandle g_engine = nullptr;

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "\nShutting down..." << std::endl;
    g_running = false;
    if (g_engine) {
        cortex_inference_engine_free(g_engine);
        g_engine = nullptr;
    }
    exit(0);
}

// Helper function to print error information
void print_error(const CortexError& error) {
    if (error.code != CORTEX_OK) {
        std::cerr << "Error " << error.code << ": " << (error.message ? error.message : "Unknown error") << std::endl;
        cortex_error_free(const_cast<CortexError*>(&error));
    }
}

// Simple text tokenization (character-based)
std::vector<float> text_to_tensor(const std::string& text) {
    std::vector<float> tensor;
    tensor.reserve(text.length() * 128);
    
    for (char ch : text) {
        float value = static_cast<float>(static_cast<unsigned char>(ch)) / 255.0f;
        tensor.push_back(value);
        
        if (tensor.size() < 128) {
            tensor.push_back(0.0f);
        }
    }
    
    while (tensor.size() < 128) {
        tensor.push_back(0.0f);
    }
    
    return tensor;
}

// Simple tensor to text conversion
std::string tensor_to_text(const std::vector<float>& tensor, int max_length) {
    std::string result;
    result.reserve(max_length);
    
    for (size_t i = 0; i < tensor.size() && result.length() < max_length; i += 2) {
        if (i + 1 < tensor.size()) {
            float value = tensor[i] * 255.0f;
            int char_code = static_cast<int>(value + 0.5f);
            
            if (char_code >= 32 && char_code <= 126) {
                result += static_cast<char>(char_code);
            } else if (char_code == 10) {
                result += '\n';
            } else if (char_code == 32) {
                result += ' ';
            }
        }
    }
    
    if (result.empty()) {
        result = "Generated text output";
    }
    
    return result;
}

// Load and compress model if needed
bool load_model(const std::string& model_path, float sparsity = 0.02f) {
    std::cout << "Loading model: " << model_path << std::endl;
    
    // Check if model is already compressed (.sdr extension)
    std::string compressed_path = model_path;
    if (compressed_path.find(".sdr") == std::string::npos) {
        compressed_path = model_path + ".sdr";
    }
    
    // Create inference engine
    CortexError error = cortex_inference_engine_create(compressed_path.c_str(), &g_engine);
    if (error.code != CORTEX_OK) {
        print_error(error);
        return false;
    }
    
    // Configure inference engine
    error = cortex_inference_engine_set_batch_size(g_engine, 1);
    if (error.code != CORTEX_OK) {
        print_error(error);
    }
    
    error = cortex_inference_engine_enable_dropout(g_engine, 0); // Disable dropout for inference
    if (error.code != CORTEX_OK) {
        print_error(error);
    }
    
    error = cortex_inference_engine_set_mode(g_engine, 0); // Set to inference mode
    if (error.code != CORTEX_OK) {
        print_error(error);
    }
    
    std::cout << "Model loaded successfully!" << std::endl;
    return true;
}

// Generate text response
std::string generate_text(const std::string& prompt, int max_length = 100) {
    if (!g_engine) {
        return "Error: No model loaded";
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert text input to tensor
    std::vector<float> input_tensor = text_to_tensor(prompt);
    
    // Prepare output buffer
    const size_t max_output_size = 1000; // Adjust based on your model's expected output size
    std::vector<float> output_tensor(max_output_size);
    size_t actual_output_size = 0;
    
    // Run inference
    CortexError error = cortex_inference_engine_run(
        g_engine,
        input_tensor.data(),
        input_tensor.size(),
        output_tensor.data(),
        output_tensor.size(),
        &actual_output_size
    );
    
    if (error.code != CORTEX_OK) {
        print_error(error);
        return "Error: Inference failed";
    }
    
    // Resize output vector to actual size
    output_tensor.resize(actual_output_size);
    
    // Convert output tensor back to text
    std::string result = tensor_to_text(output_tensor, max_length);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Generated in " << duration.count() << "ms" << std::endl;
    
    return result;
}

// Interactive chat loop
void interactive_chat() {
    std::cout << "\n=== CortexSDR Text Generation CLI ===" << std::endl;
    std::cout << "Type your messages and press Enter." << std::endl;
    std::cout << "Commands: 'quit' to exit, 'load <model>' to load a model" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    std::string input;
    while (g_running) {
        std::cout << "\n> ";
        std::getline(std::cin, input);
        
        if (input == "quit" || input == "exit") {
            break;
        }
        
        if (input.empty()) {
            continue;
        }
        
        // Check for commands
        if (input.substr(0, 5) == "load ") {
            std::string model_path = input.substr(5);
            if (load_model(model_path)) {
                std::cout << "Model loaded successfully!" << std::endl;
            } else {
                std::cout << "Failed to load model: " << model_path << std::endl;
            }
            continue;
        }
        
        if (input == "help") {
            std::cout << "Available commands:" << std::endl;
            std::cout << "  load <model_path>  - Load a new model" << std::endl;
            std::cout << "  quit/exit          - Exit the application" << std::endl;
            std::cout << "  help               - Show this help" << std::endl;
            std::cout << "  <any text>         - Generate response" << std::endl;
            continue;
        }
        
        // Check if model is loaded
        if (!g_engine) {
            std::cout << "No model loaded. Use 'load <model_path>' to load a model first." << std::endl;
            continue;
        }
        
        std::cout << "Generating response..." << std::endl;
        std::string response = generate_text(input);
        std::cout << "Response: " << response << std::endl;
    }
}

// Print usage information
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [MODEL_PATH]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  -v, --version           Show version information" << std::endl;
    std::cout << "  -i, --interactive       Start interactive chat mode" << std::endl;
    std::cout << "  -p, --prompt TEXT       Generate text from prompt and exit" << std::endl;
    std::cout << "  -m, --max-length N      Maximum output length (default: 100)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                              # Start interactive mode (no model)" << std::endl;
    std::cout << "  " << program_name << " model.sdr                    # Load model and start interactive mode" << std::endl;
    std::cout << "  " << program_name << " -i                           # Start interactive chat (load model later)" << std::endl;
    std::cout << "  " << program_name << " -p \"Hello, how are you?\" model.sdr  # Generate single response" << std::endl;
    std::cout << "  " << program_name << " -m 200 -p \"Tell me a story\" model.sdr  # Generate longer response" << std::endl;
    std::cout << std::endl;
    std::cout << "Interactive Commands:" << std::endl;
    std::cout << "  load <model_path>       Load a model" << std::endl;
    std::cout << "  help                    Show available commands" << std::endl;
    std::cout << "  quit/exit               Exit the application" << std::endl;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "CortexSDR CLI - Version: " << cortex_sdk_version() << std::endl;
    
    // Parse command line arguments
    std::string model_path;
    std::string prompt;
    bool interactive_mode = false;
    int max_length = 100;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--version") {
            std::cout << "CortexSDR CLI Version: " << cortex_sdk_version() << std::endl;
            return 0;
        } else if (arg == "-i" || arg == "--interactive") {
            interactive_mode = true;
        } else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 < argc) {
                prompt = argv[++i];
            } else {
                std::cerr << "Error: --prompt requires a text argument" << std::endl;
                return 1;
            }
        } else if (arg == "-m" || arg == "--max-length") {
            if (i + 1 < argc) {
                max_length = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --max-length requires a number argument" << std::endl;
                return 1;
            }
        } else if (arg[0] != '-') {
            // This is the model path
            if (model_path.empty()) {
                model_path = arg;
            } else {
                std::cerr << "Error: Multiple model paths specified" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown option " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Load the model if provided
    if (!model_path.empty()) {
        if (!load_model(model_path)) {
            std::cerr << "Failed to load model: " << model_path << std::endl;
            return 1;
        }
    } else {
        std::cout << "No model specified. Use 'load <model_path>' to load a model." << std::endl;
    }
    
    // Run in appropriate mode
    if (!prompt.empty()) {
        // Single prompt mode
        std::cout << "Generating response for: " << prompt << std::endl;
        std::string response = generate_text(prompt, max_length);
        std::cout << response << std::endl;
    } else {
        // Interactive mode (default)
        interactive_chat();
    }
    
    // Cleanup
    if (g_engine) {
        cortex_inference_engine_free(g_engine);
    }
    
    return 0;
}
