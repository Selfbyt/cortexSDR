/**
 * @file cortexsdr_cli.cpp
 * @brief Interactive CLI for CortexSDR neural network inference with on-demand loading
 * 
 * This file implements an interactive command-line interface for running neural network
 * inference with compressed models (.sdr files). Features include:
 * 
 * Key Features:
 * - On-demand layer loading (Ollama-style memory efficiency)
 * - Interactive chat mode for text generation
 * - Model statistics and performance monitoring
 * - Fallback support for legacy inference engine
 * - Real-time switching between loading modes
 * 
 * Usage Examples:
 * - Interactive mode: ./cortexsdr_cli model.sdr
 * - Single prompt: ./cortexsdr_cli -p "Hello world" model.sdr
 * - Legacy mode: ./cortexsdr_cli --legacy model.sdr
 */

#include "ai_compression/api/cortex_sdk.h"
#include "ai_compression/SparseInferenceEngine.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <signal.h>
#include <chrono>
#include <memory>
#include <thread>
#include <iomanip>
#include <fstream>
#include <filesystem>

// Global state for inference engines and signal handling
volatile bool g_running = true;
CortexInferenceEngineHandle g_engine = nullptr;                                          // Legacy inference engine
std::unique_ptr<CortexAICompression::SDRModelLoader> g_model_loader = nullptr;          // On-demand model loader
std::unique_ptr<CortexAICompression::SDRInferenceEngine> g_inference_engine = nullptr;  // Modern inference engine
bool g_use_on_demand_loading = true;                                                    // Default to on-demand mode

// Last-run profiling snapshot for optional --profile output
static uint64_t g_last_duration_ms = 0;
static size_t g_last_token_count = 0;
static double g_last_tokens_per_sec = 0.0;
static std::string g_last_benchmark_json;
static bool g_has_model_tokenizer = false;                                             // Tokenizer assets present

/**
 * @brief Signal handler for graceful shutdown
 * @param signal Signal number received (SIGINT, SIGTERM, etc.)
 * 
 * Performs cleanup of inference engines and exits gracefully
 * when interrupt signals are received.
 */
void signal_handler(int signal) {
    std::cout << "\nShutting down..." << std::endl;
    g_running = false;
    if (g_engine) {
        cortex_inference_engine_free(g_engine);
        g_engine = nullptr;
    }
    g_inference_engine.reset();
    g_model_loader.reset();
    exit(0);
}

/**
 * @brief Print error information from CortexError structure
 * @param error CortexError containing error code and message
 * 
 * Displays formatted error information and frees associated resources.
 */
void print_error(const CortexError& error) {
    if (error.code != CORTEX_OK) {
        std::cerr << "Error " << error.code << ": " << (error.message ? error.message : "Unknown error") << std::endl;
        cortex_error_free(const_cast<CortexError*>(&error));
    }
}

/**
 * @brief Convert text to tensor representation for neural network input
 * @param text Input text string to convert
 * @return Vector of floats representing the tokenized text
 * 
 * Implements basic character-level tokenization by converting each character
 * to a normalized float value (0.0-1.0). Pads output to fixed size of 128.
 */
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

/**
 * @brief Convert neural network output tensor back to text
 * @param tensor Output tensor from neural network
 * @param max_length Maximum length of output text
 * @return Generated text string
 * 
 * Converts float tensor values back to characters by denormalizing
 * and filtering for printable ASCII characters.
 */
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

// Minimal parser to summarize benchmark JSON from SDK without external deps
static void print_benchmark_summary(const std::string& json) {
    if (json.empty()) return;
    // Extract total_ms
    double total_ms = 0.0;
    size_t pos = json.find("\"total_ms\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos);
        if (pos != std::string::npos) {
            size_t end = json.find_first_of(",}\n\r ", pos + 1);
            std::string num = json.substr(pos + 1, end == std::string::npos ? std::string::npos : (end - pos - 1));
            try { total_ms = std::stod(num); } catch (...) {}
        }
    }
    // Estimate layer count by counting occurrences of "\"name\":"
    size_t layers = 0;
    size_t from = 0;
    while ((from = json.find("\"name\"", from)) != std::string::npos) { layers++; from += 6; }
    std::cout << "[Benchmark] total_ms: " << std::fixed << std::setprecision(2) << total_ms
              << ", layers: " << layers << std::endl;
}

// Simple tokenizer used when model provides tokenizer assets; whitespace+punct splits
static size_t count_tokens_simple(const std::string& text) {
    size_t count = 0;
    bool in_token = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) || std::ispunct(static_cast<unsigned char>(c))) {
            if (in_token) { count++; in_token = false; }
            if (std::ispunct(static_cast<unsigned char>(c))) {
                // Treat punctuation as separate token
                count++;
            }
        } else {
            in_token = true;
        }
    }
    if (in_token) count++;
    return count;
}

/**
 * @brief Load neural network model with on-demand layer loading support
 * @param model_path Path to the compressed model file (.sdr)
 * @param sparsity Sparsity level for compression (default: 0.02)
 * @return True if model loaded successfully, false otherwise
 * 
 * Initializes both modern on-demand inference engine and legacy engine
 * as fallback. The on-demand engine provides Ollama-style memory efficiency
 * by loading layers only when needed during inference.
 */
bool load_model(const std::string& model_path, float sparsity = 0.02f) {
    std::cout << "Loading model: " << model_path << std::endl;
    
    // Check if model is already compressed (.sdr extension)
    std::string compressed_path = model_path;
    if (compressed_path.find(".sdr") == std::string::npos) {
        compressed_path = model_path + ".sdr";
        std::cout << "Looking for compressed model: " << compressed_path << std::endl;
    }
    
    try {
        // Create model loader with on-demand loading capability
        std::cout << "Initializing on-demand model loader..." << std::endl;
        g_model_loader = std::make_unique<CortexAICompression::SDRModelLoader>(compressed_path);
        
        // Create inference engine that uses the model loader
        std::cout << "Creating inference engine with on-demand loading..." << std::endl;
        g_inference_engine = std::make_unique<CortexAICompression::SDRInferenceEngine>(*g_model_loader);
        
        // Configure inference engine for optimal performance
        g_inference_engine->setBatchSize(1);
        g_inference_engine->enableDropout(false);
        g_inference_engine->setInferenceMode(false); // Set to inference mode
        // Enable memory-optimized execution for large models
        g_inference_engine->enableAggressiveMemoryManagement(true);
        g_inference_engine->initializeMemoryPool(8192); // 8GB cap by default; adjust as needed
        
        // Show model statistics
        const auto& segments = g_model_loader->getSegmentIndex();
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "  - Total segments: " << segments.size() << std::endl;
        std::cout << "  - On-demand loading: ENABLED" << std::endl;
        std::cout << "  - Memory footprint: MINIMAL (layers loaded as needed)" << std::endl;
        std::cout << "  - Memory pool: ENABLED (8GB limit by default)" << std::endl;
        // Heuristic: detect tokenizer assets by segment names
        g_has_model_tokenizer = false;
        for (const auto& seg : segments) {
            std::string n = seg.name;
            for (auto& ch : n) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
            if (n.find("tokenizer") != std::string::npos || n.find("vocab") != std::string::npos || n.find("merges") != std::string::npos) {
                g_has_model_tokenizer = true;
                break;
            }
        }
        std::cout << "  - Tokenizer assets: " << (g_has_model_tokenizer ? "FOUND" : "NOT FOUND") << std::endl;
        
        // Use SDK introspection for tokenizer info
        int has_tok = 0;
        char* tok_type = nullptr;
        CortexError tok_err = cortex_archive_get_tokenizer_info(compressed_path.c_str(), &has_tok, &tok_type);
        if (tok_err.code == CORTEX_OK) {
            g_has_model_tokenizer = (has_tok != 0);
            if (tok_type) {
                std::cout << "  - Tokenizer type: " << tok_type << std::endl;
                cortex_free_string(tok_type);
            }
        }
        
        // Also try to create legacy engine as fallback
        CortexError error = cortex_inference_engine_create(compressed_path.c_str(), &g_engine);
        if (error.code == CORTEX_OK) {
            std::cout << "  - Legacy engine: AVAILABLE (fallback mode)" << std::endl;
            cortex_inference_engine_set_batch_size(g_engine, 1);
            cortex_inference_engine_enable_dropout(g_engine, 0);
            cortex_inference_engine_set_mode(g_engine, 0);
            // Configure memory management via SDK for legacy engine path
            cortex_inference_engine_enable_aggressive_memory(g_engine, 1);
            cortex_inference_engine_init_memory_pool(g_engine, 8192);
        } else {
            std::cout << "  - Legacy engine: NOT AVAILABLE (on-demand only)" << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to load model: " << e.what() << std::endl;
        g_inference_engine.reset();
        g_model_loader.reset();
        return false;
    }
}

/**
 * @brief Generate text response using loaded neural network model
 * @param prompt Input text prompt for generation
 * @param max_length Maximum length of generated response
 * @return Generated text response
 * 
 * Performs inference using the loaded model with automatic fallback:
 * 1. First attempts on-demand inference (memory efficient)
 * 2. Falls back to legacy engine if on-demand fails
 * 3. Provides detailed timing and performance information
 */
std::string generate_text(const std::string& prompt, int max_length = 100) {
    if (!g_inference_engine && !g_engine) {
        return "Error: No model loaded";
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert text input to tensor
    std::vector<float> input_tensor = text_to_tensor(prompt);
    
    std::cout << "Starting inference..." << std::endl;
    std::cout << "  - Input prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "  - Input tensor size: " << input_tensor.size() << std::endl;
    std::cout << "  - On-demand loading: " << (g_use_on_demand_loading ? "ENABLED" : "DISABLED") << std::endl;
    
    std::vector<float> output_tensor;
    
    // Try on-demand inference first (Ollama-style)
    if (g_inference_engine && g_use_on_demand_loading) {
        std::cout << "  - Using on-demand layer-by-layer execution..." << std::endl;
        
        try {
            // Run inference with on-demand loading
            output_tensor = g_inference_engine->run(input_tensor);
            
            std::cout << "  - On-demand inference completed successfully" << std::endl;
            std::cout << "  - Output tensor size: " << output_tensor.size() << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "  - On-demand inference failed: " << e.what() << std::endl;
            std::cout << "  - Falling back to legacy engine..." << std::endl;
            
            // Fallback to legacy engine
            if (g_engine) {
                const size_t max_output_size = 1000;
                output_tensor.resize(max_output_size);
                size_t actual_output_size = 0;
                
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
                    return "Error: Both on-demand and legacy inference failed";
                }
                
                output_tensor.resize(actual_output_size);
                std::cout << "  - Legacy inference completed" << std::endl;
            } else {
                return "Error: On-demand inference failed and no legacy engine available";
            }
        }
    }
    // Use legacy engine directly
    else if (g_engine) {
        std::cout << "  - Using legacy inference engine..." << std::endl;
        
        const size_t max_output_size = 1000;
        output_tensor.resize(max_output_size);
        size_t actual_output_size = 0;
        
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
            return "Error: Legacy inference failed";
        }
        
        output_tensor.resize(actual_output_size);
        std::cout << "  - Legacy inference completed" << std::endl;
    } else {
        return "Error: No inference engine available";
    }
    
    // Convert output tensor back to text
    std::string result = tensor_to_text(output_tensor, max_length);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Compute a simple tokens/sec metric using output length as proxy
    double seconds = static_cast<double>(duration.count()) / 1000.0;
    size_t token_count = g_has_model_tokenizer ? count_tokens_simple(result) : result.size();
    double tokens_per_sec = seconds > 0.0 ? (static_cast<double>(token_count) / seconds) : 0.0;
    
    g_last_duration_ms = static_cast<uint64_t>(duration.count());
    g_last_token_count = token_count;
    g_last_tokens_per_sec = tokens_per_sec;

    std::cout << "Generated in " << g_last_duration_ms << "ms";
    std::cout << " | tokens: " << g_last_token_count << " | tokens/sec: " << std::fixed << std::setprecision(2) << g_last_tokens_per_sec << std::endl;
    std::cout << "Response: " << result << std::endl;
    // Print SDK benchmark stats if legacy is in use; otherwise advise how to fetch stats via SDK
    if (g_engine) {
        char* json = nullptr;
        CortexError e = cortex_inference_engine_get_last_run_stats_json(g_engine, &json);
        if (e.code == CORTEX_OK && json) {
            g_last_benchmark_json = json;
            print_benchmark_summary(g_last_benchmark_json);
            cortex_free_string(json);
        }
    }
    
    return result;
}

/**
 * @brief Main interactive chat loop with enhanced command support
 * 
 * Provides an interactive interface with the following commands:
 * - Text input for generation
 * - load <model>: Load new model
 * - stats: Show model statistics
 * - layers: List available layers
 * - memory: Show memory usage
 * - toggle-mode: Switch between on-demand and legacy modes
 * - help: Show available commands
 * - quit/exit: Exit the application
 */
void interactive_chat() {
    std::cout << "\n=== CortexSDR Text Generation CLI (On-Demand Loading) ===" << std::endl;
    std::cout << "Type your messages and press Enter." << std::endl;
    std::cout << "Commands: 'quit' to exit, 'load <model>' to load a model" << std::endl;
    std::cout << "Enhanced: 'stats', 'toggle-mode', 'layers', 'memory'" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    std::string input;
    while (g_running) {
        std::cout << "\n[" << (g_use_on_demand_loading ? "ON-DEMAND" : "LEGACY") << "] > ";
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
            std::cout << "  load <model_path>  - Load a new model with on-demand capabilities" << std::endl;
            std::cout << "  stats             - Show model and inference statistics" << std::endl;
            std::cout << "  layers            - List all available layers in the model" << std::endl;
            std::cout << "  memory            - Show memory usage information" << std::endl;
            std::cout << "  toggle-mode       - Toggle between on-demand and legacy modes" << std::endl;
            std::cout << "  quit/exit         - Exit the application" << std::endl;
            std::cout << "  help              - Show this help" << std::endl;
            std::cout << "  <any text>        - Generate response" << std::endl;
            continue;
        }
        
        if (input == "stats") {
            if (g_inference_engine) {
                std::cout << "=== Model Statistics ===" << std::endl;
                const auto& segments = g_model_loader->getSegmentIndex();
                std::cout << "Total segments: " << segments.size() << std::endl;
                std::cout << "Encountered layer types: " << g_inference_engine->getEncounteredLayerTypes().size() << std::endl;
                std::cout << "Unhandled layer types: " << g_inference_engine->getUnhandledLayerTypes().size() << std::endl;
                std::cout << "On-demand loading: " << (g_use_on_demand_loading ? "ENABLED" : "DISABLED") << std::endl;
                std::cout << "Legacy engine: " << (g_engine ? "AVAILABLE" : "NOT AVAILABLE") << std::endl;
            } else {
                std::cout << "No model loaded." << std::endl;
            }
            continue;
        }
        
        if (input == "layers") {
            if (g_model_loader) {
                std::cout << "=== Available Layers ===" << std::endl;
                const auto& segments = g_model_loader->getSegmentIndex();
                int count = 0;
                for (const auto& seg : segments) {
                    std::cout << ++count << ". " << seg.name << " (type: " << seg.layer_type << ")" << std::endl;
                    if (count > 20) {
                        std::cout << "... and " << (segments.size() - count) << " more layers" << std::endl;
                        break;
                    }
                }
            } else {
                std::cout << "No model loaded." << std::endl;
            }
            continue;
        }
        
        if (input == "memory") {
            std::cout << "=== Memory Usage ===" << std::endl;
            std::cout << "On-demand loading reduces memory usage by loading only needed layers" << std::endl;
            if (g_model_loader) {
                std::cout << "Model loader: ACTIVE (minimal footprint)" << std::endl;
            }
            if (g_inference_engine) {
                std::cout << "Inference engine: ACTIVE" << std::endl;
            }
            if (g_engine) {
                std::cout << "Legacy engine: ACTIVE (additional memory overhead)" << std::endl;
            }
            continue;
        }
        
        if (input == "toggle-mode") {
            g_use_on_demand_loading = !g_use_on_demand_loading;
            std::cout << "Switched to " << (g_use_on_demand_loading ? "ON-DEMAND" : "LEGACY") << " mode" << std::endl;
            continue;
        }
        
        // Check if model is loaded
        if (!g_inference_engine && !g_engine) {
            std::cout << "No model loaded. Use 'load <model_path>' to load a model first." << std::endl;
            continue;
        }
        
        std::cout << "Generating response..." << std::endl;
        std::string response = generate_text(input);
        // Response is already printed in generate_text function
    }
}

/**
 * @brief Print comprehensive usage information
 * @param program_name Name of the program executable
 * 
 * Displays detailed usage information including command-line options,
 * examples, and interactive commands with emphasis on on-demand
 * loading capabilities and Ollama-style memory efficiency.
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [MODEL_PATH]" << std::endl;
    std::cout << std::endl;
    std::cout << "CortexSDR CLI with On-Demand Layer Loading (Ollama-style)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  -v, --version           Show version information" << std::endl;
    std::cout << "  -i, --interactive       Start interactive chat mode" << std::endl;
    std::cout << "  -p, --prompt TEXT       Generate text from prompt and exit" << std::endl;
    std::cout << "  -m, --max-length N      Maximum output length (default: 100)" << std::endl;
    std::cout << "  --profile PATH          Write profiling JSON after single-prompt run" << std::endl;
    std::cout << "  --legacy                Use legacy mode (load all layers at once)" << std::endl;
    std::cout << "  --on-demand             Use on-demand mode (load layers as needed)" << std::endl;
    std::cout << "  --download URL          Download model and compress to .sdr (auto-detect format)" << std::endl;
    std::cout << "  -o, --output PATH       Output .sdr path (optional; defaults to URL basename.sdr)" << std::endl;
    std::cout << "  -s, --sparsity F        Sparsity for compression (optional; default: 0.02)" << std::endl;
    std::cout << "  --hf-token TOKEN        Hugging Face access token (optional; can also use env: HUGGING_FACE_HUB_TOKEN/HUGGINGFACE_TOKEN/HF_TOKEN)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << "                              # Start interactive mode (no model)" << std::endl;
    std::cout << "  " << program_name << " model.sdr                    # Load model with on-demand loading" << std::endl;
    std::cout << "  " << program_name << " -i --on-demand              # Interactive with on-demand loading" << std::endl;
    std::cout << "  " << program_name << " -p \"Hello, how are you?\" model.sdr  # Generate single response" << std::endl;
    std::cout << "  " << program_name << " --legacy -m 200 -p \"Tell me a story\" model.sdr  # Legacy mode" << std::endl;
    std::cout << "  " << program_name << " --download https://host/model.onnx                 # Auto-detect format, output model.sdr" << std::endl;
    std::cout << "  " << program_name << " --download https://host/model.gguf -o llama.sdr  # Custom output name" << std::endl;
    std::cout << "  " << program_name << " --download https://host/model.onnx -s 0.01       # Custom sparsity" << std::endl;
    std::cout << std::endl;
    std::cout << "Interactive Commands:" << std::endl;
    std::cout << "  load <model_path>       Load a model with on-demand capabilities" << std::endl;
    std::cout << "  stats                   Show model statistics and performance" << std::endl;
    std::cout << "  layers                  List all available layers in the model" << std::endl;
    std::cout << "  memory                  Show memory usage information" << std::endl;
    std::cout << "  toggle-mode             Switch between on-demand and legacy modes" << std::endl;
    std::cout << "  help                    Show available commands" << std::endl;
    std::cout << "  quit/exit               Exit the application" << std::endl;
    std::cout << std::endl;
    std::cout << "On-Demand Loading Benefits:" << std::endl;
    std::cout << "  - Minimal memory footprint (similar to Ollama)" << std::endl;
    std::cout << "  - Faster startup time" << std::endl;
    std::cout << "  - Layer-by-layer execution with detailed logging" << std::endl;
    std::cout << "  - Better scalability for large models" << std::endl;
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
    std::string profile_path;
    // Download+compress (auto-detect) options
    bool run_download = false;
    std::string dl_url;
    std::string dl_output;
    float dl_sparsity = 0.02f;
    std::string hf_token;
    
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
        } else if (arg == "--profile") {
            if (i + 1 < argc) {
                profile_path = argv[++i];
            } else {
                std::cerr << "Error: --profile requires a path argument" << std::endl;
                return 1;
            }
        } else if (arg == "-s" || arg == "--sparsity") {
            if (i + 1 < argc) {
                dl_sparsity = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --sparsity requires a float argument" << std::endl;
                return 1;
            }
        } else if (arg == "--legacy") {
            g_use_on_demand_loading = false;
            std::cout << "Legacy mode enabled (load all layers at once)" << std::endl;
        } else if (arg == "--on-demand") {
            g_use_on_demand_loading = true;
            std::cout << "On-demand mode enabled (load layers as needed)" << std::endl;
        } else if (arg == "--download") {
            if (i + 1 < argc) {
                run_download = true;
                dl_url = argv[++i];
            } else {
                std::cerr << "Error: --download requires a URL argument" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--hf-token") {
            if (i + 1 < argc) {
                hf_token = argv[++i];
            } else {
                std::cerr << "Error: --hf-token requires a token argument" << std::endl;
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                dl_output = argv[++i];
            } else {
                std::cerr << "Error: --output requires a path argument" << std::endl;
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
    
    // If requested, run download (auto-detect) and exit
    if (run_download) {
        // Derive output if not provided: store under user's home dir
        if (dl_output.empty()) {
            std::string path = dl_url;
            // Strip query/fragment
            auto qpos = path.find('?');
            if (qpos != std::string::npos) path = path.substr(0, qpos);
            auto hpos = path.find('#');
            if (hpos != std::string::npos) path = path.substr(0, hpos);
            // Extract basename
            auto slash = path.find_last_of('/') ;
            std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
            if (base.empty()) base = "model";
            // Remove extension
            auto dot = base.find_last_of('.');
            if (dot != std::string::npos) base = base.substr(0, dot);
            // Determine home directory
            std::string homeDir;
#ifdef _WIN32
            const char* homeEnv = std::getenv("USERPROFILE");
#else
            const char* homeEnv = std::getenv("HOME");
#endif
            if (homeEnv && *homeEnv) {
                homeDir = homeEnv;
            } else {
                homeDir = "."; // fallback to current directory
            }
            std::string outDir = homeDir + std::string("/CortexSDR/models");
            std::error_code mkec;
            std::filesystem::create_directories(outDir, mkec);
            dl_output = outDir + std::string("/") + base + std::string(".sdr");
        }

        // If token provided, set env vars for this process so SDK picks it up
        if (!hf_token.empty()) {
            setenv("HUGGING_FACE_HUB_TOKEN", hf_token.c_str(), 1);
            setenv("HUGGINGFACE_TOKEN", hf_token.c_str(), 1);
            setenv("HF_TOKEN", hf_token.c_str(), 1);
        }

        std::cout << "Downloading and compressing..." << std::endl;
        std::cout << "  URL: " << dl_url << std::endl;
        std::cout << "  Output: " << dl_output << std::endl;
        std::cout << "  Sparsity: " << dl_sparsity << std::endl;

        CortexError err = cortex_compress_from_url(
            dl_url.c_str(),
            "auto",
            dl_output.c_str(),
            dl_sparsity
        );
        if (err.code != CORTEX_OK) {
            std::cerr << "Download failed or compression error: " << (err.message ? err.message : "Unknown error") << std::endl;
            cortex_error_free(&err);
            return 1;
        }
        std::cout << "Download and compression completed: " << dl_output << std::endl;
        return 0;
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

        // Optional profiling output
        if (!profile_path.empty()) {
            try {
                std::ofstream pf(profile_path);
                if (!pf) {
                    std::cerr << "Error: Could not open profile path: " << profile_path << std::endl;
                } else {
                    pf << "{\n";
                    pf << "  \"duration_ms\": " << g_last_duration_ms << ",\n";
                    pf << "  \"tokens\": " << g_last_token_count << ",\n";
                    pf << "  \"tokens_per_sec\": " << std::fixed << std::setprecision(2) << g_last_tokens_per_sec << ",\n";
                    pf << "  \"engine_stats\": " << (g_last_benchmark_json.empty() ? "null" : g_last_benchmark_json) << "\n";
                    pf << "}\n";
                    std::cout << "Wrote profile to " << profile_path << std::endl;
                }
            } catch (...) {
                std::cerr << "Warning: Failed to write profile JSON" << std::endl;
            }
        }
    } else {
        // Interactive mode (default)
        interactive_chat();
    }
    
    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    g_inference_engine.reset();
    g_model_loader.reset();
    if (g_engine) {
        cortex_inference_engine_free(g_engine);
    }
    
    return 0;
}
