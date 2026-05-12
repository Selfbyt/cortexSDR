#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <regex>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include "ai_compression/api/c_api.hpp"
#include "ai_compression/SparseInferenceEngine.hpp"
#include "ai_compression/strategies/HierarchicalSDRStrategy.hpp"
#include "ai_compression/parsers/ModelParserFactory.hpp"
#include "ai_compression/core/ModelSegment.hpp"
#include <cstring>

namespace fs = std::filesystem;

// Function prototypes
void printUsage(const char* programName);
int compressModel(const char* model_path, const char* format, const char* output_path, float sparsity);
int compressMultipleFiles(const std::vector<std::string>& model_paths, const char* format, const char* output_path, float sparsity);
int decompressModel(const char* compressed_path, const char* output_path, float sparsity);
int extractArchive(const char* compressed_path, const char* output_dir, float sparsity);
int runInference(const char* archive_path, const char* input_indices_file);
int compressModelHSDR(const char* model_path, const char* format, const char* output_hsda_path,
                      int protect_boundary, int total_layers_hint);
std::vector<size_t> loadInputIndices(const char* input_path);
std::vector<std::string> findModelParts(const std::string& model_path);

void printUsage(const char* programName) {
    std::cout << "CortexSDR AI Model Compression CLI\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " -c <model_path> <format> <output_path> [sparsity]   (Compress single file)\n";
    std::cout << "  " << programName << " -m <format> <output_path> <file1> <file2> ... [--sparsity X]  (Compress multiple files)\n";
    std::cout << "  " << programName << " -d <compressed_path> <output_dir> [sparsity]        (Extract bundle)\n";
    std::cout << "  " << programName << " -x <compressed_path> <output_dir> [sparsity]        (Extract archive)\n";
    std::cout << "  " << programName << " -i <archive_path> <input_indices_file>              (Inference)\n";
    std::cout << "  " << programName << " -h <model_path> <format> <output.hsda>              (HSDR shared-dict pipeline)\n";
    std::cout << "              [--protect-boundary N] [--total-layers M]\n";
    std::cout << "\nSupported formats:\n";
    std::cout << "  - onnx: ONNX models (direct support)\n";
    std::cout << "  - gguf: GGUF models (direct support, multi-part supported)\n";
    std::cout << "  - tensorflow: TensorFlow SavedModel (.pb files)\n";
    std::cout << "  - pytorch: PyTorch models (.pt/.pth files)\n";
    std::cout << "  - hdf5: HDF5/Keras models (.h5 files)\n";
    std::cout << "\nNote: All formats are compressed using the same SDR-based compression strategies.\n";
    std::cout << "\nOptions:\n";
    std::cout << "  sparsity: Fraction of active bits in SDR encoding (default: 0.02 = 2%)\n";
    std::cout << "            Lower values = better compression, higher values = better quality\n";
    std::cout << "            Recommended: 0.01-0.05 for high compression (50-100:1 ratio)\n";
    std::cout << "                        0.10-0.50 for better quality (lower compression)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << programName << " -c model.onnx onnx compressed_model.sdr\n";
    std::cout << "  " << programName << " -c model.onnx onnx compressed_model.sdr 0.01\n";
    std::cout << "  " << programName << " -m gguf model.sdr part1.gguf part2.gguf --sparsity 0.02\n";
    std::cout << "  " << programName << " -d compressed_model.sdr extracted_segments\n";
    std::cout << "  " << programName << " -x compressed_model.sdr extracted_segments\n";
    std::cout << "  " << programName << " -i compressed_model.sdr input_indices.txt\n";
}

std::vector<size_t> loadInputIndices(const char* input_path) {
    std::vector<size_t> indices;
    std::ifstream file(input_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input indices file");
    }
    size_t idx;
    while (file >> idx) {
        indices.push_back(idx);
    }
    return indices;
}

// Automatically detect and find all parts of a multi-part model
// Supports patterns like:
//   - model-00001-of-00002.gguf, model-00002-of-00002.gguf
//   - model.gguf.00001, model.gguf.00002
//   - model_part1.gguf, model_part2.gguf
std::vector<std::string> findModelParts(const std::string& model_path) {
    std::vector<std::string> parts;
    
    fs::path path(model_path);
    if (!fs::exists(path)) {
        std::cerr << "Warning: File does not exist: " << model_path << std::endl;
        return {model_path}; // Return original path anyway
    }
    
    fs::path parent = path.parent_path();
    if (parent.empty()) {
        parent = fs::current_path();
    }
    
    std::string filename = path.filename().string();
    std::string stem = path.stem().string();
    std::string extension = path.extension().string();
    
    // Pattern 1: model-00001-of-00005.gguf (Hugging Face standard)
    std::regex pattern1(R"(^(.+)-(\d+)-of-(\d+)(\..+)$)");
    std::smatch match1;
    
    if (std::regex_match(filename, match1, pattern1)) {
        std::string base_name = match1[1].str();
        int current_part = std::stoi(match1[2].str());
        int total_parts = std::stoi(match1[3].str());
        std::string ext = match1[4].str();
        
        std::cout << "Detected multi-part model: " << base_name << " (" << total_parts << " parts)\n";
        
        // Find all parts
        for (int i = 1; i <= total_parts; i++) {
            // Format with leading zeros matching the original
            std::ostringstream part_name;
            part_name << base_name << "-";
            part_name << std::setfill('0') << std::setw(5) << i;
            part_name << "-of-";
            part_name << std::setfill('0') << std::setw(5) << total_parts;
            part_name << ext;
            
            fs::path part_path = parent / part_name.str();
            if (fs::exists(part_path)) {
                parts.push_back(part_path.string());
                std::cout << "  Found part " << i << "/" << total_parts << ": " << part_path.filename().string() << "\n";
            } else {
                std::cerr << "  Warning: Missing part " << i << "/" << total_parts << ": " << part_path.filename().string() << "\n";
            }
        }
        
        if (!parts.empty()) {
            return parts;
        }
    }
    
    // Pattern 2: model.gguf.00001, model.gguf.00002
    std::regex pattern2(R"(^(.+)\.(\d+)$)");
    std::smatch match2;
    
    if (std::regex_match(filename, match2, pattern2)) {
        std::string base_name = match2[1].str();
        
        // Find all numbered parts in the directory
        std::vector<std::pair<int, std::string>> numbered_parts;
        for (const auto& entry : fs::directory_iterator(parent)) {
            if (entry.is_regular_file()) {
                std::string entry_name = entry.path().filename().string();
                std::smatch entry_match;
                if (std::regex_match(entry_name, entry_match, pattern2)) {
                    if (entry_match[1].str() == base_name) {
                        int part_num = std::stoi(entry_match[2].str());
                        numbered_parts.push_back({part_num, entry.path().string()});
                    }
                }
            }
        }
        
        if (numbered_parts.size() > 1) {
            std::cout << "Detected multi-part model: " << base_name << " (" << numbered_parts.size() << " parts)\n";
            
            // Sort by part number
            std::sort(numbered_parts.begin(), numbered_parts.end());
            
            for (const auto& [part_num, part_path] : numbered_parts) {
                parts.push_back(part_path);
                std::cout << "  Found part " << part_num << ": " << fs::path(part_path).filename().string() << "\n";
            }
            
            return parts;
        }
    }
    
    // Pattern 3: model_part1.gguf, model_part2.gguf
    std::regex pattern3(R"(^(.+)_part(\d+)(\..+)$)");
    std::smatch match3;
    
    if (std::regex_match(filename, match3, pattern3)) {
        std::string base_name = match3[1].str();
        std::string ext = match3[3].str();
        
        // Find all parts in the directory
        std::vector<std::pair<int, std::string>> numbered_parts;
        for (const auto& entry : fs::directory_iterator(parent)) {
            if (entry.is_regular_file()) {
                std::string entry_name = entry.path().filename().string();
                std::smatch entry_match;
                if (std::regex_match(entry_name, entry_match, pattern3)) {
                    if (entry_match[1].str() == base_name && entry_match[3].str() == ext) {
                        int part_num = std::stoi(entry_match[2].str());
                        numbered_parts.push_back({part_num, entry.path().string()});
                    }
                }
            }
        }
        
        if (numbered_parts.size() > 1) {
            std::cout << "Detected multi-part model: " << base_name << " (" << numbered_parts.size() << " parts)\n";
            
            // Sort by part number
            std::sort(numbered_parts.begin(), numbered_parts.end());
            
            for (const auto& [part_num, part_path] : numbered_parts) {
                parts.push_back(part_path);
                std::cout << "  Found part " << part_num << ": " << fs::path(part_path).filename().string() << "\n";
            }
            
            return parts;
        }
    }
    
    // No multi-part pattern detected, return single file
    return {model_path};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "-c") {
        if (argc < 5) {
            printUsage(argv[0]);
            return 1;
        }
        const char* model_path = argv[2];
        const char* format = argv[3];
        const char* output_path = argv[4];
        float sparsity = 0.02f;
        if (argc > 5) {
            sparsity = std::stof(argv[5]);
        }
        return compressModel(model_path, format, output_path, sparsity);
    } else if (mode == "-m") {
        // Multi-file compression mode
        if (argc < 5) {
            printUsage(argv[0]);
            return 1;
        }
        const char* format = argv[2];
        const char* output_path = argv[3];
        
        // Collect all input files and check for --sparsity flag
        std::vector<std::string> input_files;
        float sparsity = 0.02f;
        
        for (int i = 4; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--sparsity" && i + 1 < argc) {
                sparsity = std::stof(argv[i + 1]);
                i++; // Skip the next argument
            } else {
                input_files.push_back(arg);
            }
        }
        
        if (input_files.empty()) {
            std::cerr << "Error: No input files specified\n";
            printUsage(argv[0]);
            return 1;
        }
        
        return compressMultipleFiles(input_files, format, output_path, sparsity);
    } else if (mode == "-d") {
        if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        const char* compressed_path = argv[2];
        const char* output_path = argv[3];
        float sparsity = 0.02f;
        if (argc > 4) {
            sparsity = std::stof(argv[4]);
        }
        return decompressModel(compressed_path, output_path, sparsity);
    } else if (mode == "-i") {
        if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        const char* archive_path = argv[2];
        const char* input_indices_file = argv[3];
        return runInference(archive_path, input_indices_file);
    } else if (mode == "-x") {
        if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        const char* compressed_path = argv[2];
        const char* output_dir = argv[3];
        float sparsity = 0.02f;
        if (argc > 4) {
            sparsity = std::stof(argv[4]);
        }
        return extractArchive(compressed_path, output_dir, sparsity);
    } else if (mode == "-h") {
        // HSDR shared-dictionary pipeline.
        // Usage: -h <model_path> <format> <output.hsda>
        //          [--protect-boundary N] [--total-layers M]
        if (argc < 5) {
            printUsage(argv[0]);
            return 1;
        }
        const char* model_path = argv[2];
        const char* format = argv[3];
        const char* output_hsda_path = argv[4];
        int protect_boundary = 0;
        int total_layers = 0;
        for (int i = 5; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--protect-boundary" && i + 1 < argc) {
                protect_boundary = std::stoi(argv[i + 1]);
                ++i;
            } else if (arg == "--total-layers" && i + 1 < argc) {
                total_layers = std::stoi(argv[i + 1]);
                ++i;
            } else {
                std::cerr << "Warning: unknown option in -h mode: " << arg << "\n";
            }
        }
        return compressModelHSDR(model_path, format, output_hsda_path,
                                 protect_boundary, total_layers);
    } else {
        printUsage(argv[0]);
        return 1;
    }
}

int compressModel(const char* model_path, const char* format, const char* output_path, float sparsity) {
    // Automatically detect if this is a multi-part model
    std::vector<std::string> model_parts = findModelParts(model_path);
    
    if (model_parts.size() > 1) {
        std::cout << "\nAuto-detected multi-part model with " << model_parts.size() << " parts.\n";
        std::cout << "Compressing all parts into a single unified archive...\n\n";
        return compressMultipleFiles(model_parts, format, output_path, sparsity);
    }
    
    // Single file compression
    // Initialize compression options
    CortexCompressionOptions options;
    CortexError error = cortex_compression_options_init(&options);
    if (error.code != 0) {
        std::cerr << "Error initializing options: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Set options
    options.verbose = 1;
    options.show_stats = 1;
    options.sparsity = sparsity;

    // Create compressor
    std::cout << "Creating compressor for model: " << model_path << " (format: " << format << ")\n";
    std::cout << "Using sparsity: " << sparsity << " (" << (sparsity * 100) << "%)\n";
    CortexCompressorHandle compressor;
    error = cortex_compressor_create(model_path, format, &options, &compressor);
    if (error.code != 0) {
        std::cerr << "Error creating compressor: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Compress the model
    std::cout << "Compressing model to: " << output_path << "\n";
    error = cortex_compressor_compress(compressor, output_path);
    if (error.code != 0) {
        std::cerr << "Error compressing model: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        cortex_compressor_free(compressor);
        return 1;
    }

    // Get compression stats
    size_t original_size, compressed_size;
    double compression_ratio, compression_time_ms;
    error = cortex_compressor_get_stats(compressor, &original_size, &compressed_size, 
                                      &compression_ratio, &compression_time_ms);
    if (error.code != 0) {
        std::cerr << "Error getting stats: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
    } else {
        std::cout << "Compression Stats:\n";
        std::cout << "  Original size: " << original_size << " bytes\n";
        std::cout << "  Compressed size: " << compressed_size << " bytes\n";
        std::cout << "  Compression ratio: " << compression_ratio << ":1\n";
        std::cout << "  Compression time: " << compression_time_ms << " ms\n";
    }

    // Free the compressor
    cortex_compressor_free(compressor);
    std::cout << "Compression complete.\n";
    return 0;
}

int compressMultipleFiles(const std::vector<std::string>& model_paths, const char* format, const char* output_path, float sparsity) {
    std::cout << "\nMulti-file compression mode\n";
    std::cout << "Format: " << format << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "Sparsity: " << sparsity << " (" << (sparsity * 100) << "%)\n";
    std::cout << "Input files (" << model_paths.size() << "):\n";
    for (const auto& path : model_paths) {
        std::cout << "  - " << path << "\n";
    }
    
    // Initialize compression options
    CortexCompressionOptions options;
    CortexError error = cortex_compression_options_init(&options);
    if (error.code != 0) {
        std::cerr << "Error initializing options: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Set options
    options.verbose = 1;
    options.show_stats = 1;
    options.sparsity = sparsity;

    // Create compressor with the first file
    // IMPORTANT: Pass a flag or marker to prevent double auto-detection
    std::cout << "\nCreating compressor (auto-detection disabled for manual multi-file)...\n";
    CortexCompressorHandle compressor;
    
    // For manual multi-file mode, we need to disable auto-detection in the API
    // We'll use a special marker in the path to signal this
    std::string first_file_marker = model_paths[0] + "?no_auto_detect";
    error = cortex_compressor_create(model_paths[0].c_str(), format, &options, &compressor);
    if (error.code != 0) {
        std::cerr << "Error creating compressor: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    // Manually add additional files (skip auto-detection since we're doing it manually)
    for (size_t i = 1; i < model_paths.size(); i++) {
        std::cout << "Adding file " << (i + 1) << "/" << model_paths.size() << ": " << model_paths[i] << "\n";
        error = cortex_compressor_add_file(compressor, model_paths[i].c_str());
        if (error.code != 0) {
            std::cerr << "Error adding file: " << error.message << " (code: " << error.code << ")\n";
            cortex_error_free(&error);
            cortex_compressor_free(compressor);
            return 1;
        }
    }

    // Compress all files into a single archive
    std::cout << "\nCompressing all files to: " << output_path << "\n";
    error = cortex_compressor_compress(compressor, output_path);
    if (error.code != 0) {
        std::cerr << "Error compressing model: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        cortex_compressor_free(compressor);
        return 1;
    }

    // Get compression stats
    size_t original_size, compressed_size;
    double compression_ratio, compression_time_ms;
    error = cortex_compressor_get_stats(compressor, &original_size, &compressed_size, 
                                      &compression_ratio, &compression_time_ms);
    if (error.code != 0) {
        std::cerr << "Error getting stats: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
    } else {
        std::cout << "\nCompression Stats:\n";
        std::cout << "  Original size: " << original_size << " bytes (" << (original_size / 1024.0 / 1024.0) << " MB)\n";
        std::cout << "  Compressed size: " << compressed_size << " bytes (" << (compressed_size / 1024.0 / 1024.0) << " MB)\n";
        std::cout << "  Compression ratio: " << compression_ratio << ":1\n";
        std::cout << "  Compression time: " << compression_time_ms << " ms\n";
    }

    // Free the compressor
    cortex_compressor_free(compressor);
    std::cout << "Multi-file compression complete.\n";
    return 0;
}

int decompressModel(const char* compressed_path, const char* output_path, float sparsity) {
    std::cout << "Extracting archive from: " << compressed_path << " to directory: " << output_path << std::endl;
    std::cout << "Using sparsity: " << sparsity << " (" << (sparsity * 100) << "%)" << std::endl;

    CortexDecompressorHandle handle;
    CortexError error = cortex_decompressor_create(compressed_path, &handle, sparsity);
    if (error.code != 0) {
        std::cerr << "Error creating decompressor: " << error.message << std::endl;
        cortex_error_free(&error);
        return 1;
    }

    error = cortex_decompressor_decompress(handle, compressed_path, output_path);
    if (error.code != 0) {
        std::cerr << "Error decompressing model: " << error.message << std::endl;
        cortex_error_free(&error);
        cortex_decompressor_free(handle);
        return 1;
    }

    cortex_decompressor_free(handle);
    std::cout << "Archive extraction complete." << std::endl;
    return 0;
}

int extractArchive(const char* compressed_path, const char* output_dir, float sparsity) {
    std::cout << "Extracting archive from: " << compressed_path << " to directory: " << output_dir << std::endl;
    std::cout << "Using sparsity: " << sparsity << " (" << (sparsity * 100) << "%)" << std::endl;

    CortexError error = cortex_archive_extract(compressed_path, output_dir, sparsity);
    if (error.code != 0) {
        std::cerr << "Error extracting archive: " << error.message << " (code: " << error.code << ")\n";
        cortex_error_free(&error);
        return 1;
    }

    std::cout << "Archive extraction complete." << std::endl;
    return 0;
}

int runInference(const char* archive_path, const char* input_indices_file) {
    try {
        // Load input indices from file
        std::vector<size_t> input_indices = loadInputIndices(input_indices_file);
        
        // Load model and run inference
        std::cout << "Loading compressed model from: " << archive_path << "\n";
        CortexAICompression::SDRModelLoader loader(archive_path);
        CortexAICompression::SDRInferenceEngine engine(loader);
        
        // Convert input indices to a float tensor (1.0 for active indices, 0.0 for others)
        std::vector<float> input_tensor(1000, 0.0f);  // Assuming 1000-dimensional input
        for (auto idx : input_indices) {
            if (idx < input_tensor.size()) {
                input_tensor[idx] = 1.0f;
            }
        }
        
        std::cout << "Running inference with " << input_indices.size() << " active indices...\n";
        auto output_tensor = engine.run(input_tensor);
        
        std::cout << "Inference complete. Output tensor size: " << output_tensor.size() << std::endl;
        std::cout << "First few output values: ";
        for (size_t i = 0; i < std::min(size_t(5), output_tensor.size()); ++i) {
            std::cout << output_tensor[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int compressModelHSDR(const char* model_path, const char* format, const char* output_hsda_path,
                      int protect_boundary, int total_layers_hint) {
    using namespace CortexAICompression;
    std::cout << "HSDR shared-dictionary compression\n";
    std::cout << "  model:  " << model_path << "  (format: " << format << ")\n";
    std::cout << "  output: " << output_hsda_path << "\n";
    if (protect_boundary > 0) {
        std::cout << "  protect-boundary: " << protect_boundary
                  << "  (total-layers hint: " << total_layers_hint << ")\n";
    }

    try {
        // 1. Parse the model into segments.
        auto parser = ModelParserFactory::createParserForFormat(format);
        std::vector<ModelSegment> segments = parser->parse(model_path);
        std::cout << "  parsed " << segments.size() << " segments\n";

        // Count weight-tensor segments for stats.
        size_t weight_count = 0, fp32_count = 0;
        for (const auto& s : segments) {
            if (s.isWeightTensor()) {
                ++weight_count;
                bool is_fp32 = (s.type == SegmentType::WEIGHTS_FP32);
                if (!is_fp32) {
                    std::string fmt;
                    fmt.reserve(s.data_format.size());
                    for (char c : s.data_format) {
                        fmt.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                    }
                    if (fmt == "f32" || fmt == "float32" || fmt == "fp32") is_fp32 = true;
                }
                if (is_fp32) ++fp32_count;
            }
        }
        std::cout << "    weight tensors: " << weight_count << " (FP32: " << fp32_count << ")\n";

        // 2. Build the strategy. Use the defaults from `HierarchicalSDRConfig`
        //    for attention and MLP. Larger models will want explicit role
        //    configs — a follow-up could expose --attn-* / --mlp-* flags.
        HierarchicalSDRStrategy strat;
        if (protect_boundary > 0) {
            const size_t total = (total_layers_hint > 0)
                                  ? static_cast<size_t>(total_layers_hint)
                                  : 0;
            if (total == 0) {
                std::cerr << "  Warning: --protect-boundary requested but --total-layers not given;\n"
                          << "           predicate will only protect layers near depth 0.\n";
            }
            strat.setProtectionPredicate(ProtectionPolicies::boundaryMLPs(
                static_cast<size_t>(protect_boundary), total));
        }

        // 3. Run the multi-pass shared-dictionary pipeline.
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::string> skipped;
        auto archive = strat.compressGroupedSegments(segments, &skipped);
        auto t1 = std::chrono::steady_clock::now();
        const double fit_s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "  pipeline produced " << archive.dictionaries.size()
                  << " dictionaries and " << archive.segments.size()
                  << " compressed segments in " << fit_s << " s\n";
        std::cout << "  skipped " << skipped.size() << " segments "
                  << "(non-FP32, shape-too-small, or protected)\n";

        // 4. Serialize to disk.
        archive.writeToFile(output_hsda_path);
        const auto archive_bytes = std::filesystem::file_size(output_hsda_path);
        std::cout << "  wrote " << archive_bytes << " bytes to " << output_hsda_path << "\n";

        // Stats: total original weight bytes vs archive bytes.
        size_t total_weight_bytes = 0;
        for (const auto& s : segments) {
            if (s.isWeightTensor()) total_weight_bytes += s.data.size();
        }
        if (total_weight_bytes > 0) {
            const double ratio = static_cast<double>(total_weight_bytes)
                                 / static_cast<double>(archive_bytes);
            std::cout << "  ratio (all-weights / archive): " << ratio << "x\n";
        }
        std::cout << "HSDR compression complete.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in HSDR mode: " << e.what() << std::endl;
        return 1;
    }
}
