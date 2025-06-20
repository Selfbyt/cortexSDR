#include "../src/ai_compression/SparseInferenceEngine.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

using namespace CortexAICompression;

// Minimal tokenizer stub (replace with real tokenizer for production)
std::vector<int> tokenize(const std::string& prompt) {
    std::vector<int> tokens;
    for (char c : prompt) tokens.push_back(static_cast<int>(c) % 100); // Dummy
    return tokens;
}

// Minimal detokenizer stub (replace with real vocab for production)
std::string detokenize(const std::vector<int>& tokens) {
    std::string out;
    for (int id : tokens) out += static_cast<char>(id + 32); // Dummy
    return out;
}

int main() {
    std::string compressed_path = "compressed_model.sdr";
    SDRModelLoader loader(compressed_path);
    SDRInferenceEngine engine(loader);

    // 1. Tokenize prompt
    std::string prompt = "Hello, my name is";
    std::vector<int> input_ids = tokenize(prompt);

    // 2. Manual embedding lookup for each token ID
    const auto& wte_layer = loader.getLayerMap().at("wte");
    const std::vector<float>& embedding_matrix = wte_layer.weights; // [vocab_size * hidden_size]
    size_t vocab_size = wte_layer.input_shape[0];
    size_t hidden_size = wte_layer.input_shape[1];

    std::vector<float> embedding_output;
    for (int token_id : input_ids) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
            std::cerr << "Token ID out of range: " << token_id << std::endl;
            continue;
        }
        size_t offset = token_id * hidden_size;
        embedding_output.insert(embedding_output.end(),
                                embedding_matrix.begin() + offset,
                                embedding_matrix.begin() + offset + hidden_size);
    }

    // Print output shape and preview
    std::cout << "Embedding output size: " << embedding_output.size() << std::endl;
    std::cout << "Embedding output preview: ";
    for (size_t i = 0; i < std::min<size_t>(10, embedding_output.size()); ++i) {
        std::cout << embedding_output[i] << " ";
    }
    std::cout << (embedding_output.size() > 10 ? "..." : "") << std::endl;

    // (Optional) Detokenize and print prompt
    std::string generated_text = detokenize(input_ids);
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Detokenized: " << generated_text << std::endl;

    return 0;
} 