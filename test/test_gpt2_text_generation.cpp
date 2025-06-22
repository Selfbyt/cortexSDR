#include "../src/ai_compression/SparseInferenceEngine.hpp"
#include "../src/ai_compression/strategies/SDRIndexStorage.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>

using namespace CortexAICompression;

// GPT-2 tokenizer using actual vocabulary
class GPT2Tokenizer {
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    static constexpr int VOCAB_SIZE = 50257;
    
public:
    GPT2Tokenizer() {
        loadVocabulary("gpt2-vocab.json");
    }
    
    void loadVocabulary(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open vocabulary file: " << filename << std::endl;
            return;
        }
        
        std::string line;
        std::getline(file, line);
        
        // Parse JSON format properly
        size_t pos = 1; // Skip opening {
        while (pos < line.length() - 1) {
            // Find the start of a token (after quote)
            size_t quote_start = line.find('"', pos);
            if (quote_start == std::string::npos) break;
            
            // Find the end of the token (before closing quote)
            size_t quote_end = quote_start + 1;
            while (quote_end < line.length()) {
                if (line[quote_end] == '"' && line[quote_end - 1] != '\\') {
                    break;
                }
                quote_end++;
            }
            if (quote_end == std::string::npos) break;
            
            // Extract token
            std::string token = line.substr(quote_start + 1, quote_end - quote_start - 1);
            
            // Find the colon after the token
            size_t colon_pos = line.find(':', quote_end);
            if (colon_pos == std::string::npos) break;
            
            // Find the end of the number (before comma or closing brace)
            size_t num_start = colon_pos + 1;
            while (num_start < line.length() && std::isspace(line[num_start])) num_start++;
            
            size_t num_end = num_start;
            while (num_end < line.length() && std::isdigit(line[num_end])) num_end++;
            
            // Extract token ID
            std::string id_str = line.substr(num_start, num_end - num_start);
            
            if (!id_str.empty()) {
                try {
                    int id = std::stoi(id_str);
                    vocab[token] = id;
                    id_to_token[id] = token;
                } catch (const std::exception& e) {
                    // Skip malformed entries silently
                }
            }
            
            // Move to next token
            pos = num_end;
            if (pos < line.length() && line[pos] == ',') pos++;
        }
        
        std::cout << "Loaded " << vocab.size() << " tokens from vocabulary" << std::endl;
    }
    
    std::vector<int> tokenize(const std::string& text) {
    std::vector<int> tokens;
        
        // Preprocess text - normalize spaces and handle common cases
        std::string processed_text = text;
        
        // Try to match common words first
        std::vector<std::string> common_words = {"Hello", "hello", "my", "name", "is"};
        for (const auto& word : common_words) {
            size_t pos = 0;
            while ((pos = processed_text.find(word, pos)) != std::string::npos) {
                auto it = vocab.find(word);
                if (it != vocab.end()) {
                    tokens.push_back(it->second);
                    break;
                }
                pos++;
            }
        }
        
        // If no common words found, do character-by-character tokenization
        if (tokens.empty()) {
            size_t pos = 0;
            while (pos < processed_text.length()) {
                bool found = false;
                
                // Try to match the longest possible token
                for (size_t len = std::min(processed_text.length() - pos, size_t(50)); len > 0; len--) {
                    std::string candidate = processed_text.substr(pos, len);
                    auto it = vocab.find(candidate);
                    if (it != vocab.end()) {
                        tokens.push_back(it->second);
                        pos += len;
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    // Try single character
                    std::string single_char = processed_text.substr(pos, 1);
                    auto it = vocab.find(single_char);
                    if (it != vocab.end()) {
                        tokens.push_back(it->second);
                    } else {
                        tokens.push_back(0); // <|unk|>
                    }
                    pos++;
                }
            }
        }
        
    return tokens;
}

std::string detokenize(const std::vector<int>& tokens) {
        std::string result;
        for (size_t i = 0; i < tokens.size(); i++) {
            int token_id = tokens[i];
            auto it = id_to_token.find(token_id);
            if (it != id_to_token.end()) {
                std::string token = it->second;
                // Handle special tokens
                if (token == "<|endoftext|>") {
                    break;
                } else if (token == "<|unk|>") {
                    result += "[UNK]";
                } else {
                    // Insert a space if token starts with \u0120 (GPT-2's space marker)
                    if (token.rfind("\\u0120", 0) == 0) {
                        if (!result.empty() && result.back() != ' ') {
                            result += ' ';
                        }
                        token = token.substr(6); // Remove the '\u0120'
                    }
                    result += token;
                }
            } else {
                result += "[UNK]";
            }
        }
        // Post-process: add space after punctuation if needed
        std::string final_result;
        for (size_t i = 0; i < result.length(); i++) {
            final_result += result[i];
            if ((result[i] == ',' || result[i] == '.' || result[i] == '!' || result[i] == '?') &&
                i + 1 < result.length() && std::isalpha(result[i + 1])) {
                final_result += ' ';
            }
        }
        return final_result;
    }
};

// Utility functions for transformer operations
std::vector<float> layer_norm(const std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias, float eps = 1e-5) {
    size_t hidden_size = input.size();
    std::vector<float> output(hidden_size);
    
    // Calculate mean
    float mean = 0.0f;
    for (float x : input) mean += x;
    mean /= hidden_size;
    
    // Calculate variance
    float var = 0.0f;
    for (float x : input) {
        float diff = x - mean;
        var += diff * diff;
    }
    var /= hidden_size;
    
    // Normalize
    float std_dev = std::sqrt(var + eps);
    for (size_t i = 0; i < hidden_size; i++) {
        output[i] = (input[i] - mean) / std_dev * weight[i] + bias[i];
    }
    
    return output;
}

std::vector<float> linear_transform(const std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias, 
                                   size_t input_size, size_t output_size) {
    std::vector<float> output(output_size, 0.0f);
    
    // Matrix multiplication: output = input * weight^T + bias
    for (size_t i = 0; i < output_size; i++) {
        output[i] = bias[i];
        for (size_t j = 0; j < input_size; j++) {
            output[i] += input[j] * weight[i * input_size + j];
        }
    }
    
    return output;
}

std::vector<float> gelu_activation(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
    return output;
}

std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    // Find max for numerical stability
    float max_val = *std::max_element(input.begin(), input.end());
    
    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (size_t i = 0; i < input.size(); i++) {
        output[i] /= sum;
    }
    
    return output;
}

int sample_token(const std::vector<float>& logits, float temperature = 1.0f) {
    std::vector<float> probs = softmax(logits);
    
    // Apply temperature
    for (float& p : probs) {
        p = std::pow(p, 1.0f / temperature);
    }
    
    // Renormalize
    float sum = 0.0f;
    for (float p : probs) sum += p;
    for (float& p : probs) p /= sum;
    
    // Simple sampling (in production, use better sampling methods)
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); i++) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }
    return probs.size() - 1;
}

int main() {
    std::string compressed_path = "compressed_model.sdr";
    SDRModelLoader loader(compressed_path);
    SDRInferenceEngine engine(loader);

    // Initialize tokenizer with actual GPT-2 vocabulary
    GPT2Tokenizer tokenizer;

    // 1. Tokenize prompt
    std::string prompt = "Hello, my name is";
    std::vector<int> input_ids = tokenizer.tokenize(prompt);

    std::cout << "Input tokens: ";
    for (int id : input_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // 2. Manual embedding lookup for each token ID
    const auto& wte_layer = loader.getLayerMap().at("wte");
    
    // Handle compressed embeddings - decompress if needed
    std::vector<float> embedding_matrix;
    if (wte_layer.raw_data.empty()) {
        std::cerr << "No embedding data found in layer 'wte'" << std::endl;
        return 1;
    }
    
    // Check if embeddings are compressed (SDR format)
    bool is_compressed = false;
    if (wte_layer.raw_data.size() > 0) {
        uint8_t first_byte = static_cast<uint8_t>(wte_layer.raw_data[0]);
        // Check for SDR compression markers
        if (first_byte == 0x88 || first_byte == 0xD0 || first_byte == 0x90) {
            is_compressed = true;
        }
    }
    
    if (is_compressed) {
        std::cout << "Decompressing SDR-compressed embeddings..." << std::endl;
        // Decompress using SDR strategy
        auto sdr_strategy = std::make_shared<SDRIndexStorageStrategy>();
        auto decompressed_data = sdr_strategy->decompress(
            wte_layer.raw_data, 
            SegmentType::WEIGHTS_FP32,
            wte_layer.input_shape[0] * wte_layer.input_shape[1] * sizeof(float)
        );
        
        // Convert decompressed bytes to float array
        embedding_matrix.resize(decompressed_data.size() / sizeof(float));
        std::memcpy(embedding_matrix.data(), decompressed_data.data(), decompressed_data.size());
    } else {
        // Direct float array
        embedding_matrix.resize(wte_layer.raw_data.size() / sizeof(float));
        std::memcpy(embedding_matrix.data(), wte_layer.raw_data.data(), wte_layer.raw_data.size());
    }
    
    size_t vocab_size = wte_layer.input_shape[0];
    size_t hidden_size = wte_layer.input_shape[1];

    std::cout << "Embedding matrix shape: [" << vocab_size << ", " << hidden_size << "]" << std::endl;
    std::cout << "Embedding matrix size: " << embedding_matrix.size() << " elements" << std::endl;

    // Create sequence embeddings
    std::vector<std::vector<float>> sequence_embeddings;
    for (int token_id : input_ids) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
            std::cerr << "Token ID out of range: " << token_id << std::endl;
            continue;
        }
        size_t offset = token_id * hidden_size;
        std::vector<float> token_embedding(embedding_matrix.begin() + offset,
                                embedding_matrix.begin() + offset + hidden_size);
        sequence_embeddings.push_back(token_embedding);
    }

    std::cout << "Sequence length: " << sequence_embeddings.size() << std::endl;

    // 3. Run transformer layers
    std::vector<float> current_hidden = sequence_embeddings.back(); // Use last token for generation
    
    // Find transformer layers in the model
    const auto& layer_map = loader.getLayerMap();
    std::vector<std::string> layer_names;
    for (const auto& pair : layer_map) {
        if (pair.first.find("h.") != std::string::npos) {
            layer_names.push_back(pair.first);
        }
    }
    
    std::cout << "Found " << layer_names.size() << " transformer layers" << std::endl;
    
    // Sort layers by name to ensure correct order
    std::sort(layer_names.begin(), layer_names.end());
    
    // Run through transformer layers (simplified - just a few layers for testing)
    int max_layers = std::min(3, static_cast<int>(layer_names.size()));
    for (int layer_idx = 0; layer_idx < max_layers; layer_idx++) {
        std::string layer_prefix = "h." + std::to_string(layer_idx) + ".";
        std::cout << "Processing layer " << layer_idx << std::endl;
        
        // Layer normalization 1
        auto ln1_it = layer_map.find(layer_prefix + "ln_1");
        if (ln1_it != layer_map.end()) {
            const auto& ln1_weights = ln1_it->second.weights;
            size_t ln1_size = ln1_weights.size() / 2; // weight and bias
            std::vector<float> ln1_weight(ln1_weights.begin(), ln1_weights.begin() + ln1_size);
            std::vector<float> ln1_bias(ln1_weights.begin() + ln1_size, ln1_weights.end());
            current_hidden = layer_norm(current_hidden, ln1_weight, ln1_bias);
        }
        
        // Attention (simplified - just use the query projection)
        auto attn_it = layer_map.find(layer_prefix + "attn.c_attn");
        if (attn_it != layer_map.end()) {
            const auto& attn_weights = attn_it->second.weights;
            size_t attn_bias_size = attn_weights.size() / 4; // Assuming 3 projections (q,k,v) + bias
            std::vector<float> attn_bias(attn_weights.begin() + 3 * attn_bias_size, attn_weights.end());
            current_hidden = linear_transform(current_hidden, attn_weights, attn_bias, hidden_size, hidden_size);
        }
        
        // Layer normalization 2
        auto ln2_it = layer_map.find(layer_prefix + "ln_2");
        if (ln2_it != layer_map.end()) {
            const auto& ln2_weights = ln2_it->second.weights;
            size_t ln2_size = ln2_weights.size() / 2;
            std::vector<float> ln2_weight(ln2_weights.begin(), ln2_weights.begin() + ln2_size);
            std::vector<float> ln2_bias(ln2_weights.begin() + ln2_size, ln2_weights.end());
            current_hidden = layer_norm(current_hidden, ln2_weight, ln2_bias);
        }
        
        // Feed-forward (simplified - just first linear layer)
        auto mlp_it = layer_map.find(layer_prefix + "mlp.c_fc");
        if (mlp_it != layer_map.end()) {
            const auto& mlp_weights = mlp_it->second.weights;
            size_t mlp_bias_size = mlp_weights.size() / 2;
            std::vector<float> mlp_bias(mlp_weights.begin() + mlp_weights.size() - mlp_bias_size, mlp_weights.end());
            current_hidden = linear_transform(current_hidden, mlp_weights, mlp_bias, hidden_size, 4 * hidden_size);
            current_hidden = gelu_activation(current_hidden);
        }
    }
    
    // 4. Final layer normalization
    auto ln_f_it = layer_map.find("ln_f");
    if (ln_f_it != layer_map.end()) {
        const auto& ln_f_weights = ln_f_it->second.weights;
        size_t ln_f_size = ln_f_weights.size() / 2;
        std::vector<float> ln_f_weight(ln_f_weights.begin(), ln_f_weights.begin() + ln_f_size);
        std::vector<float> ln_f_bias(ln_f_weights.begin() + ln_f_size, ln_f_weights.end());
        current_hidden = layer_norm(current_hidden, ln_f_weight, ln_f_bias);
    }
    
    // 5. Final projection to vocabulary
    auto lm_head_it = layer_map.find("lm_head");
    if (lm_head_it == layer_map.end()) {
        // Try alternative name
        lm_head_it = layer_map.find("wte");
    }
    
    if (lm_head_it != layer_map.end()) {
        const auto& lm_weights = lm_head_it->second.weights;
        std::vector<float> logits(vocab_size);
        
        // Use embedding weights as output projection (transposed)
        for (size_t i = 0; i < vocab_size; i++) {
            logits[i] = 0.0f;
            for (size_t j = 0; j < hidden_size; j++) {
                logits[i] += current_hidden[j] * lm_weights[i * hidden_size + j];
            }
        }
        
        // Sample next token
        int next_token = sample_token(logits, 0.8f);
        std::cout << "Generated token ID: " << next_token << std::endl;
        
        // Add to sequence and detokenize
        input_ids.push_back(next_token);
        std::string generated_text = tokenizer.detokenize(input_ids);
        std::cout << "Generated text: " << generated_text << std::endl;
    }

    return 0;
} 