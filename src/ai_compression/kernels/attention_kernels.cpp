/**
 * @file attention_kernels.cpp
 * @brief Implementation of multi-head attention kernels
 */

#include "attention_kernels.hpp"
#include "blas_kernels.hpp"
#include "simd_kernels.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace CortexAICompression {
namespace Kernels {

void apply_causal_mask(float* scores, size_t seq_len) {
    // Set upper triangle to -inf
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = i + 1; j < seq_len; ++j) {
            scores[i * seq_len + j] = -std::numeric_limits<float>::infinity();
        }
    }
}

void scaled_dot_product_attention(const float* query, const float* key, const float* value,
                                  float* output,
                                  size_t seq_len_q, size_t seq_len_k,
                                  size_t d_k, size_t d_v,
                                  bool use_causal_mask) {
    // Step 1: Compute attention scores: QK^T / sqrt(d_k)
    std::vector<float> scores(seq_len_q * seq_len_k);
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    
    // Matrix multiplication: Q @ K^T
    gemm(query, key, scores.data(),
         seq_len_q, seq_len_k, d_k,
         scale, 0.0f, false, true);  // transpose K
    
    // Step 2: Apply causal mask if needed
    if (use_causal_mask && seq_len_q == seq_len_k) {
        apply_causal_mask(scores.data(), seq_len_q);
    }
    
    // Step 3: Apply softmax row-wise
    for (size_t i = 0; i < seq_len_q; ++i) {
        softmax(scores.data() + i * seq_len_k,
               scores.data() + i * seq_len_k,
               seq_len_k);
    }
    
    // Step 4: Apply attention to values: Attention @ V
    gemm(scores.data(), value, output,
         seq_len_q, d_v, seq_len_k,
         1.0f, 0.0f, false, false);
}

void multi_head_attention(const float* query, const float* key, const float* value,
                          float* output,
                          size_t batch_size, size_t seq_len, size_t hidden_dim,
                          size_t num_heads, bool use_causal_mask) {
    // Head dimension
    size_t head_dim = hidden_dim / num_heads;
    
    if (hidden_dim % num_heads != 0) {
        // Invalid configuration
        return;
    }
    
    // Process each batch and head
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            // Extract Q, K, V for this head
            size_t head_offset = h * head_dim;
            size_t batch_offset = b * seq_len * hidden_dim;
            
            // Temporary buffers for this head's Q, K, V
            std::vector<float> q_head(seq_len * head_dim);
            std::vector<float> k_head(seq_len * head_dim);
            std::vector<float> v_head(seq_len * head_dim);
            std::vector<float> out_head(seq_len * head_dim);
            
            // Extract head slices from input tensors
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t src_idx = batch_offset + s * hidden_dim + head_offset + d;
                    size_t dst_idx = s * head_dim + d;
                    
                    q_head[dst_idx] = query[src_idx];
                    k_head[dst_idx] = key[src_idx];
                    v_head[dst_idx] = value[src_idx];
                }
            }
            
            // Compute attention for this head
            scaled_dot_product_attention(
                q_head.data(), k_head.data(), v_head.data(),
                out_head.data(),
                seq_len, seq_len, head_dim, head_dim,
                use_causal_mask
            );
            
            // Write back to output
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t src_idx = s * head_dim + d;
                    size_t dst_idx = batch_offset + s * hidden_dim + head_offset + d;
                    output[dst_idx] = out_head[src_idx];
                }
            }
        }
    }
}

} // namespace Kernels
} // namespace CortexAICompression
