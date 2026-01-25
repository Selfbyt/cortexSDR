/**
 * @file flash_attention.cpp
 * @brief Implementation of Flash Attention algorithm
 */

#include "flash_attention.hpp"
#include "blas_kernels.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>

namespace CortexAICompression {
namespace Kernels {

void flash_attention_forward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    size_t batch_size,
    size_t seq_len,
    size_t hidden_dim,
    size_t num_heads,
    const FlashAttentionConfig& config
) {
    size_t head_dim = hidden_dim / num_heads;
    
    float scale = config.softmax_scale;
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }
    
    size_t Bc = config.block_size_q;   // Query block size
    size_t Br = config.block_size_kv;  // KV block size
    
    // Process each batch and head independently
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            size_t head_offset = h * head_dim;
            size_t batch_offset = b * seq_len * hidden_dim;
            
            // Allocate temporary buffers for this head
            std::vector<float> O(seq_len * head_dim, 0.0f);  // Output accumulator
            std::vector<float> l(seq_len, 0.0f);              // Softmax denominators
            std::vector<float> m(seq_len, -std::numeric_limits<float>::infinity());  // Row maxima
            
            // Tile over sequence length for queries
            for (size_t q_start = 0; q_start < seq_len; q_start += Bc) {
                size_t q_end = std::min(q_start + Bc, seq_len);
                size_t q_block_size = q_end - q_start;
                
                // Load Q block
                std::vector<float> Q_block(q_block_size * head_dim);
                for (size_t i = 0; i < q_block_size; ++i) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t src_idx = batch_offset + (q_start + i) * hidden_dim + head_offset + d;
                        Q_block[i * head_dim + d] = query[src_idx];
                    }
                }
                
                // Tile over sequence length for keys/values
                for (size_t kv_start = 0; kv_start < seq_len; kv_start += Br) {
                    size_t kv_end = std::min(kv_start + Br, seq_len);
                    
                    // Apply causal mask if enabled
                    if (config.use_causal_mask && kv_start > q_end) {
                        continue;  // Skip future tokens
                    }
                    
                    size_t kv_block_size = kv_end - kv_start;
                    
                    // Load K block
                    std::vector<float> K_block(kv_block_size * head_dim);
                    for (size_t i = 0; i < kv_block_size; ++i) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t src_idx = batch_offset + (kv_start + i) * hidden_dim + head_offset + d;
                            K_block[i * head_dim + d] = key[src_idx];
                        }
                    }
                    
                    // Load V block
                    std::vector<float> V_block(kv_block_size * head_dim);
                    for (size_t i = 0; i < kv_block_size; ++i) {
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t src_idx = batch_offset + (kv_start + i) * hidden_dim + head_offset + d;
                            V_block[i * head_dim + d] = value[src_idx];
                        }
                    }
                    
                    // Compute attention scores: S = Q @ K^T * scale
                    std::vector<float> S(q_block_size * kv_block_size);
                    gemm(Q_block.data(), K_block.data(), S.data(),
                         q_block_size, kv_block_size, head_dim,
                         scale, 0.0f, false, true);
                    
                    // Apply causal mask within block if needed
                    if (config.use_causal_mask) {
                        for (size_t i = 0; i < q_block_size; ++i) {
                            for (size_t j = 0; j < kv_block_size; ++j) {
                                size_t q_pos = q_start + i;
                                size_t kv_pos = kv_start + j;
                                if (kv_pos > q_pos) {
                                    S[i * kv_block_size + j] = -std::numeric_limits<float>::infinity();
                                }
                            }
                        }
                    }
                    
                    // Online softmax update for each query token
                    for (size_t i = 0; i < q_block_size; ++i) {
                        size_t q_idx = q_start + i;
                        
                        // Find max score in this block
                        float m_new = m[q_idx];
                        for (size_t j = 0; j < kv_block_size; ++j) {
                            m_new = std::max(m_new, S[i * kv_block_size + j]);
                        }
                        
                        // Compute exponentials and sum
                        float l_new = 0.0f;
                        std::vector<float> P(kv_block_size);
                        for (size_t j = 0; j < kv_block_size; ++j) {
                            P[j] = std::exp(S[i * kv_block_size + j] - m_new);
                            l_new += P[j];
                        }
                        
                        // Rescale previous output and accumulator
                        float scale_old = std::exp(m[q_idx] - m_new);
                        float scale_new = l_new / (scale_old * l[q_idx] + l_new);
                        
                        if (l[q_idx] > 0.0f) {
                            for (size_t d = 0; d < head_dim; ++d) {
                                O[q_idx * head_dim + d] *= (scale_old * l[q_idx]) / (scale_old * l[q_idx] + l_new);
                            }
                        }
                        
                        // Accumulate new attention output
                        for (size_t j = 0; j < kv_block_size; ++j) {
                            float p_scaled = P[j] / (scale_old * l[q_idx] + l_new);
                            for (size_t d = 0; d < head_dim; ++d) {
                                O[q_idx * head_dim + d] += p_scaled * V_block[j * head_dim + d];
                            }
                        }
                        
                        // Update running statistics
                        l[q_idx] = scale_old * l[q_idx] + l_new;
                        m[q_idx] = m_new;
                    }
                }
            }
            
            // Write output for this head
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t dst_idx = batch_offset + i * hidden_dim + head_offset + d;
                    output[dst_idx] = O[i * head_dim + d];
                }
            }
        }
    }
}

void flash_attention_with_kv_cache(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    size_t batch_size,
    size_t new_tokens,
    size_t cached_len,
    size_t hidden_dim,
    size_t num_heads,
    const FlashAttentionConfig& config
) {
    size_t head_dim = hidden_dim / num_heads;
    size_t total_len = cached_len + new_tokens;
    
    float scale = config.softmax_scale;
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }
    
    // For KV cache, queries only attend to cached keys/values
    // This is optimized for autoregressive generation where new_tokens is typically 1
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            size_t head_offset = h * head_dim;
            
            for (size_t t = 0; t < new_tokens; ++t) {
                // Extract query for this token
                std::vector<float> q(head_dim);
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t src_idx = b * new_tokens * hidden_dim + t * hidden_dim + head_offset + d;
                    q[d] = query[src_idx];
                }
                
                // Compute attention scores with all cached KV pairs
                std::vector<float> scores(cached_len);
                for (size_t i = 0; i < cached_len; ++i) {
                    float score = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t k_idx = b * cached_len * hidden_dim + i * hidden_dim + head_offset + d;
                        score += q[d] * key_cache[k_idx];
                    }
                    scores[i] = score * scale;
                }
                
                // Apply causal mask if needed (though typically not needed for KV cache)
                if (config.use_causal_mask) {
                    size_t current_pos = cached_len + t;
                    for (size_t i = current_pos; i < cached_len; ++i) {
                        scores[i] = -std::numeric_limits<float>::infinity();
                    }
                }
                
                // Softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (size_t i = 0; i < cached_len; ++i) {
                    scores[i] = std::exp(scores[i] - max_score);
                    sum_exp += scores[i];
                }
                for (size_t i = 0; i < cached_len; ++i) {
                    scores[i] /= sum_exp;
                }
                
                // Apply attention to values
                std::vector<float> out(head_dim, 0.0f);
                for (size_t i = 0; i < cached_len; ++i) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        size_t v_idx = b * cached_len * hidden_dim + i * hidden_dim + head_offset + d;
                        out[d] += scores[i] * value_cache[v_idx];
                    }
                }
                
                // Write output
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t dst_idx = b * new_tokens * hidden_dim + t * hidden_dim + head_offset + d;
                    output[dst_idx] = out[d];
                }
            }
        }
    }
}

} // namespace Kernels
} // namespace CortexAICompression
