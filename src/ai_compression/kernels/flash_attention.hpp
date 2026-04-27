/**
 * @file flash_attention.hpp
 * @brief Flash Attention - Memory-efficient attention with O(1) memory complexity
 * 
 * Based on "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 * Implements tiled computation to reduce memory bandwidth requirements
 */

#pragma once

#include <cstddef>
#include <cmath>
#include <vector>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief Flash Attention configuration
 */
struct FlashAttentionConfig {
    size_t block_size_q = 64;   // Query block size (tune for cache)
    size_t block_size_kv = 64;  // KV block size
    bool use_causal_mask = false;
    float softmax_scale = 0.0f;  // 0 = auto (1/sqrt(d_k))
};

/**
 * @brief Flash Attention forward pass with tiled computation
 * 
 * Memory complexity: O(N) instead of O(N^2) for standard attention
 * 
 * @param query Query matrix (batch_size x seq_len x hidden_dim)
 * @param key Key matrix (batch_size x seq_len x hidden_dim)
 * @param value Value matrix (batch_size x seq_len x hidden_dim)
 * @param output Output matrix (batch_size x seq_len x hidden_dim)
 * @param batch_size Number of samples
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param num_heads Number of attention heads
 * @param config Flash attention configuration
 */
void flash_attention_forward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    size_t batch_size,
    size_t seq_len,
    size_t hidden_dim,
    size_t num_heads,
    const FlashAttentionConfig& config = FlashAttentionConfig()
);

/**
 * @brief Flash Attention with KV cache (for autoregressive generation)
 * 
 * @param query Query for new token(s) (batch_size x new_tokens x hidden_dim)
 * @param key_cache Cached keys (batch_size x cached_len x hidden_dim)
 * @param value_cache Cached values (batch_size x cached_len x hidden_dim)
 * @param output Output matrix (batch_size x new_tokens x hidden_dim)
 * @param batch_size Number of samples
 * @param new_tokens Number of new query tokens
 * @param cached_len Number of cached KV tokens
 * @param hidden_dim Hidden dimension
 * @param num_heads Number of attention heads
 * @param config Flash attention configuration
 */
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
    const FlashAttentionConfig& config = FlashAttentionConfig()
);

/**
 * @brief Online softmax for incremental attention computation
 * Helper for Flash Attention implementation
 */
class OnlineSoftmax {
public:
    OnlineSoftmax() : m_max(-std::numeric_limits<float>::infinity()), m_sum(0.0f) {}
    
    void update(float value) {
        if (value > m_max) {
            float exp_diff = std::exp(m_max - value);
            m_sum = m_sum * exp_diff + 1.0f;
            m_max = value;
        } else {
            m_sum += std::exp(value - m_max);
        }
    }
    
    float normalize(float value) const {
        return std::exp(value - m_max) / m_sum;
    }
    
    float get_max() const { return m_max; }
    float get_sum() const { return m_sum; }
    
private:
    float m_max;
    float m_sum;
};

} // namespace Kernels
} // namespace CortexAICompression
