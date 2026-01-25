/**
 * @file attention_kernels.hpp
 * @brief Multi-head attention kernels for transformer models
 */

#pragma once

#include <cstddef>
#include <vector>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief Multi-head self-attention forward pass
 * @param query Query tensor (batch, seq_len, hidden_dim)
 * @param key Key tensor (batch, seq_len, hidden_dim)
 * @param value Value tensor (batch, seq_len, hidden_dim)
 * @param output Output tensor (batch, seq_len, hidden_dim)
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param num_heads Number of attention heads
 * @param use_causal_mask Whether to apply causal masking (for autoregressive models)
 */
void multi_head_attention(const float* query, const float* key, const float* value,
                          float* output,
                          size_t batch_size, size_t seq_len, size_t hidden_dim,
                          size_t num_heads, bool use_causal_mask = false);

/**
 * @brief Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
 * @param query Query matrix (seq_len_q x d_k)
 * @param key Key matrix (seq_len_k x d_k)
 * @param value Value matrix (seq_len_k x d_v)
 * @param output Output matrix (seq_len_q x d_v)
 * @param seq_len_q Query sequence length
 * @param seq_len_k Key/Value sequence length
 * @param d_k Key dimension
 * @param d_v Value dimension
 * @param use_causal_mask Apply causal mask
 */
void scaled_dot_product_attention(const float* query, const float* key, const float* value,
                                  float* output,
                                  size_t seq_len_q, size_t seq_len_k,
                                  size_t d_k, size_t d_v,
                                  bool use_causal_mask = false);

/**
 * @brief Apply causal mask to attention scores (for autoregressive generation)
 * @param scores Attention score matrix (seq_len x seq_len)
 * @param seq_len Sequence length
 */
void apply_causal_mask(float* scores, size_t seq_len);

} // namespace Kernels
} // namespace CortexAICompression
