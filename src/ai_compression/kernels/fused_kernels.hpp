/**
 * @file fused_kernels.hpp
 * @brief Fused operation kernels for improved performance
 * 
 * Fused kernels combine multiple operations to reduce memory traffic and improve cache utilization.
 */

#pragma once

#include <cstddef>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief Fused linear + ReLU: y = max(0, W*x + b)
 * @param input Input vector
 * @param weights Weight matrix (row-major)
 * @param bias Bias vector
 * @param output Output vector
 * @param batch_size Number of samples
 * @param input_size Input dimension
 * @param output_size Output dimension
 */
void linear_relu_fused(const float* input, const float* weights, const float* bias,
                       float* output, size_t batch_size, size_t input_size, size_t output_size);

/**
 * @brief Fused linear + GELU: y = GELU(W*x + b)
 * @param input Input vector
 * @param weights Weight matrix
 * @param bias Bias vector
 * @param output Output vector
 * @param batch_size Number of samples
 * @param input_size Input dimension
 * @param output_size Output dimension
 */
void linear_gelu_fused(const float* input, const float* weights, const float* bias,
                       float* output, size_t batch_size, size_t input_size, size_t output_size);

/**
 * @brief Fused residual add + LayerNorm: y = LayerNorm(x + residual)
 * @param x Primary input
 * @param residual Residual connection input
 * @param gamma Scale parameter
 * @param beta Shift parameter
 * @param output Output vector
 * @param size Tensor size
 * @param eps Epsilon for numerical stability
 */
void residual_layernorm_fused(const float* x, const float* residual,
                               const float* gamma, const float* beta,
                               float* output, size_t size, float eps = 1e-5f);

/**
 * @brief Fused batch norm + ReLU
 * @param input Input tensor
 * @param gamma Scale parameter
 * @param beta Shift parameter
 * @param mean Running mean
 * @param var Running variance
 * @param output Output tensor
 * @param size Tensor size
 * @param eps Epsilon for numerical stability
 */
void batchnorm_relu_fused(const float* input, const float* gamma, const float* beta,
                          const float* mean, const float* var,
                          float* output, size_t size, float eps = 1e-5f);

} // namespace Kernels
} // namespace CortexAICompression
