/**
 * @file simd_kernels.hpp
 * @brief SIMD-optimized kernels for activation functions and element-wise operations
 * 
 * Provides vectorized implementations using SSE/AVX intrinsics for maximum performance.
 */

#pragma once

#include <vector>
#include <cstddef>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief ReLU activation: y = max(0, x)
 * @param x Input array
 * @param y Output array (can be same as x for in-place)
 * @param size Array size
 */
void relu(const float* x, float* y, size_t size);

/**
 * @brief Leaky ReLU: y = x if x > 0 else alpha * x
 * @param x Input array
 * @param y Output array
 * @param size Array size
 * @param alpha Negative slope (default 0.01)
 */
void leaky_relu(const float* x, float* y, size_t size, float alpha = 0.01f);

/**
 * @brief GELU activation (Gaussian Error Linear Unit)
 * Uses tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * @param x Input array
 * @param y Output array
 * @param size Array size
 */
void gelu(const float* x, float* y, size_t size);

/**
 * @brief Swish/SiLU activation: y = x * sigmoid(x)
 * @param x Input array
 * @param y Output array
 * @param size Array size
 */
void swish(const float* x, float* y, size_t size);

/**
 * @brief Sigmoid activation: y = 1 / (1 + exp(-x))
 * @param x Input array
 * @param y Output array
 * @param size Array size
 */
void sigmoid(const float* x, float* y, size_t size);

/**
 * @brief Tanh activation: y = tanh(x)
 * @param x Input array
 * @param y Output array
 * @param size Array size
 */
void tanh_activation(const float* x, float* y, size_t size);

/**
 * @brief Softmax activation: y_i = exp(x_i) / sum(exp(x_j))
 * @param x Input array
 * @param y Output array
 * @param size Array size
 */
void softmax(const float* x, float* y, size_t size);

/**
 * @brief Layer normalization
 * @param x Input array
 * @param y Output array
 * @param gamma Scale parameter
 * @param beta Shift parameter
 * @param size Feature dimension
 * @param eps Epsilon for numerical stability
 */
void layer_norm(const float* x, float* y, const float* gamma, const float* beta,
                size_t size, float eps = 1e-5f);

/**
 * @brief Element-wise addition: y = a + b
 * @param a First input
 * @param b Second input
 * @param y Output
 * @param size Array size
 */
void add(const float* a, const float* b, float* y, size_t size);

/**
 * @brief Element-wise multiplication: y = a * b
 * @param a First input
 * @param b Second input
 * @param y Output
 * @param size Array size
 */
void multiply(const float* a, const float* b, float* y, size_t size);

/**
 * @brief Scaled addition: y = alpha * a + beta * b
 * @param a First input
 * @param b Second input
 * @param y Output
 * @param size Array size
 * @param alpha Scale for a
 * @param beta Scale for b
 */
void axpby(const float* a, const float* b, float* y, size_t size,
           float alpha = 1.0f, float beta = 1.0f);

/**
 * @brief Check SIMD capability
 * @return String describing SIMD level ("AVX2", "SSE4.1", "None")
 */
const char* get_simd_level();

} // namespace Kernels
} // namespace CortexAICompression
