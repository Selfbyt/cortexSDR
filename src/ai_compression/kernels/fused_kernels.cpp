/**
 * @file fused_kernels.cpp
 * @brief Implementation of fused operation kernels
 */

#include "fused_kernels.hpp"
#include "blas_kernels.hpp"
#include "simd_kernels.hpp"
#include <cmath>
#include <vector>

namespace CortexAICompression {
namespace Kernels {

void linear_relu_fused(const float* input, const float* weights, const float* bias,
                       float* output, size_t batch_size, size_t input_size, size_t output_size) {
    // Perform linear transformation
    linear_forward(input, weights, bias, output, batch_size, input_size, output_size);
    
    // Apply ReLU in-place
    size_t total_size = batch_size * output_size;
    relu(output, output, total_size);
}

void linear_gelu_fused(const float* input, const float* weights, const float* bias,
                       float* output, size_t batch_size, size_t input_size, size_t output_size) {
    // Perform linear transformation
    linear_forward(input, weights, bias, output, batch_size, input_size, output_size);
    
    // Apply GELU in-place
    size_t total_size = batch_size * output_size;
    gelu(output, output, total_size);
}

void residual_layernorm_fused(const float* x, const float* residual,
                               const float* gamma, const float* beta,
                               float* output, size_t size, float eps) {
    // Add residual connection
    add(x, residual, output, size);
    
    // Apply layer normalization in-place
    layer_norm(output, output, gamma, beta, size, eps);
}

void batchnorm_relu_fused(const float* input, const float* gamma, const float* beta,
                          const float* mean, const float* var,
                          float* output, size_t size, float eps) {
    // Compute normalized value and apply ReLU simultaneously
    float inv_std = 1.0f / std::sqrt(*var + eps);
    
    for (size_t i = 0; i < size; ++i) {
        float normalized = (input[i] - *mean) * inv_std;
        float scaled = gamma[i] * normalized + beta[i];
        output[i] = std::max(0.0f, scaled);  // Fused ReLU
    }
}

} // namespace Kernels
} // namespace CortexAICompression
