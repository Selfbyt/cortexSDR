/**
 * @file quantized_kernels.cpp
 * @brief Implementation of quantized inference kernels
 */

#include "quantized_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

namespace CortexAICompression {
namespace Kernels {

QuantizationParams quantize_tensor_int8(const float* input, int8_t* output, size_t size,
                                        QuantizationParams params) {
    if (params.scale == 0.0f) {
        // Compute quantization parameters
        float min_val = std::numeric_limits<float>::infinity();
        float max_val = -std::numeric_limits<float>::infinity();
        
        for (size_t i = 0; i < size; ++i) {
            min_val = std::min(min_val, input[i]);
            max_val = std::max(max_val, input[i]);
        }
        
        // Compute scale and zero point
        float range = max_val - min_val;
        if (range < 1e-8f) {
            range = 1.0f;
        }
        
        params.scale = range / 255.0f;
        params.zero_point = static_cast<int32_t>(-min_val / params.scale) - 128;
        params.qmin = -128;
        params.qmax = 127;
    }
    
    // Quantize
    for (size_t i = 0; i < size; ++i) {
        int32_t quantized = static_cast<int32_t>(std::round(input[i] / params.scale)) + params.zero_point;
        quantized = std::clamp(quantized, params.qmin, params.qmax);
        output[i] = static_cast<int8_t>(quantized);
    }
    
    return params;
}

void dequantize_tensor_int8(const int8_t* input, float* output, size_t size,
                            const QuantizationParams& params) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = params.scale * (static_cast<int32_t>(input[i]) - params.zero_point);
    }
}

void gemv_int8(const int8_t* A, const float* x, float* y,
               size_t M, size_t N,
               const QuantizationParams& A_params) {
    // Quantize input vector
    std::vector<int8_t> x_quant(N);
    QuantizationParams x_params = dynamic_quantize_int8(x, x_quant.data(), N);
    
    // Perform INT8 accumulation
    for (size_t i = 0; i < M; ++i) {
        int32_t acc = 0;
        for (size_t j = 0; j < N; ++j) {
            int32_t a_val = static_cast<int32_t>(A[i * N + j]) - A_params.zero_point;
            int32_t x_val = static_cast<int32_t>(x_quant[j]) - x_params.zero_point;
            acc += a_val * x_val;
        }
        
        // Dequantize result
        y[i] = A_params.scale * x_params.scale * static_cast<float>(acc);
    }
}

void gemm_int8(const int8_t* A, const int8_t* B, float* C,
               size_t M, size_t N, size_t K,
               const QuantizationParams& A_params,
               const QuantizationParams& B_params) {
    // INT8 matrix multiplication with accumulation
    float combined_scale = A_params.scale * B_params.scale;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int32_t acc = 0;
            
            for (size_t k = 0; k < K; ++k) {
                int32_t a_val = static_cast<int32_t>(A[i * K + k]) - A_params.zero_point;
                int32_t b_val = static_cast<int32_t>(B[k * N + j]) - B_params.zero_point;
                acc += a_val * b_val;
            }
            
            C[i * N + j] = combined_scale * static_cast<float>(acc);
        }
    }
}

std::vector<QuantizationParams> quantize_per_channel(const float* input, int8_t* output,
                                                      size_t out_channels, size_t in_features) {
    std::vector<QuantizationParams> params_vec(out_channels);
    
    for (size_t oc = 0; oc < out_channels; ++oc) {
        const float* channel_input = input + oc * in_features;
        int8_t* channel_output = output + oc * in_features;
        
        params_vec[oc] = quantize_tensor_int8(channel_input, channel_output, in_features);
    }
    
    return params_vec;
}

QuantizationParams dynamic_quantize_int8(const float* input, int8_t* output, size_t size) {
    return quantize_tensor_int8(input, output, size);
}

} // namespace Kernels
} // namespace CortexAICompression
