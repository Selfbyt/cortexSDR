/**
 * @file quantized_kernels.hpp
 * @brief Quantized inference kernels for INT8/INT4 operations
 * 
 * Provides quantized GEMM and other operations for memory-efficient inference.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief Quantization parameters for a tensor
 */
struct QuantizationParams {
    float scale;
    int32_t zero_point;
    int32_t qmin;
    int32_t qmax;
};

/**
 * @brief Quantize FP32 tensor to INT8
 * @param input Float input tensor
 * @param output INT8 output tensor
 * @param size Tensor size
 * @param params Quantization parameters (computed if scale is 0)
 * @return Quantization parameters used
 */
QuantizationParams quantize_tensor_int8(const float* input, int8_t* output, size_t size,
                                        QuantizationParams params = {0.0f, 0, -128, 127});

/**
 * @brief Dequantize INT8 tensor to FP32
 * @param input INT8 input tensor
 * @param output Float output tensor
 * @param size Tensor size
 * @param params Quantization parameters
 */
void dequantize_tensor_int8(const int8_t* input, float* output, size_t size,
                            const QuantizationParams& params);

/**
 * @brief Quantized matrix-vector multiplication (INT8)
 * @param A Quantized matrix (M x N) in INT8
 * @param x Float input vector (N)
 * @param y Float output vector (M)
 * @param M Rows in A
 * @param N Columns in A
 * @param A_params Quantization params for A
 */
void gemv_int8(const int8_t* A, const float* x, float* y,
               size_t M, size_t N,
               const QuantizationParams& A_params);

/**
 * @brief Quantized matrix-matrix multiplication (INT8 x INT8 -> FP32)
 * @param A Quantized matrix (M x K)
 * @param B Quantized matrix (K x N)
 * @param C Float output matrix (M x N)
 * @param M Rows in A and C
 * @param N Columns in B and C
 * @param K Inner dimension
 * @param A_params Quantization params for A
 * @param B_params Quantization params for B
 */
void gemm_int8(const int8_t* A, const int8_t* B, float* C,
               size_t M, size_t N, size_t K,
               const QuantizationParams& A_params,
               const QuantizationParams& B_params);

/**
 * @brief Per-channel quantization (for weights)
 * @param input Float weights (out_channels x in_features)
 * @param output INT8 quantized weights
 * @param out_channels Number of output channels
 * @param in_features Number of input features
 * @return Vector of quantization params (one per channel)
 */
std::vector<QuantizationParams> quantize_per_channel(const float* input, int8_t* output,
                                                      size_t out_channels, size_t in_features);

/**
 * @brief Dynamic INT8 quantization (quantize activations on-the-fly)
 * @param input Float input
 * @param output INT8 output
 * @param size Tensor size
 * @return Quantization parameters
 */
QuantizationParams dynamic_quantize_int8(const float* input, int8_t* output, size_t size);

} // namespace Kernels
} // namespace CortexAICompression
