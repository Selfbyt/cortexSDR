/**
 * @file quantized_kernels_int4.hpp
 * @brief INT4 quantization kernels for ultra-low precision inference
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief INT4 quantization parameters (per group or per channel)
 */
struct QuantizationParamsINT4 {
    float scale;
    int8_t zero_point;  // -8 to 7 for INT4
    
    QuantizationParamsINT4() : scale(1.0f), zero_point(0) {}
    QuantizationParamsINT4(float s, int8_t z) : scale(s), zero_point(z) {}
};

/**
 * @brief Quantize FP32 tensor to INT4 (packed format: 2 values per byte)
 * @param input Float input tensor
 * @param output Packed INT4 output (size / 2 bytes)
 * @param size Tensor size (must be even)
 * @param group_size Quantization group size (0 = per-tensor, >0 = grouped)
 * @return Vector of quantization parameters (one per group/tensor)
 */
std::vector<QuantizationParamsINT4> quantize_tensor_int4(
    const float* input, 
    uint8_t* output, 
    size_t size,
    size_t group_size = 128
);

/**
 * @brief Dequantize INT4 tensor to FP32
 * @param input Packed INT4 input
 * @param output Float output tensor
 * @param size Tensor size
 * @param params Quantization parameters per group
 * @param group_size Group size used during quantization
 */
void dequantize_tensor_int4(
    const uint8_t* input,
    float* output,
    size_t size,
    const std::vector<QuantizationParamsINT4>& params,
    size_t group_size = 128
);

/**
 * @brief INT4 matrix-vector multiplication (grouped quantization)
 * @param A Packed INT4 matrix (M x N)
 * @param x Float input vector (N)
 * @param y Float output vector (M)
 * @param M Rows in A
 * @param N Columns in A
 * @param A_params Quantization params for A (one per group)
 * @param group_size Group size
 */
void gemv_int4_grouped(
    const uint8_t* A,
    const float* x,
    float* y,
    size_t M, size_t N,
    const std::vector<QuantizationParamsINT4>& A_params,
    size_t group_size = 128
);

/**
 * @brief Pack two INT4 values into one byte
 */
inline uint8_t pack_int4_pair(int8_t low, int8_t high) {
    return static_cast<uint8_t>((high & 0x0F) << 4) | (low & 0x0F);
}

/**
 * @brief Unpack two INT4 values from one byte
 */
inline void unpack_int4_pair(uint8_t packed, int8_t& low, int8_t& high) {
    low = static_cast<int8_t>(packed & 0x0F);
    high = static_cast<int8_t>((packed >> 4) & 0x0F);
    // Sign extend from 4-bit to 8-bit
    if (low & 0x08) low |= 0xF0;
    if (high & 0x08) high |= 0xF0;
}

} // namespace Kernels
} // namespace CortexAICompression
