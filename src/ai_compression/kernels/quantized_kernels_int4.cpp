/**
 * @file quantized_kernels_int4.cpp
 * @brief Implementation of INT4 quantization kernels
 */

#include "quantized_kernels_int4.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace CortexAICompression {
namespace Kernels {

std::vector<QuantizationParamsINT4> quantize_tensor_int4(
    const float* input,
    uint8_t* output,
    size_t size,
    size_t group_size
) {
    std::vector<QuantizationParamsINT4> params;
    
    if (group_size == 0) {
        group_size = size;  // Per-tensor quantization
    }
    
    size_t num_groups = (size + group_size - 1) / group_size;
    params.reserve(num_groups);
    
    for (size_t g = 0; g < num_groups; ++g) {
        size_t start = g * group_size;
        size_t end = std::min(start + group_size, size);
        size_t group_len = end - start;
        
        // Find min/max in this group
        float min_val = std::numeric_limits<float>::infinity();
        float max_val = -std::numeric_limits<float>::infinity();
        
        for (size_t i = start; i < end; ++i) {
            min_val = std::min(min_val, input[i]);
            max_val = std::max(max_val, input[i]);
        }
        
        // Compute scale and zero point
        float range = max_val - min_val;
        if (range < 1e-8f) {
            range = 1.0f;
        }
        
        // INT4 range: -8 to 7 (16 values)
        float scale = range / 15.0f;
        int8_t zero_point = static_cast<int8_t>(std::round(-min_val / scale)) - 8;
        zero_point = std::clamp(zero_point, static_cast<int8_t>(-8), static_cast<int8_t>(7));
        
        params.emplace_back(scale, zero_point);
        
        // Quantize this group
        for (size_t i = start; i < end; i += 2) {
            int8_t q_low = 0, q_high = 0;
            
            // Quantize low nibble
            {
                int32_t q = static_cast<int32_t>(std::round(input[i] / scale)) + zero_point;
                q_low = static_cast<int8_t>(std::clamp(q, -8, 7));
            }
            
            // Quantize high nibble (if exists)
            if (i + 1 < end) {
                int32_t q = static_cast<int32_t>(std::round(input[i + 1] / scale)) + zero_point;
                q_high = static_cast<int8_t>(std::clamp(q, -8, 7));
            }
            
            // Pack into byte
            output[(start + i - start) / 2] = pack_int4_pair(q_low, q_high);
        }
    }
    
    return params;
}

void dequantize_tensor_int4(
    const uint8_t* input,
    float* output,
    size_t size,
    const std::vector<QuantizationParamsINT4>& params,
    size_t group_size
) {
    if (group_size == 0) {
        group_size = size;
    }
    
    size_t num_groups = (size + group_size - 1) / group_size;
    
    for (size_t g = 0; g < num_groups && g < params.size(); ++g) {
        size_t start = g * group_size;
        size_t end = std::min(start + group_size, size);
        
        const auto& p = params[g];
        
        for (size_t i = start; i < end; i += 2) {
            uint8_t packed = input[(start + i - start) / 2];
            int8_t q_low, q_high;
            unpack_int4_pair(packed, q_low, q_high);
            
            output[i] = p.scale * (q_low - p.zero_point);
            if (i + 1 < end) {
                output[i + 1] = p.scale * (q_high - p.zero_point);
            }
        }
    }
}

void gemv_int4_grouped(
    const uint8_t* A,
    const float* x,
    float* y,
    size_t M, size_t N,
    const std::vector<QuantizationParamsINT4>& A_params,
    size_t group_size
) {
    if (group_size == 0) {
        group_size = N;
    }
    
    // Zero output
    std::fill_n(y, M, 0.0f);
    
    size_t num_groups = (N + group_size - 1) / group_size;
    
    for (size_t i = 0; i < M; ++i) {
        float accumulator = 0.0f;
        
        for (size_t g = 0; g < num_groups && g < A_params.size(); ++g) {
            size_t col_start = g * group_size;
            size_t col_end = std::min(col_start + group_size, N);
            
            const auto& params = A_params[g];
            
            // Process this group
            for (size_t j = col_start; j < col_end; j += 2) {
                size_t packed_idx = (i * N + j) / 2;
                uint8_t packed = A[packed_idx];
                
                int8_t q_low, q_high;
                unpack_int4_pair(packed, q_low, q_high);
                
                float w_low = params.scale * (q_low - params.zero_point);
                accumulator += w_low * x[j];
                
                if (j + 1 < col_end) {
                    float w_high = params.scale * (q_high - params.zero_point);
                    accumulator += w_high * x[j + 1];
                }
            }
        }
        
        y[i] = accumulator;
    }
}

} // namespace Kernels
} // namespace CortexAICompression
