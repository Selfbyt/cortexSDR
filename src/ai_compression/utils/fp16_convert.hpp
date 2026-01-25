/**
 * @file fp16_convert.hpp
 * @brief FP16 to FP32 conversion utilities for mixed precision inference
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#if defined(__F16C__)
#include <immintrin.h>
#define USE_F16C
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define USE_NEON_FP16
#endif

namespace CortexAICompression {
namespace Utils {

/**
 * @brief Convert single FP16 value to FP32
 */
inline float fp16_to_fp32(uint16_t h) {
#ifdef USE_F16C
    __m128i h_vec = _mm_cvtsi32_si128(h);
    __m128 f_vec = _mm_cvtph_ps(h_vec);
    return _mm_cvtss_f32(f_vec);
#else
    // Manual conversion using bit manipulation
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            f = sign << 31;
        } else {
            // Denormalized number
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        // Infinity or NaN
        f = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
        // Normalized number
        f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    
    return *reinterpret_cast<float*>(&f);
#endif
}

/**
 * @brief Convert FP32 value to FP16
 */
inline uint16_t fp32_to_fp16(float f) {
#ifdef USE_F16C
    __m128 f_vec = _mm_set_ss(f);
    __m128i h_vec = _mm_cvtps_ph(f_vec, _MM_FROUND_TO_NEAREST_INT);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(h_vec));
#else
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exponent = (x >> 23) & 0xFF;
    uint32_t mantissa = x & 0x7FFFFF;
    
    uint16_t h;
    if (exponent == 0) {
        h = sign << 15;
    } else if (exponent == 0xFF) {
        h = (sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0);
    } else {
        int32_t new_exp = static_cast<int32_t>(exponent) - 127 + 15;
        if (new_exp >= 0x1F) {
            h = (sign << 15) | 0x7C00;
        } else if (new_exp <= 0) {
            h = sign << 15;
        } else {
            h = (sign << 15) | (new_exp << 10) | (mantissa >> 13);
        }
    }
    return h;
#endif
}

/**
 * @brief Batch convert FP16 array to FP32
 */
inline void fp16_to_fp32_array(const uint16_t* src, float* dst, size_t count) {
#ifdef USE_F16C
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i h_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f_vec = _mm256_cvtph_ps(h_vec);
        _mm256_storeu_ps(dst + i, f_vec);
    }
    for (; i < count; ++i) {
        dst[i] = fp16_to_fp32(src[i]);
    }
#elif defined(USE_NEON_FP16)
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float16x4_t h_vec = vld1_f16(reinterpret_cast<const __fp16*>(src + i));
        float32x4_t f_vec = vcvt_f32_f16(h_vec);
        vst1q_f32(dst + i, f_vec);
    }
    for (; i < count; ++i) {
        dst[i] = fp16_to_fp32(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = fp16_to_fp32(src[i]);
    }
#endif
}

/**
 * @brief Batch convert FP32 array to FP16
 */
inline void fp32_to_fp16_array(const float* src, uint16_t* dst, size_t count) {
#ifdef USE_F16C
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 f_vec = _mm256_loadu_ps(src + i);
        __m128i h_vec = _mm256_cvtps_ph(f_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h_vec);
    }
    for (; i < count; ++i) {
        dst[i] = fp32_to_fp16(src[i]);
    }
#elif defined(USE_NEON_FP16)
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t f_vec = vld1q_f32(src + i);
        float16x4_t h_vec = vcvt_f16_f32(f_vec);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i), h_vec);
    }
    for (; i < count; ++i) {
        dst[i] = fp32_to_fp16(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = fp32_to_fp16(src[i]);
    }
#endif
}

/**
 * @brief Convert FP16 buffer to FP32 vector
 */
inline std::vector<float> fp16_buffer_to_fp32(const std::vector<std::byte>& buffer) {
    size_t count = buffer.size() / sizeof(uint16_t);
    std::vector<float> result(count);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(buffer.data());
    fp16_to_fp32_array(src, result.data(), count);
    return result;
}

} // namespace Utils
} // namespace CortexAICompression
