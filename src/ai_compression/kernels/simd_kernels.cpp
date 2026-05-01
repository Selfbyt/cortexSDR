/**
 * @file simd_kernels.cpp
 * @brief Implementation of SIMD-optimized activation and element-wise kernels
 */

#include "simd_kernels.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(CORTEXSDR_SIMD_AVX2) || defined(__AVX2__)
        #include <immintrin.h>
        #define USE_AVX2
        #define SIMD_LEVEL "AVX2"
    #elif defined(CORTEXSDR_SIMD_SSE41) || defined(__SSE4_1__)
        #include <smmintrin.h>
        #define USE_SSE41
        #define SIMD_LEVEL "SSE4.1"
    #elif defined(CORTEXSDR_SIMD_SSE2) || defined(__SSE2__)
        #include <emmintrin.h>
        #define USE_SSE2
        #define SIMD_LEVEL "SSE2"
    #else
        #define SIMD_LEVEL "None"
    #endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define USE_NEON
    #define SIMD_LEVEL "NEON"
#else
    #define SIMD_LEVEL "None"
#endif

namespace CortexAICompression {
namespace Kernels {

const char* get_simd_level() {
    return SIMD_LEVEL;
}

void relu(const float* x, float* y, size_t size) {
#if defined(USE_AVX2)
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 val = _mm256_loadu_ps(x + i);
        val = _mm256_max_ps(val, zero);
        _mm256_storeu_ps(y + i, val);
    }
    // Handle remainder
    for (; i < size; ++i) {
        y[i] = std::max(0.0f, x[i]);
    }
#elif defined(USE_SSE2)
    const __m128 zero = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m128 val = _mm_loadu_ps(x + i);
        val = _mm_max_ps(val, zero);
        _mm_storeu_ps(y + i, val);
    }
    for (; i < size; ++i) {
        y[i] = std::max(0.0f, x[i]);
    }
#elif defined(USE_NEON)
    const float32x4_t zero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vmaxq_f32(val, zero);
        vst1q_f32(y + i, val);
    }
    for (; i < size; ++i) {
        y[i] = std::max(0.0f, x[i]);
    }
#else
    for (size_t i = 0; i < size; ++i) {
        y[i] = std::max(0.0f, x[i]);
    }
#endif
}

void leaky_relu(const float* x, float* y, size_t size, float alpha) {
#if defined(USE_AVX2)
    const __m256 alpha_vec = _mm256_set1_ps(alpha);
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 val = _mm256_loadu_ps(x + i);
        __m256 mask = _mm256_cmp_ps(val, zero, _CMP_GT_OQ);
        __m256 neg_part = _mm256_mul_ps(val, alpha_vec);
        val = _mm256_blendv_ps(neg_part, val, mask);
        _mm256_storeu_ps(y + i, val);
    }
    for (; i < size; ++i) {
        y[i] = x[i] > 0 ? x[i] : alpha * x[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
        y[i] = x[i] > 0 ? x[i] : alpha * x[i];
    }
#endif
}

void gelu(const float* x, float* y, size_t size) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    const float coeff = 0.044715f;
    
#if defined(USE_AVX2)
    const __m256 c_half = _mm256_set1_ps(0.5f);
    const __m256 c_one = _mm256_set1_ps(1.0f);
    const __m256 c_sqrt_2_pi = _mm256_set1_ps(sqrt_2_over_pi);
    const __m256 c_coeff = _mm256_set1_ps(coeff);
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 val = _mm256_loadu_ps(x + i);
        __m256 val3 = _mm256_mul_ps(_mm256_mul_ps(val, val), val);
        __m256 inner = _mm256_add_ps(val, _mm256_mul_ps(c_coeff, val3));
        inner = _mm256_mul_ps(c_sqrt_2_pi, inner);
        
        // Approximate tanh using rational function
        // tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2) for |x| < 2
        __m256 x2 = _mm256_mul_ps(inner, inner);
        __m256 num = _mm256_add_ps(_mm256_set1_ps(27.0f), x2);
        __m256 den = _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), x2));
        __m256 tanh_approx = _mm256_mul_ps(inner, _mm256_div_ps(num, den));
        
        __m256 result = _mm256_mul_ps(c_half, _mm256_mul_ps(val, _mm256_add_ps(c_one, tanh_approx)));
        _mm256_storeu_ps(y + i, result);
    }
    for (; i < size; ++i) {
        float val = x[i];
        float inner = sqrt_2_over_pi * (val + coeff * val * val * val);
        float tanh_val = std::tanh(inner);
        y[i] = 0.5f * val * (1.0f + tanh_val);
    }
#else
    for (size_t i = 0; i < size; ++i) {
        float val = x[i];
        float inner = sqrt_2_over_pi * (val + coeff * val * val * val);
        float tanh_val = std::tanh(inner);
        y[i] = 0.5f * val * (1.0f + tanh_val);
    }
#endif
}

void swish(const float* x, float* y, size_t size) {
    // Swish: x * sigmoid(x) = x / (1 + exp(-x))
#if defined(USE_AVX2)
    const __m256 c_one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 val = _mm256_loadu_ps(x + i);
        // Approximate exp using faster polynomial
        __m256 neg_val = _mm256_sub_ps(_mm256_setzero_ps(), val);
        
        // Fast exp approximation (less accurate but faster)
        // exp(x) ≈ (1 + x/256)^256 for small x
        // For better accuracy, use hardware intrinsics if available
        __m256 exp_val = _mm256_set1_ps(1.0f);
        for (int j = 0; j < 8; ++j) {
            exp_val = _mm256_mul_ps(exp_val, _mm256_add_ps(c_one, _mm256_mul_ps(neg_val, _mm256_set1_ps(1.0f/256.0f))));
            exp_val = _mm256_mul_ps(exp_val, _mm256_add_ps(c_one, _mm256_mul_ps(neg_val, _mm256_set1_ps(1.0f/256.0f))));
        }
        
        __m256 sigmoid = _mm256_div_ps(c_one, _mm256_add_ps(c_one, exp_val));
        __m256 result = _mm256_mul_ps(val, sigmoid);
        _mm256_storeu_ps(y + i, result);
    }
    for (; i < size; ++i) {
        float sigmoid = 1.0f / (1.0f + std::exp(-x[i]));
        y[i] = x[i] * sigmoid;
    }
#else
    for (size_t i = 0; i < size; ++i) {
        float sigmoid = 1.0f / (1.0f + std::exp(-x[i]));
        y[i] = x[i] * sigmoid;
    }
#endif
}

void sigmoid(const float* x, float* y, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

void tanh_activation(const float* x, float* y, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        y[i] = std::tanh(x[i]);
    }
}

void softmax(const float* x, float* y, size_t size) {
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < size; ++i) {
        max_val = std::max(max_val, x[i]);
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        y[i] = std::exp(x[i] - max_val);
        sum += y[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < size; ++i) {
        y[i] *= inv_sum;
    }
}

void layer_norm(const float* x, float* y, const float* gamma, const float* beta,
                size_t size, float eps) {
    // Compute mean
    float mean = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        mean += x[i];
    }
    mean /= size;
    
    // Compute variance
    float var = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    // Normalize and apply affine transform
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (size_t i = 0; i < size; ++i) {
        y[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

void add(const float* a, const float* b, float* y, size_t size) {
#if defined(USE_AVX2)
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 result = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(y + i, result);
    }
    for (; i < size; ++i) {
        y[i] = a[i] + b[i];
    }
#elif defined(USE_SSE2)
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 result = _mm_add_ps(va, vb);
        _mm_storeu_ps(y + i, result);
    }
    for (; i < size; ++i) {
        y[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
        y[i] = a[i] + b[i];
    }
#endif
}

void multiply(const float* a, const float* b, float* y, size_t size) {
#if defined(USE_AVX2)
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 result = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(y + i, result);
    }
    for (; i < size; ++i) {
        y[i] = a[i] * b[i];
    }
#elif defined(USE_SSE2)
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 result = _mm_mul_ps(va, vb);
        _mm_storeu_ps(y + i, result);
    }
    for (; i < size; ++i) {
        y[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
        y[i] = a[i] * b[i];
    }
#endif
}

void axpby(const float* a, const float* b, float* y, size_t size,
           float alpha, float beta) {
#if defined(USE_AVX2)
    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbeta = _mm256_set1_ps(beta);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 result = _mm256_add_ps(_mm256_mul_ps(valpha, va), _mm256_mul_ps(vbeta, vb));
        _mm256_storeu_ps(y + i, result);
    }
    for (; i < size; ++i) {
        y[i] = alpha * a[i] + beta * b[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
        y[i] = alpha * a[i] + beta * b[i];
    }
#endif
}

} // namespace Kernels
} // namespace CortexAICompression
