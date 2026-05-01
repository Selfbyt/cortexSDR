/**
 * @file blas_kernels.cpp
 * @brief Implementation of BLAS-based optimized kernels
 * 
 * @note Threading Support:
 * Most BLAS libraries (Intel MKL, OpenBLAS, Apple Accelerate) automatically
 * use multi-threading for GEMM operations. Thread count can be controlled via:
 * - MKL: MKL_NUM_THREADS or mkl_set_num_threads()
 * - OpenBLAS: OPENBLAS_NUM_THREADS or openblas_set_num_threads()
 * - Generic BLAS: OMP_NUM_THREADS environment variable
 * 
 * For optimal performance, set the thread count to match your CPU core count.
 */

#include "blas_kernels.hpp"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <thread>

#if defined(CORTEXSDR_SIMD_AVX2) || defined(__AVX2__)
    #include <immintrin.h>
    #define CORTEXSDR_BLAS_FALLBACK_AVX2 1
#endif

// BLAS headers depending on what's available
#if defined(USE_MKL)
    #include <mkl.h>
    #define BLAS_IMPL "Intel MKL"
#elif defined(USE_OPENBLAS)
    extern "C" {
        #include <cblas.h>
    }
    #define BLAS_IMPL "OpenBLAS"
#elif defined(ENABLE_BLAS)
    extern "C" {
        // Standard BLAS interface
        void cblas_sgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                        const int M, const int N, const float alpha, const float *A,
                        const int lda, const float *X, const int incX,
                        const float beta, float *Y, const int incY);
        void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                        const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const float alpha, const float *A,
                        const int lda, const float *B, const int ldb,
                        const float beta, float *C, const int ldc);
    }
    #define BLAS_IMPL "Generic BLAS"
#else
    #define BLAS_IMPL "None"
#endif

namespace CortexAICompression {
namespace Kernels {

#if !defined(ENABLE_BLAS)
namespace {

inline float dot_product_scalar(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(CORTEXSDR_BLAS_FALLBACK_AVX2)
inline float hsum256_ps(__m256 v) {
    const __m128 low = _mm256_castps256_ps128(v);
    const __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline float dot_product_avx2(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float sum = hsum256_ps(acc);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

inline float dot_product_fast(const float* a, const float* b, size_t n) {
#if defined(CORTEXSDR_BLAS_FALLBACK_AVX2)
    return dot_product_avx2(a, b, n);
#else
    return dot_product_scalar(a, b, n);
#endif
}

void linear_forward_fallback_single_batch(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t input_size,
    size_t output_size) {
    const size_t min_rows_per_thread = 64;
    const size_t desired_threads = (std::max)(size_t(1), output_size / min_rows_per_thread);
    const size_t worker_count = (std::min)(desired_threads, static_cast<size_t>((std::max)(1u, std::thread::hardware_concurrency())));

    auto run_range = [&](size_t row_begin, size_t row_end) {
        for (size_t row = row_begin; row < row_end; ++row) {
            const float* weight_row = weights + row * input_size;
            float value = dot_product_fast(input, weight_row, input_size);
            if (bias != nullptr) {
                value += bias[row];
            }
            output[row] = value;
        }
    };

    if (worker_count <= 1 || output_size < min_rows_per_thread) {
        run_range(0, output_size);
        return;
    }

    const size_t chunk = (output_size + worker_count - 1) / worker_count;
    std::vector<std::thread> workers;
    workers.reserve(worker_count - 1);
    for (size_t worker = 1; worker < worker_count; ++worker) {
        const size_t begin = worker * chunk;
        const size_t end = (std::min)(output_size, begin + chunk);
        if (begin >= end) {
            continue;
        }
        workers.emplace_back(run_range, begin, end);
    }
    run_range(0, (std::min)(output_size, chunk));
    for (auto& worker : workers) {
        worker.join();
    }
}

} // namespace
#endif

const char* get_blas_implementation() {
    return BLAS_IMPL;
}

void gemv(const float* A, const float* x, float* y,
          size_t M, size_t N,
          float alpha, float beta,
          bool transA) {
#if defined(ENABLE_BLAS)
    cblas_sgemv(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                M, N,
                alpha, A, N,
                x, 1,
                beta, y, 1);
#else
    // Fallback implementation
    if (beta == 0.0f) {
        std::fill(y, y + M, 0.0f);
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < M; ++i) {
            y[i] *= beta;
        }
    }
    
    if (!transA) {
        for (size_t i = 0; i < M; ++i) {
            const float sum = dot_product_fast(A + i * N, x, N);
            y[i] += alpha * sum;
        }
    } else {
        for (size_t j = 0; j < N; ++j) {
            float x_val = alpha * x[j];
            for (size_t i = 0; i < M; ++i) {
                y[i] += A[j * M + i] * x_val;
            }
        }
    }
#endif
}

void gemm(const float* A, const float* B, float* C,
          size_t M, size_t N, size_t K,
          float alpha, float beta,
          bool transA, bool transB) {
#if defined(ENABLE_BLAS)
    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;
    
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
#else
    // Fallback naive implementation
    if (beta == 0.0f) {
        std::fill(C, C + M * N, 0.0f);
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < M * N; ++i) {
            C[i] *= beta;
        }
    }
    
    // Simple triple-loop matrix multiplication
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a_val = transA ? A[k * M + i] : A[i * K + k];
                float b_val = transB ? B[j * K + k] : B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] += alpha * sum;
        }
    }
#endif
}

void gemm_batched(size_t batch_size,
                  const float** A, const float** B, float** C,
                  size_t M, size_t N, size_t K) {
#if defined(USE_MKL)
    // MKL provides optimized batched GEMM
    std::vector<CBLAS_TRANSPOSE> transA(batch_size, CblasNoTrans);
    std::vector<CBLAS_TRANSPOSE> transB(batch_size, CblasNoTrans);
    std::vector<int> m_array(batch_size, M);
    std::vector<int> n_array(batch_size, N);
    std::vector<int> k_array(batch_size, K);
    std::vector<float> alpha_array(batch_size, 1.0f);
    std::vector<float> beta_array(batch_size, 0.0f);
    std::vector<int> lda_array(batch_size, K);
    std::vector<int> ldb_array(batch_size, N);
    std::vector<int> ldc_array(batch_size, N);
    std::vector<int> group_size(1, batch_size);
    
    cblas_sgemm_batch(CblasRowMajor,
                     transA.data(), transB.data(),
                     m_array.data(), n_array.data(), k_array.data(),
                     alpha_array.data(),
                     A, lda_array.data(),
                     B, ldb_array.data(),
                     beta_array.data(),
                     C, ldc_array.data(),
                     1, group_size.data());
#else
    // Fall back to sequential processing
    for (size_t b = 0; b < batch_size; ++b) {
        gemm(A[b], B[b], C[b], M, N, K, 1.0f, 0.0f, false, false);
    }
#endif
}

void linear_forward(const float* input, const float* weights, const float* bias,
                   float* output, size_t batch_size, size_t input_size, size_t output_size) {
#if !defined(ENABLE_BLAS)
    if (batch_size == 1) {
        linear_forward_fallback_single_batch(
            input, weights, bias, output, input_size, output_size);
        return;
    }
#endif

    // For linear layer: output = input * weights^T + bias
    // input: (batch_size x input_size)
    // weights: (output_size x input_size)
    // output: (batch_size x output_size)
    
    // This is equivalent to: output = input * weights^T
    // So we do GEMM with weights transposed
    gemm(input, weights, output,
         batch_size, output_size, input_size,
         1.0f, 0.0f, false, true);
    
    // Add bias if provided
    if (bias != nullptr) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < output_size; ++i) {
                output[b * output_size + i] += bias[i];
            }
        }
    }
}

void sparse_gemv(const size_t* indices, const float* values, size_t num_nonzero,
                 const float* input, float* output,
                 size_t input_size, size_t output_size) {
    // Optimized sparse matrix-vector multiplication
    // Assumes output is already zeroed
    
    for (size_t i = 0; i < num_nonzero; ++i) {
        size_t flat_idx = indices[i];
        float weight = values[i];
        
        // Decode flat index to (row, col)
        size_t row = flat_idx / input_size;
        size_t col = flat_idx % input_size;
        
        if (row < output_size && col < input_size) {
            output[row] += weight * input[col];
        }
    }
}

} // namespace Kernels
} // namespace CortexAICompression
