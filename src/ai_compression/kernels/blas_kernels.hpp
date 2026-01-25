/**
 * @file blas_kernels.hpp
 * @brief Optimized BLAS-based kernels for neural network operations
 * 
 * Provides high-performance implementations of matrix operations using
 * BLAS libraries (MKL, OpenBLAS, or standard BLAS).
 */

#pragma once

#include <vector>
#include <cstddef>

namespace CortexAICompression {
namespace Kernels {

/**
 * @brief Matrix-vector multiplication: y = alpha * A * x + beta * y
 * @param A Matrix in row-major format (M x N)
 * @param x Input vector (N)
 * @param y Output vector (M)
 * @param M Number of rows in A
 * @param N Number of columns in A
 * @param alpha Scaling factor for A*x
 * @param beta Scaling factor for y
 * @param transA Whether to transpose A
 */
void gemv(const float* A, const float* x, float* y,
          size_t M, size_t N,
          float alpha = 1.0f, float beta = 0.0f,
          bool transA = false);

/**
 * @brief Matrix-matrix multiplication: C = alpha * A * B + beta * C
 * @param A Matrix (M x K)
 * @param B Matrix (K x N)
 * @param C Output matrix (M x N)
 * @param M Rows in A and C
 * @param N Columns in B and C
 * @param K Columns in A, rows in B
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C
 * @param transA Whether to transpose A
 * @param transB Whether to transpose B
 */
void gemm(const float* A, const float* B, float* C,
          size_t M, size_t N, size_t K,
          float alpha = 1.0f, float beta = 0.0f,
          bool transA = false, bool transB = false);

/**
 * @brief Batched matrix-matrix multiplication for parallel processing
 * @param batch_size Number of independent matrix multiplications
 * @param A Array of input matrices
 * @param B Array of input matrices
 * @param C Array of output matrices
 * @param M Rows in each A matrix
 * @param N Columns in each B matrix
 * @param K Inner dimension
 */
void gemm_batched(size_t batch_size,
                  const float** A, const float** B, float** C,
                  size_t M, size_t N, size_t K);

/**
 * @brief Optimized linear layer forward pass: output = input * weights^T + bias
 * @param input Input tensor (batch_size x input_size)
 * @param weights Weight matrix (output_size x input_size)
 * @param bias Bias vector (output_size), can be nullptr
 * @param output Output tensor (batch_size x output_size)
 * @param batch_size Number of samples
 * @param input_size Input dimension
 * @param output_size Output dimension
 */
void linear_forward(const float* input, const float* weights, const float* bias,
                   float* output, size_t batch_size, size_t input_size, size_t output_size);

/**
 * @brief Sparse matrix-vector multiplication for SDR compressed weights
 * @param indices Sparse index array
 * @param values Sparse value array
 * @param num_nonzero Number of non-zero elements
 * @param input Input vector
 * @param output Output vector (must be pre-zeroed)
 * @param input_size Size of input vector
 * @param output_size Size of output vector
 */
void sparse_gemv(const size_t* indices, const float* values, size_t num_nonzero,
                 const float* input, float* output,
                 size_t input_size, size_t output_size);

/**
 * @brief Check if BLAS is available and which implementation
 * @return String describing BLAS implementation ("MKL", "OpenBLAS", "Generic", or "None")
 */
const char* get_blas_implementation();

} // namespace Kernels
} // namespace CortexAICompression
