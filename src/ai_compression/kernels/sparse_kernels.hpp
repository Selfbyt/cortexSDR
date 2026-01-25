/**
 * @file sparse_kernels.hpp
 * @brief Optimized sparse matrix operations for SDR inference
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace CortexAICompression {
namespace SparseKernels {

/**
 * @brief Sparse matrix-vector multiplication (SpMV) with CSR format
 * @param row_ptr Row pointer array (size: num_rows + 1)
 * @param col_indices Column indices of non-zero elements
 * @param values Non-zero values
 * @param input Dense input vector
 * @param output Dense output vector (pre-allocated)
 * @param num_rows Number of output rows
 * 
 * @complexity O(nnz) where nnz is number of non-zeros
 */
void sparse_matrix_vector_multiply(
    const std::vector<size_t>& row_ptr,
    const std::vector<size_t>& col_indices,
    const std::vector<float>& values,
    const float* input,
    float* output,
    size_t num_rows
);

/**
 * @brief Optimized sparse linear layer with bias
 * @param indices Flat indices of active weights
 * @param values Active weight values
 * @param input Input tensor
 * @param bias Bias vector (nullable)
 * @param output Output tensor (pre-allocated)
 * @param input_size Input dimension
 * @param output_size Output dimension
 */
void sparse_linear_forward(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    const float* input,
    const float* bias,
    float* output,
    size_t input_size,
    size_t output_size
);

/**
 * @brief Block-sparse matrix-vector multiplication (for structured sparsity)
 * @param block_rows Block row indices (size: num_blocks)
 * @param block_cols Block column indices (size: num_blocks)
 * @param block_values Dense block values (size: num_blocks × block_size²)
 * @param input Dense input vector
 * @param output Dense output vector (pre-allocated)
 * @param num_block_rows Number of block rows
 * @param block_size Block dimension (e.g., 8 for 8×8 blocks)
 */
void block_sparse_matrix_vector_multiply(
    const std::vector<size_t>& block_rows,
    const std::vector<size_t>& block_cols,
    const std::vector<float>& block_values,
    const float* input,
    float* output,
    size_t num_block_rows,
    size_t block_size
);

/**
 * @brief Convert flat (index, value) pairs to CSR format for faster SpMV
 * @param indices Flat indices
 * @param values Corresponding values
 * @param input_size Input dimension (number of columns)
 * @param output_size Output dimension (number of rows)
 * @param out_row_ptr Output: row pointer array
 * @param out_col_indices Output: column indices
 * @param out_values Output: values
 */
void convert_to_csr(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    size_t input_size,
    size_t output_size,
    std::vector<size_t>& out_row_ptr,
    std::vector<size_t>& out_col_indices,
    std::vector<float>& out_values
);

/**
 * @brief Streaming sparse computation (zero-copy from compressed data)
 * @param compressed_data Varint-encoded sparse tensor
 * @param input Input vector
 * @param output Output vector (accumulated, should be pre-zeroed)
 * @param input_size Input dimension
 * @param callback Lambda called for each (index, value) pair
 */
void streaming_sparse_compute(
    const std::vector<std::byte>& compressed_data,
    const float* input,
    float* output,
    size_t input_size,
    const std::function<void(size_t, float, const float*, float*)>& callback
);

} // namespace SparseKernels
} // namespace CortexAICompression
