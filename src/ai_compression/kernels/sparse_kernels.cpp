/**
 * @file sparse_kernels.cpp
 * @brief Implementation of optimized sparse matrix operations
 */

#include "sparse_kernels.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace CortexAICompression {
namespace SparseKernels {

/**
 * @brief CSR sparse matrix-vector multiplication
 * 
 * CSR (Compressed Sparse Row) format is optimal for row-wise access patterns.
 * For matrix W (M×N) with nnz non-zeros:
 *   row_ptr[i] = starting index of row i in col_indices/values
 *   col_indices[k] = column index of non-zero k
 *   values[k] = value of non-zero k
 * 
 * Complexity: O(nnz) - only touches non-zero elements
 */
void sparse_matrix_vector_multiply(
    const std::vector<size_t>& row_ptr,
    const std::vector<size_t>& col_indices,
    const std::vector<float>& values,
    const float* input,
    float* output,
    size_t num_rows
) {
    // Zero output
    std::fill_n(output, num_rows, 0.0f);
    
    // Process each row
    for (size_t row = 0; row < num_rows; ++row) {
        float accumulator = 0.0f;
        size_t row_start = row_ptr[row];
        size_t row_end = row_ptr[row + 1];
        
        // Accumulate dot product for this row
        for (size_t k = row_start; k < row_end; ++k) {
            size_t col = col_indices[k];
            accumulator += values[k] * input[col];
        }
        
        output[row] = accumulator;
    }
}

/**
 * @brief Optimized sparse linear layer
 * 
 * For linear layer: y = Wx + b
 * Where W is stored as flat (index, value) pairs
 * 
 * Flat index mapping: index = row * input_size + col
 * This allows direct streaming without intermediate CSR conversion
 */
void sparse_linear_forward(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    const float* input,
    const float* bias,
    float* output,
    size_t input_size,
    size_t output_size
) {
    // Initialize output with bias (if provided)
    if (bias) {
        std::memcpy(output, bias, output_size * sizeof(float));
    } else {
        std::fill_n(output, output_size, 0.0f);
    }
    
    // Accumulate sparse weights
    const size_t nnz = indices.size();
    
    // Cache-friendly processing: group by output row
    for (size_t k = 0; k < nnz; ++k) {
        size_t flat_idx = indices[k];
        size_t row = flat_idx / input_size;
        size_t col = flat_idx % input_size;
        
        if (row < output_size && col < input_size) {
            output[row] += values[k] * input[col];
        }
    }
}

/**
 * @brief Block-sparse matrix-vector multiplication
 * 
 * Block sparsity provides better cache locality and vectorization opportunities.
 * Instead of storing individual elements, stores dense blocks.
 * 
 * Example: 8×8 blocks mean 64 elements per block
 * Block-sparse (1% blocks, block_size=8) ≈ 64% individual sparsity
 * 
 * This is particularly effective for quantized networks where blocks are
 * the natural granularity.
 */
void block_sparse_matrix_vector_multiply(
    const std::vector<size_t>& block_rows,
    const std::vector<size_t>& block_cols,
    const std::vector<float>& block_values,
    const float* input,
    float* output,
    size_t num_block_rows,
    size_t block_size
) {
    std::fill_n(output, num_block_rows * block_size, 0.0f);
    
    const size_t block_area = block_size * block_size;
    const size_t num_blocks = block_rows.size();
    
    for (size_t b = 0; b < num_blocks; ++b) {
        size_t br = block_rows[b];
        size_t bc = block_cols[b];
        const float* block = &block_values[b * block_area];
        
        // Dense multiplication within block
        for (size_t i = 0; i < block_size; ++i) {
            float accumulator = 0.0f;
            
#ifdef __AVX2__
            // SIMD version for AVX2
            if (block_size >= 8) {
                __m256 sum_vec = _mm256_setzero_ps();
                for (size_t j = 0; j + 7 < block_size; j += 8) {
                    __m256 block_vec = _mm256_loadu_ps(&block[i * block_size + j]);
                    __m256 input_vec = _mm256_loadu_ps(&input[bc * block_size + j]);
                    sum_vec = _mm256_fmadd_ps(block_vec, input_vec, sum_vec);
                }
                
                // Horizontal sum
                float sum_array[8];
                _mm256_storeu_ps(sum_array, sum_vec);
                for (int k = 0; k < 8; ++k) {
                    accumulator += sum_array[k];
                }
                
                // Handle remaining elements
                for (size_t j = (block_size / 8) * 8; j < block_size; ++j) {
                    accumulator += block[i * block_size + j] * input[bc * block_size + j];
                }
            } else
#endif
            {
                // Scalar fallback
                for (size_t j = 0; j < block_size; ++j) {
                    accumulator += block[i * block_size + j] * input[bc * block_size + j];
                }
            }
            
            output[br * block_size + i] += accumulator;
        }
    }
}

/**
 * @brief Convert flat index format to CSR for better cache locality
 * 
 * CSR format provides better memory access patterns for SpMV:
 * - Sequential access to values/col_indices
 * - Predictable branches
 * - Better prefetching
 * 
 * Conversion overhead is amortized over multiple inferences with same weights.
 */
void convert_to_csr(
    const std::vector<size_t>& indices,
    const std::vector<float>& values,
    size_t input_size,
    size_t output_size,
    std::vector<size_t>& out_row_ptr,
    std::vector<size_t>& out_col_indices,
    std::vector<float>& out_values
) {
    const size_t nnz = indices.size();
    
    // Initialize row_ptr
    out_row_ptr.assign(output_size + 1, 0);
    
    // Count non-zeros per row
    for (size_t k = 0; k < nnz; ++k) {
        size_t row = indices[k] / input_size;
        if (row < output_size) {
            out_row_ptr[row + 1]++;
        }
    }
    
    // Compute cumulative sum to get row pointers
    for (size_t row = 0; row < output_size; ++row) {
        out_row_ptr[row + 1] += out_row_ptr[row];
    }
    
    // Allocate output arrays
    out_col_indices.resize(nnz);
    out_values.resize(nnz);
    
    // Fill in column indices and values
    std::vector<size_t> current_row_positions = out_row_ptr;
    
    for (size_t k = 0; k < nnz; ++k) {
        size_t flat_idx = indices[k];
        size_t row = flat_idx / input_size;
        size_t col = flat_idx % input_size;
        
        if (row < output_size && col < input_size) {
            size_t pos = current_row_positions[row]++;
            out_col_indices[pos] = col;
            out_values[pos] = values[k];
        }
    }
}

/**
 * @brief Zero-copy streaming sparse computation
 * 
 * This function processes compressed sparse data without intermediate decompression.
 * The callback is invoked for each (index, value) pair, allowing custom accumulation.
 * 
 * This is the CORE of CortexSDR's zero-decompression inference.
 */
void streaming_sparse_compute(
    const std::vector<std::byte>& compressed_data,
    const float* input,
    float* output,
    size_t input_size,
    const std::function<void(size_t, float, const float*, float*)>& callback
) {
    // This function is a wrapper around the SDR decoder
    // For actual implementation, it would call forEachIndexValue
    // Here we provide the interface signature
    
    // Example usage (pseudocode):
    // SDRIndexStorageStrategy decoder;
    // decoder.forEachIndexValue(compressed_data, [&](size_t idx, float val) {
    //     callback(idx, val, input, output);
    // });
}

} // namespace SparseKernels
} // namespace CortexAICompression
