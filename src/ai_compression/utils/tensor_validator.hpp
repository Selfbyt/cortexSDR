/**
 * @file tensor_validator.hpp
 * @brief Tensor shape validation and error checking utilities
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

namespace CortexAICompression {
namespace Utils {

class TensorValidationError : public std::runtime_error {
public:
    explicit TensorValidationError(const std::string& msg) : std::runtime_error(msg) {}
};

class TensorValidator {
public:
    /**
     * @brief Validate tensor dimensions match expected size
     */
    static void validate_size(
        const std::vector<float>& tensor,
        const std::vector<size_t>& expected_shape,
        const std::string& tensor_name
    ) {
        size_t expected_size = 1;
        for (size_t dim : expected_shape) {
            expected_size *= dim;
        }
        
        if (tensor.size() != expected_size) {
            std::ostringstream oss;
            oss << "Tensor '" << tensor_name << "' size mismatch. "
                << "Expected " << expected_size << " (shape: [";
            for (size_t i = 0; i < expected_shape.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << expected_shape[i];
            }
            oss << "]), got " << tensor.size();
            throw TensorValidationError(oss.str());
        }
    }
    
    /**
     * @brief Validate shapes are compatible for matrix multiplication
     */
    static void validate_matmul(
        const std::vector<size_t>& shape_a,
        const std::vector<size_t>& shape_b,
        const std::string& context
    ) {
        if (shape_a.empty() || shape_b.empty()) {
            throw TensorValidationError(context + ": Empty shape for matrix multiplication");
        }
        
        // Get inner dimensions
        size_t inner_a = shape_a.back();
        size_t inner_b = shape_b.size() >= 2 ? shape_b[shape_b.size() - 2] : shape_b[0];
        
        if (inner_a != inner_b) {
            std::ostringstream oss;
            oss << context << ": Incompatible shapes for matrix multiplication. "
                << "Inner dimensions " << inner_a << " and " << inner_b << " don't match";
            throw TensorValidationError(oss.str());
        }
    }
    
    /**
     * @brief Validate weight matrix dimensions for linear layer
     */
    static void validate_linear_weights(
        size_t weight_size,
        size_t input_dim,
        size_t output_dim,
        const std::string& layer_name
    ) {
        size_t expected = input_dim * output_dim;
        if (weight_size != expected && weight_size != 0) {  // 0 = compressed/sparse
            std::ostringstream oss;
            oss << "Layer '" << layer_name << "' weight size mismatch. "
                << "Expected " << expected << " (" << output_dim << " x " << input_dim 
                << "), got " << weight_size;
            throw TensorValidationError(oss.str());
        }
    }
    
    /**
     * @brief Validate convolution parameters
     */
    static void validate_conv_params(
        const std::vector<size_t>& input_shape,
        const std::vector<size_t>& kernel_shape,
        const std::vector<size_t>& strides,
        const std::vector<size_t>& padding,
        const std::string& layer_name
    ) {
        if (input_shape.size() != 4) {
            throw TensorValidationError(layer_name + ": Conv input must be 4D (NCHW)");
        }
        
        if (kernel_shape.size() != 2) {
            throw TensorValidationError(layer_name + ": Kernel shape must be 2D [H, W]");
        }
        
        if (strides.size() != 2) {
            throw TensorValidationError(layer_name + ": Strides must be 2D [H, W]");
        }
        
        if (padding.size() != 2) {
            throw TensorValidationError(layer_name + ": Padding must be 2D [H, W]");
        }
        
        // Validate output dimensions are positive
        int out_h = (static_cast<int>(input_shape[2]) + 2 * static_cast<int>(padding[0]) 
                     - static_cast<int>(kernel_shape[0])) / static_cast<int>(strides[0]) + 1;
        int out_w = (static_cast<int>(input_shape[3]) + 2 * static_cast<int>(padding[1]) 
                     - static_cast<int>(kernel_shape[1])) / static_cast<int>(strides[1]) + 1;
        
        if (out_h <= 0 || out_w <= 0) {
            std::ostringstream oss;
            oss << layer_name << ": Invalid conv parameters. "
                << "Output dimensions would be non-positive: " << out_h << " x " << out_w;
            throw TensorValidationError(oss.str());
        }
    }
    
    /**
     * @brief Validate shapes are broadcastable
     */
    static bool are_broadcastable(
        const std::vector<size_t>& shape_a,
        const std::vector<size_t>& shape_b
    ) {
        size_t max_ndim = std::max(shape_a.size(), shape_b.size());
        
        for (size_t i = 0; i < max_ndim; ++i) {
            size_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
            size_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;
            
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * @brief Check if tensor is finite (no NaN or Inf)
     */
    static bool is_finite(const std::vector<float>& tensor) {
        for (float val : tensor) {
            if (!std::isfinite(val)) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * @brief Get shape string for error messages
     */
    static std::string shape_to_string(const std::vector<size_t>& shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    }
};

} // namespace Utils
} // namespace CortexAICompression
