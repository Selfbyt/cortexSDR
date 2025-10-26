/**
 * @file QuantizedTensorStrategy.hpp
 * @brief Quantization-based compression strategy for neural network tensors
 * 
 * This header defines the QuantizedTensorStrategy class which provides
 * efficient compression of floating-point neural network tensors through
 * quantization to lower precision representations (INT8, INT4) with
 * optional symmetric/asymmetric quantization schemes.
 * 
 * Key Features:
 * - Configurable bit-width quantization (4-bit, 8-bit, etc.)
 * - Symmetric and asymmetric quantization modes
 * - Scale and zero-point parameter optimization
 * - Memory-efficient packed data storage
 * - High compression ratios with controlled accuracy loss
 * - Hardware-friendly quantized formats
 */

#ifndef QUANTIZED_TENSOR_STRATEGY_HPP
#define QUANTIZED_TENSOR_STRATEGY_HPP

#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"
#include <vector>
#include <cstdint>
#include <cmath>

namespace CortexAICompression {

/**
 * @brief Quantization-based compression strategy for floating-point tensors
 * 
 * Implements neural network tensor compression through quantization, converting
 * floating-point values to lower-precision integer representations with
 * configurable bit-widths and quantization schemes. Provides significant
 * compression ratios while maintaining acceptable model accuracy.
 */
class QuantizedTensorStrategy : public ICompressionStrategy {
public:
    /**
     * @brief Constructor with configurable quantization parameters
     * @param bits Number of bits for quantization (4, 8, 16, etc.)
     * @param symmetric True for symmetric quantization, false for asymmetric
     * 
     * Symmetric quantization centers around zero with equal positive/negative
     * ranges. Asymmetric quantization optimizes range utilization for biased data.
     */
    explicit QuantizedTensorStrategy(uint8_t bits = 8, bool symmetric = true)
        : bits_(bits), symmetric_(symmetric) {}

    /**
     * @brief Compress floating-point tensor data using quantization
     * @param segment ModelSegment containing floating-point tensor data
     * @return Compressed quantized data with scale/zero-point parameters
     * @throws CompressionError if segment type is unsupported
     */
    std::vector<std::byte> compress(const ModelSegment& segment) const override;
    
    /**
     * @brief Decompress quantized data back to floating-point representation
     * @param compressedData Quantized tensor data with parameters
     * @param originalType Original segment type for validation
     * @param originalSize Expected output size for validation
     * @return Reconstructed floating-point tensor data
     * @throws CompressionError if decompression fails or validation errors
     */
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData,
                                    SegmentType originalType,
                                    size_t originalSize) const override;

private:
    uint8_t bits_;      ///< Quantization bit-width (4, 8, 16, etc.)
    bool symmetric_;    ///< Symmetric vs asymmetric quantization mode

    /**
     * @brief Quantization parameters and results structure
     * 
     * Contains all information needed to quantize and dequantize tensor data
     * including scaling factors and quantized representation.
     */
    struct QuantizationParams {
        float scale;                        ///< Scaling factor for quantization
        float zero_point;                   ///< Zero-point offset for asymmetric quantization
        std::vector<int8_t> quantized_data; ///< Quantized integer representation
    };

    /**
     * @brief Quantize floating-point tensor to integer representation
     * @param data Input floating-point tensor data
     * @return Quantization parameters and quantized integer data
     */
    QuantizationParams quantizeTensor(const std::vector<float>& data) const;
    
    /**
     * @brief Dequantize integer data back to floating-point representation
     * @param qdata Quantized integer data
     * @param scale Scaling factor used during quantization
     * @param zero_point Zero-point offset for asymmetric quantization
     * @return Reconstructed floating-point tensor data
     */
    std::vector<float> dequantizeTensor(const std::vector<int8_t>& qdata,
                                      float scale,
                                      float zero_point) const;

    /**
     * @brief Pack quantized integer data into byte array for storage
     * @param qdata Quantized integer data to pack
     * @return Packed byte representation for efficient storage
     */
    std::vector<std::byte> packQuantizedData(const std::vector<int8_t>& qdata) const;
    
    /**
     * @brief Unpack byte array back to quantized integer data
     * @param packed Packed byte data from storage
     * @param originalSize Expected size of unpacked data for validation
     * @return Unpacked quantized integer data
     */
    std::vector<int8_t> unpackQuantizedData(const std::vector<std::byte>& packed,
                                          size_t originalSize) const;
};

} // namespace CortexAICompression

#endif // QUANTIZED_TENSOR_STRATEGY_HPP 