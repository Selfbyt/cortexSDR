#ifndef QUANTIZED_TENSOR_STRATEGY_HPP
#define QUANTIZED_TENSOR_STRATEGY_HPP

#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"
#include <vector>
#include <cstdint>
#include <cmath>

namespace CortexAICompression {

// Strategy that combines quantization with compression for floating-point tensors
class QuantizedTensorStrategy : public ICompressionStrategy {
public:
    // Constructor allows configuring quantization parameters
    explicit QuantizedTensorStrategy(uint8_t bits = 8, bool symmetric = true)
        : bits_(bits), symmetric_(symmetric) {}

    std::vector<std::byte> compress(const ModelSegment& segment) const override;
    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData,
                                    SegmentType originalType,
                                    size_t originalSize) const override;

private:
    uint8_t bits_;      // Number of bits for quantization (e.g., 8 for int8, 4 for int4)
    bool symmetric_;    // Whether to use symmetric quantization

    // Helper methods for quantization
    struct QuantizationParams {
        float scale;
        float zero_point;
        std::vector<int8_t> quantized_data;
    };

    QuantizationParams quantizeTensor(const std::vector<float>& data) const;
    std::vector<float> dequantizeTensor(const std::vector<int8_t>& qdata,
                                      float scale,
                                      float zero_point) const;

    // Helpers for packing/unpacking quantized values
    std::vector<std::byte> packQuantizedData(const std::vector<int8_t>& qdata) const;
    std::vector<int8_t> unpackQuantizedData(const std::vector<std::byte>& packed,
                                          size_t originalSize) const;
};

} // namespace CortexAICompression

#endif // QUANTIZED_TENSOR_STRATEGY_HPP 