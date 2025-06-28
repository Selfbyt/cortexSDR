#include "QuantizedTensorStrategy.hpp"
#include <iostream>
#include <cstring> // For memcpy
#include <algorithm> // For minmax_element
#include <limits>
#include <stdexcept>

namespace CortexAICompression {

std::vector<std::byte> QuantizedTensorStrategy::compress(const ModelSegment& segment) const {
    if (!segment.isWeightTensor()) {
        throw CompressionError("QuantizedTensorStrategy only supports weight tensors");
    }

    // Convert raw bytes to float vector
    const float* float_data = reinterpret_cast<const float*>(segment.data.data());
    size_t num_elements = segment.data.size() / sizeof(float);
    std::vector<float> float_vec(float_data, float_data + num_elements);

    // Quantize the data
    auto params = quantizeTensor(float_vec);

    // Pack the quantized data with metadata
    std::vector<std::byte> compressed;
    
    // Reserve space for metadata (scale, zero_point) + quantized data
    compressed.reserve(2 * sizeof(float) + params.quantized_data.size());

    // Write scale and zero_point
    const std::byte* scale_bytes = reinterpret_cast<const std::byte*>(&params.scale);
    const std::byte* zero_point_bytes = reinterpret_cast<const std::byte*>(&params.zero_point);
    
    compressed.insert(compressed.end(), scale_bytes, scale_bytes + sizeof(float));
    compressed.insert(compressed.end(), zero_point_bytes, zero_point_bytes + sizeof(float));

    // Pack and append the quantized data
    auto packed_data = packQuantizedData(params.quantized_data);
    compressed.insert(compressed.end(), packed_data.begin(), packed_data.end());

    return compressed;
}

std::vector<std::byte> QuantizedTensorStrategy::decompress(
    const std::vector<std::byte>& compressedData,
    SegmentType originalType,
    size_t originalSize) const {
    
    if (compressedData.size() < 2 * sizeof(float)) {
        throw CompressionError("Compressed data too small to contain metadata");
    }

    // Extract scale and zero_point
    float scale = *reinterpret_cast<const float*>(compressedData.data());
    float zero_point = *reinterpret_cast<const float*>(compressedData.data() + sizeof(float));

    // Extract and unpack quantized data
    std::vector<std::byte> packed_data(
        compressedData.begin() + 2 * sizeof(float),
        compressedData.end()
    );

    auto quantized_data = unpackQuantizedData(packed_data, originalSize / sizeof(float));

    // Dequantize
    auto float_data = dequantizeTensor(quantized_data, scale, zero_point);

    // Convert back to bytes
    std::vector<std::byte> decompressed(originalSize);
    std::memcpy(decompressed.data(), float_data.data(), originalSize);

    return decompressed;
}

QuantizedTensorStrategy::QuantizationParams 
QuantizedTensorStrategy::quantizeTensor(const std::vector<float>& data) const {
    QuantizationParams params;
    
    // Find data range
    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    float data_min = *min_it;
    float data_max = *max_it;

    // Calculate quantization parameters
    if (symmetric_) {
        float abs_max = std::max(std::abs(data_min), std::abs(data_max));
        params.scale = abs_max / ((1 << (bits_ - 1)) - 1);
        params.zero_point = 0.0f;
    } else {
        params.scale = (data_max - data_min) / ((1 << bits_) - 1);
        params.zero_point = -data_min / params.scale;
    }

    // Quantize the data
    params.quantized_data.reserve(data.size());
    for (float val : data) {
        float scaled = val / params.scale + params.zero_point;
        int8_t quantized = std::round(scaled);
        params.quantized_data.push_back(quantized);
    }

    return params;
}

std::vector<float> QuantizedTensorStrategy::dequantizeTensor(
    const std::vector<int8_t>& qdata,
    float scale,
    float zero_point) const {
    
    std::vector<float> dequantized;
    dequantized.reserve(qdata.size());

    for (int8_t val : qdata) {
        float dequantized_val = (val - zero_point) * scale;
        dequantized.push_back(dequantized_val);
    }

    return dequantized;
}

std::vector<std::byte> QuantizedTensorStrategy::packQuantizedData(
    const std::vector<int8_t>& qdata) const {
    
    if (bits_ == 8) {
        // For 8-bit quantization, just reinterpret as bytes
        return std::vector<std::byte>(
            reinterpret_cast<const std::byte*>(qdata.data()),
            reinterpret_cast<const std::byte*>(qdata.data() + qdata.size())
        );
    } else if (bits_ == 4) {
        // For 4-bit quantization, pack two values per byte
        std::vector<std::byte> packed((qdata.size() + 1) / 2);
        for (size_t i = 0; i < qdata.size(); i += 2) {
            uint8_t byte = (qdata[i] & 0x0F) << 4;
            if (i + 1 < qdata.size()) {
                byte |= (qdata[i + 1] & 0x0F);
            }
            packed[i / 2] = static_cast<std::byte>(byte);
        }
        return packed;
    }
    
    throw CompressionError("Unsupported number of bits for quantization");
}

std::vector<int8_t> QuantizedTensorStrategy::unpackQuantizedData(
    const std::vector<std::byte>& packed,
    size_t originalSize) const {
    
    if (bits_ == 8) {
        // For 8-bit quantization, just reinterpret as int8_t
        return std::vector<int8_t>(
            reinterpret_cast<const int8_t*>(packed.data()),
            reinterpret_cast<const int8_t*>(packed.data() + packed.size())
        );
    } else if (bits_ == 4) {
        // For 4-bit quantization, unpack two values per byte
        std::vector<int8_t> unpacked(originalSize);
        for (size_t i = 0; i < packed.size() && i * 2 < originalSize; ++i) {
            uint8_t byte = static_cast<uint8_t>(packed[i]);
            unpacked[i * 2] = (byte >> 4) & 0x0F;
            if (i * 2 + 1 < originalSize) {
                unpacked[i * 2 + 1] = byte & 0x0F;
            }
        }
        return unpacked;
    }
    
    throw CompressionError("Unsupported number of bits for quantization");
}

} // namespace CortexAICompression 