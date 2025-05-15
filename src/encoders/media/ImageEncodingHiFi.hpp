#pragma once

#include <vector>
#include <string>
#include <utility>
#include <stdexcept>

/**
 * @brief High-fidelity SDR image codec using block-based DCT and sparse encoding.
 */
class ImageEncodingHiFi {
public:
    struct HiFiParams {
        int blockSize = 8; // 8x8 blocks
        int topCoeffs = 10; // Number of DCT coefficients per block to keep
        int quantLevel = 8; // Quantization level (bits per coefficient)
    };

    struct EncodedBlock {
        int x, y; // Block position
        std::vector<std::pair<int, int>> coeffs; // (index, quantized value)
    };

    struct EncodedImage {
        int width, height;
        int blockSize;
        int quantLevel;
        int topCoeffs;
        std::vector<EncodedBlock> blocks;
    };

    ImageEncodingHiFi() = default;

    /**
     * @brief Encode an image file to a high-fidelity SDR representation.
     * @param imagePath Path to the input image (PNG, JPEG, BMP).
     * @param params Encoding parameters.
     * @return EncodedImage structure.
     */
    EncodedImage encodeImageHiFi(const std::string& imagePath, const HiFiParams& params = HiFiParams()) const;

    /**
     * @brief Decode a high-fidelity SDR representation back to an image file.
     * @param encoded EncodedImage structure.
     * @param outputPath Path to write the reconstructed image.
     */
    void decodeImageHiFi(const EncodedImage& encoded, const std::string& outputPath) const;

    // Utility: Save/load EncodedImage to/from file (for CLI integration)
    void saveEncodedImage(const EncodedImage& encoded, const std::string& filePath) const;
    EncodedImage loadEncodedImage(const std::string& filePath) const;
};

class ImageEncodingHiFiError : public std::runtime_error {
public:
    explicit ImageEncodingHiFiError(const std::string& message) : std::runtime_error(message) {}
};
