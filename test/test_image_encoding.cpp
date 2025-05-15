#include "../src/encoders/media/ImageEncoding.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>

int main() {
    ImageEncoding encoder;
    ImageEncoding::CompressionOptions opts;
    opts.quality = 100; // HiFi SDR
    opts.compressionLevel = 1;

    // Test image path (provide your own PNG/JPEG/BMP for real test)
    std::string input = "../Screenshot From 2025-02-27 12-38-40.png";
    std::string output = "test_image_recon.png";

    // Encode
    std::vector<size_t> indices;
    try {
        indices = encoder.encodeImage(input, ImageEncoding::Format::PNG, opts);
        std::cout << "Encoded indices size: " << indices.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Encoding failed: " << e.what() << std::endl;
        return 1;
    }

    // Decode
    try {
        encoder.decodeIndices(indices, output, ImageEncoding::Format::PNG);
        std::cout << "Decoded image written to: " << output << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Decoding failed: " << e.what() << std::endl;
        return 1;
    }

    // Optionally, add pixel-wise comparison or PSNR check here
    return 0;
}
