#include "ImageEncoding.hpp"
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <cstring>

class ImageEncoding::Impl {
public:
    // Implementation details will go here
    std::vector<unsigned char> buffer;
    
    bool checkFileHeader(const std::vector<unsigned char>& data) const {
        if (data.size() < 8) return false;
        
        // Check PNG signature
        static const unsigned char PNG_SIG[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
        if (std::equal(data.begin(), data.begin() + 8, PNG_SIG)) {
            return true;
        }
        
        // Check JPEG signature
        if (data[0] == 0xFF && data[1] == 0xD8) {
            return true;
        }
        
        // Check BMP signature
        if (data[0] == 0x42 && data[1] == 0x4D) {
            return true;
        }
        
        return false;
    }
};

ImageEncoding::ImageEncoding(Format defaultFormat)
    : pimpl(std::make_unique<Impl>())
    , defaultFormat(defaultFormat)
{
}

ImageEncoding::~ImageEncoding() = default;

std::vector<size_t> ImageEncoding::encodeImage(
    const std::string& imagePath,
    Format format,
    const CompressionOptions& options) const
{
    if (!isValidPath(imagePath)) {
        throw ImageEncodingError("Invalid image path: " + imagePath);
    }
    std::vector<unsigned char> imageData = readImageFile(imagePath);
    if (imageData.empty()) {
        throw ImageEncodingError("Failed to read image file: " + imagePath);
    }
    if (!pimpl->checkFileHeader(imageData)) {
        throw ImageEncodingError("Invalid image format or corrupted file: " + imagePath);
    }

    std::vector<size_t> encoded;
    // --- SDR-inspired encoding: quantize, sparsify, and map to indices ---
    // For demonstration, quantize each byte to a smaller set (e.g., 16 levels), sparsify by skipping zeros
    constexpr unsigned char quantLevels = 16;
    for (size_t i = 0; i < imageData.size(); ++i) {
        unsigned char quantized = static_cast<unsigned char>((imageData[i] * quantLevels) / 256);
        if (quantized == 0) continue; // sparsify: skip zeros
        // Map (position, quantized value) to a unique index
        size_t idx = i * quantLevels + quantized;
        encoded.push_back(idx);
    }
    // Store image size as the first index for decoding
    encoded.insert(encoded.begin(), imageData.size());
    return encoded;
}

void ImageEncoding::decodeIndices(
    const std::vector<size_t>& indices,
    const std::string& outputPath,
    Format format) const
{
    if (!isValidPath(outputPath)) {
        throw ImageEncodingError("Invalid output path: " + outputPath);
    }
    if (indices.empty()) {
        throw ImageEncodingError("No indices to decode");
    }
    // First index is image size
    size_t imageSize = indices[0];
    std::vector<unsigned char> decoded(imageSize, 0);
    constexpr unsigned char quantLevels = 16;
    // Each index after the first encodes (position, quantized value)
    for (size_t k = 1; k < indices.size(); ++k) {
        size_t idx = indices[k];
        size_t pos = idx / quantLevels;
        unsigned char quantized = static_cast<unsigned char>(idx % quantLevels);
        // Dequantize
        unsigned char value = static_cast<unsigned char>((quantized * 256) / quantLevels);
        if (pos < decoded.size()) decoded[pos] = value;
    }
    writeImageFile(outputPath, decoded);
}

void ImageEncoding::convertFormat(
    const std::string& inputPath,
    const std::string& outputPath,
    Format format,
    const CompressionOptions& options) const
{
    if (!isValidPath(inputPath) || !isValidPath(outputPath)) {
        throw ImageEncodingError("Invalid input or output path");
    }

    std::vector<unsigned char> imageData = readImageFile(inputPath);
    if (imageData.empty()) {
        throw ImageEncodingError("Failed to read input image");
    }

    // In a real implementation, you'd convert the image format here
    writeImageFile(outputPath, imageData);
}

ImageEncoding::Format ImageEncoding::detectFormat(const std::string& imagePath) const {
    std::vector<unsigned char> header = readImageFile(imagePath);
    if (header.size() < 8) {
        throw ImageEncodingError("File too small to determine format");
    }

    if (header[0] == 0x89 && header[1] == 0x50) return Format::PNG;
    if (header[0] == 0xFF && header[1] == 0xD8) return Format::JPEG;
    if (header[0] == 0x42 && header[1] == 0x4D) return Format::BMP;

    throw ImageEncodingError("Unknown image format");
}

bool ImageEncoding::validateImage(const std::string& imagePath) const {
    try {
        std::vector<unsigned char> imageData = readImageFile(imagePath);
        return !imageData.empty() && pimpl->checkFileHeader(imageData);
    } catch (...) {
        return false;
    }
}

std::pair<size_t, size_t> ImageEncoding::getImageDimensions(const std::string& imagePath) const {
    std::vector<unsigned char> imageData = readImageFile(imagePath);
    if (imageData.size() < 24) {
        throw ImageEncodingError("File too small to determine dimensions");
    }

    // This is a simplified example - real implementation would need proper parsing
    // based on image format
    size_t width = 0, height = 0;
    
    Format format = detectFormat(imagePath);
    switch (format) {
        case Format::PNG:
            // PNG dimensions are at offset 16
            width = (imageData[16] << 24) | (imageData[17] << 16) | 
                   (imageData[18] << 8) | imageData[19];
            height = (imageData[20] << 24) | (imageData[21] << 16) | 
                    (imageData[22] << 8) | imageData[23];
            break;
        // Add other format parsing...
        default:
            throw ImageEncodingError("Unsupported format for dimension detection");
    }

    return {width, height};
}

// Helper method implementations
bool ImageEncoding::isValidPath(const std::string& path) const {
    return !path.empty() && std::filesystem::path(path).has_filename();
}

std::vector<unsigned char> ImageEncoding::readImageFile(const std::string& path) const {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw ImageEncodingError("Cannot open file: " + path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    return buffer;
}

void ImageEncoding::writeImageFile(
    const std::string& path,
    const std::vector<unsigned char>& data) const
{
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw ImageEncodingError("Cannot create output file: " + path);
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}