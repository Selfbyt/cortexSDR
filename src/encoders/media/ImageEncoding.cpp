#include "ImageEncoding.hpp"
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <limits>
using std::fabsf;
using std::sqrtf;
using std::cosf;
using std::roundf;
using std::min;
using std::max;


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

    // --- Select encoding method based on options ---
    // 1. HiFi SDR (block-based DCT, adaptive quantization, sparse SDR)
    // 2. RLE (Run-Length Encoding)
    // 3. Simple SDR (legacy)

    // HiFi mode: Use block-based SDR (DCT)
    if (options.quality > 95) {
        std::vector<size_t> encoded;
        // Use block-based SDR encoding
        // --- Inline DCT-based encoding (grayscale only) ---
        int width = 0, height = 0;
        // Try to get dimensions from format
        try {
            auto dims = getImageDimensions(imagePath);
            width = static_cast<int>(dims.first);
            height = static_cast<int>(dims.second);
        } catch (...) {
            throw ImageEncodingError("Failed to get image dimensions for HiFi encoding");
        }
        int blockSize = 8;
        int quantLevel = 8;
        int topCoeffs = 10;
        int N = blockSize;
        int nBlocksX = (width + N - 1) / N;
        int nBlocksY = (height + N - 1) / N;
        encoded.push_back(width);
        encoded.push_back(height);
        encoded.push_back(blockSize);
        encoded.push_back(quantLevel);
        encoded.push_back(topCoeffs);
        for (int by = 0; by < nBlocksY; ++by) {
            for (int bx = 0; bx < nBlocksX; ++bx) {
                float block[64] = {0};
                for (int y = 0; y < N; ++y) {
                    for (int x = 0; x < N; ++x) {
                        int ix = bx * N + x;
                        int iy = by * N + y;
                        int idx = iy * width + ix;
                        block[y * N + x] = (ix < width && iy < height && idx < (int)imageData.size()) ? imageData[idx] : 0.0f;
                    }
                }
                // 2D DCT
                float dctCoeffs[64];
                const float PI = 3.14159265358979323846f;
                for (int u = 0; u < N; ++u) {
                    for (int v = 0; v < N; ++v) {
                        float sum = 0.0f;
                        for (int x = 0; x < N; ++x) {
                            for (int y = 0; y < N; ++y) {
                                sum += block[x * N + y] *
                                    cosf(PI * (2 * x + 1) * u / (2 * N)) *
                                    cosf(PI * (2 * y + 1) * v / (2 * N));
                            }
                        }
                        float alphaU = (u == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
                        float alphaV = (v == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
                        dctCoeffs[u * N + v] = alphaU * alphaV * sum;
                    }
                }
                // Quantize and select top-K
                std::vector<std::pair<int, int>> qcoeffs;
                std::vector<std::pair<float, int>> abs_coeffs;
                for (int i = 0; i < N * N; ++i) {
                    abs_coeffs.push_back(std::pair<float,int>(fabsf(dctCoeffs[i]), i));
                }
                std::partial_sort(abs_coeffs.begin(), abs_coeffs.begin() + topCoeffs, abs_coeffs.end(), [](const std::pair<float,int>& a, const std::pair<float,int>& b){return a.first > b.first;});
                for (int k = 0; k < topCoeffs; ++k) {
                    int idx = abs_coeffs[k].second;
                    float val = dctCoeffs[idx];
                    int qval = (int)roundf(val * ((1 << quantLevel) - 1) / 1024.0f);
                    if (qval != 0) {
                        // Encode as (block x, block y, coeff idx, qval)
                        encoded.push_back(bx);
                        encoded.push_back(by);
                        encoded.push_back(idx);
                        encoded.push_back(qval);
                    }
                }
            }
        }
        return encoded;
    }

    // RLE mode: Use improved run-length encoding if requested
    if (options.compressionLevel >= 8) {
        std::vector<size_t> encoded;
        // Improved RLE: [value, count] pairs, marker for long runs
        encoded.push_back(imageData.size());
        size_t i = 0;
        while (i < imageData.size()) {
            unsigned char value = imageData[i];
            size_t count = 1;
            while (i + count < imageData.size() && imageData[i + count] == value && count < 65535) {
                count++;
            }
            encoded.push_back((size_t)value << 16 | (count & 0xFFFF));
            i += count;
        }
        return encoded;
    }

    // Simple SDR (legacy, default)
    std::vector<size_t> encoded;
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

    // Detect encoding mode by header
    if (indices.size() > 5 && indices[2] == 8 && indices[3] == 8) {
        // HiFi SDR mode (block-based DCT)
        int width = (int)indices[0];
        int height = (int)indices[1];
        int blockSize = (int)indices[2];
        int quantLevel = (int)indices[3];
        int topCoeffs = (int)indices[4];
        int N = blockSize;
        std::vector<float> imgF(width * height, 0.0f);
        std::vector<unsigned char> imgU(width * height, 0);
        // Each set of 4: bx, by, idx, qval
        for (size_t i = 5; i + 3 < indices.size(); i += 4) {
            int bx = (int)indices[i];
            int by = (int)indices[i + 1];
            int idx = (int)indices[i + 2];
            int qval = (int)indices[i + 3];
            float dctCoeffs[64] = {0};
            dctCoeffs[idx] = qval * 1024.0f / ((1 << quantLevel) - 1);
            // Inverse DCT for block
            float block[64];
            const float PI = 3.14159265358979323846f;
            for (int x = 0; x < N; ++x) {
                for (int y = 0; y < N; ++y) {
                    float sum = 0.0f;
                    for (int u = 0; u < N; ++u) {
                        for (int v = 0; v < N; ++v) {
                            float alphaU = (u == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
                            float alphaV = (v == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
                            sum += alphaU * alphaV * dctCoeffs[u * N + v] *
                                cosf(PI * (2 * x + 1) * u / (2 * N)) *
                                cosf(PI * (2 * y + 1) * v / (2 * N));
                        }
                    }
                    block[x * N + y] = sum;
                }
            }
            for (int y = 0; y < N; ++y) {
                for (int x = 0; x < N; ++x) {
                    int ix = bx * N + x;
                    int iy = by * N + y;
                    int idx2 = iy * width + ix;
                    if (ix < width && iy < height && idx2 < (int)imgF.size()) imgF[idx2] += block[y * N + x];
                }
            }
        }
        for (int i = 0; i < width * height; ++i) imgU[i] = (unsigned char)std::max(0.0f, std::min(255.0f, roundf(imgF[i])));
        writeImageFile(outputPath, imgU);
        return;
    }

    // RLE mode
    if (indices.size() > 1 && (indices[1] >> 16) > 0) {
        size_t imageSize = indices[0];
        std::vector<unsigned char> decoded;
        decoded.reserve(imageSize);
        for (size_t k = 1; k < indices.size(); ++k) {
            unsigned char value = (unsigned char)((indices[k] >> 16) & 0xFF);
            size_t count = indices[k] & 0xFFFF;
            for (size_t i = 0; i < count; ++i) {
                decoded.push_back(value);
            }
        }
        if (decoded.size() > imageSize) decoded.resize(imageSize);
        writeImageFile(outputPath, decoded);
        return;
    }

    // Simple SDR (legacy)
    size_t imageSize = indices[0];
    std::vector<unsigned char> decoded(imageSize, 0);
    constexpr unsigned char quantLevels = 16;
    for (size_t k = 1; k < indices.size(); ++k) {
        size_t idx = indices[k];
        size_t pos = idx / quantLevels;
        unsigned char quantized = static_cast<unsigned char>(idx % quantLevels);
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