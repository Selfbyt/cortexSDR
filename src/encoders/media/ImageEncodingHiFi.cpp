#include "ImageEncodingHiFi.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cstring>

// Minimal PNG/BMP loading/saving for demonstration (replace with stb_image or similar for production)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {
// 2D DCT for 8x8 block (or NxN)
void dct2d(const float* in, float* out, int N) {
    const float PI = 3.14159265358979323846f;
    for (int u = 0; u < N; ++u) {
        for (int v = 0; v < N; ++v) {
            float sum = 0.0f;
            for (int x = 0; x < N; ++x) {
                for (int y = 0; y < N; ++y) {
                    sum += in[x * N + y] *
                        cosf(PI * (2 * x + 1) * u / (2 * N)) *
                        cosf(PI * (2 * y + 1) * v / (2 * N));
                }
            }
            float alphaU = (u == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
            float alphaV = (v == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
            out[u * N + v] = alphaU * alphaV * sum;
        }
    }
}

void idct2d(const float* in, float* out, int N) {
    const float PI = 3.14159265358979323846f;
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            float sum = 0.0f;
            for (int u = 0; u < N; ++u) {
                for (int v = 0; v < N; ++v) {
                    float alphaU = (u == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
                    float alphaV = (v == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
                    sum += alphaU * alphaV * in[u * N + v] *
                        cosf(PI * (2 * x + 1) * u / (2 * N)) *
                        cosf(PI * (2 * y + 1) * v / (2 * N));
                }
            }
            out[x * N + y] = sum;
        }
    }
}

// Helper: Clamp float to 0-255 and convert to uint8
inline uint8_t clamp255(float v) {
    return (uint8_t)std::max(0.0f, std::min(255.0f, roundf(v)));
}
} // namespace

ImageEncodingHiFi::EncodedImage ImageEncodingHiFi::encodeImageHiFi(const std::string& imagePath, const HiFiParams& params) const {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath.c_str(), &width, &height, &channels, 1); // grayscale for now
    if (!img) throw ImageEncodingHiFiError("Failed to load image: " + imagePath);

    int N = params.blockSize;
    int nBlocksX = (width + N - 1) / N;
    int nBlocksY = (height + N - 1) / N;
    EncodedImage encoded;
    encoded.width = width;
    encoded.height = height;
    encoded.blockSize = N;
    encoded.quantLevel = params.quantLevel;
    encoded.topCoeffs = params.topCoeffs;

    // For each block
    for (int by = 0; by < nBlocksY; ++by) {
        for (int bx = 0; bx < nBlocksX; ++bx) {
            float block[64] = {0};
            for (int y = 0; y < N; ++y) {
                for (int x = 0; x < N; ++x) {
                    int ix = bx * N + x;
                    int iy = by * N + y;
                    int idx = iy * width + ix;
                    block[y * N + x] = (ix < width && iy < height) ? img[idx] : 0.0f;
                }
            }
            float dctCoeffs[64];
            dct2d(block, dctCoeffs, N);
            // Quantize and select top-K
            std::vector<std::pair<int, int>> qcoeffs;
            std::vector<std::pair<float, int>> abs_coeffs;
            for (int i = 0; i < N * N; ++i) {
                abs_coeffs.push_back({fabsf(dctCoeffs[i]), i});
            }
            std::partial_sort(abs_coeffs.begin(), abs_coeffs.begin() + params.topCoeffs, abs_coeffs.end(), std::greater<>());
            for (int k = 0; k < params.topCoeffs; ++k) {
                int idx = abs_coeffs[k].second;
                float val = dctCoeffs[idx];
                int qval = (int)std::round(val * ((1 << params.quantLevel) - 1) / 1024.0f);
                if (qval != 0) qcoeffs.emplace_back(idx, qval);
            }
            if (!qcoeffs.empty()) {
                EncodedBlock blockinfo;
                blockinfo.x = bx;
                blockinfo.y = by;
                blockinfo.coeffs = std::move(qcoeffs);
                encoded.blocks.push_back(std::move(blockinfo));
            }
        }
    }
    stbi_image_free(img);
    return encoded;
}

void ImageEncodingHiFi::decodeImageHiFi(const EncodedImage& encoded, const std::string& outputPath) const {
    int width = encoded.width, height = encoded.height, N = encoded.blockSize;
    std::vector<float> imgF(width * height, 0.0f);
    std::vector<uint8_t> imgU(width * height, 0);

    // For each block
    for (const auto& blockinfo : encoded.blocks) {
        float dctCoeffs[64] = {0};
        for (const auto& p : blockinfo.coeffs) {
            int idx = p.first;
            int qval = p.second;
            dctCoeffs[idx] = qval * 1024.0f / ((1 << encoded.quantLevel) - 1);
        }
        float block[64];
        idct2d(dctCoeffs, block, N);
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int ix = blockinfo.x * N + x;
                int iy = blockinfo.y * N + y;
                int idx = iy * width + ix;
                if (ix < width && iy < height) imgF[idx] += block[y * N + x];
            }
        }
    }
    // Convert to uint8
    for (int i = 0; i < width * height; ++i) imgU[i] = clamp255(imgF[i]);
    stbi_write_png(outputPath.c_str(), width, height, 1, imgU.data(), width);
}

void ImageEncodingHiFi::saveEncodedImage(const EncodedImage& encoded, const std::string& filePath) const {
    std::ofstream ofs(filePath, std::ios::binary);
    if (!ofs) throw ImageEncodingHiFiError("Failed to open file for writing: " + filePath);
    ofs.write((char*)&encoded.width, sizeof(int));
    ofs.write((char*)&encoded.height, sizeof(int));
    ofs.write((char*)&encoded.blockSize, sizeof(int));
    ofs.write((char*)&encoded.quantLevel, sizeof(int));
    ofs.write((char*)&encoded.topCoeffs, sizeof(int));
    int nBlocks = (int)encoded.blocks.size();
    ofs.write((char*)&nBlocks, sizeof(int));
    for (const auto& blk : encoded.blocks) {
        ofs.write((char*)&blk.x, sizeof(int));
        ofs.write((char*)&blk.y, sizeof(int));
        int nCoeffs = (int)blk.coeffs.size();
        ofs.write((char*)&nCoeffs, sizeof(int));
        for (const auto& p : blk.coeffs) {
            ofs.write((char*)&p.first, sizeof(int));
            ofs.write((char*)&p.second, sizeof(int));
        }
    }
}

ImageEncodingHiFi::EncodedImage ImageEncodingHiFi::loadEncodedImage(const std::string& filePath) const {
    std::ifstream ifs(filePath, std::ios::binary);
    if (!ifs) throw ImageEncodingHiFiError("Failed to open file for reading: " + filePath);
    EncodedImage encoded;
    ifs.read((char*)&encoded.width, sizeof(int));
    ifs.read((char*)&encoded.height, sizeof(int));
    ifs.read((char*)&encoded.blockSize, sizeof(int));
    ifs.read((char*)&encoded.quantLevel, sizeof(int));
    ifs.read((char*)&encoded.topCoeffs, sizeof(int));
    int nBlocks;
    ifs.read((char*)&nBlocks, sizeof(int));
    encoded.blocks.resize(nBlocks);
    for (int i = 0; i < nBlocks; ++i) {
        EncodedBlock blk;
        ifs.read((char*)&blk.x, sizeof(int));
        ifs.read((char*)&blk.y, sizeof(int));
        int nCoeffs;
        ifs.read((char*)&nCoeffs, sizeof(int));
        blk.coeffs.resize(nCoeffs);
        for (int j = 0; j < nCoeffs; ++j) {
            ifs.read((char*)&blk.coeffs[j].first, sizeof(int));
            ifs.read((char*)&blk.coeffs[j].second, sizeof(int));
        }
        encoded.blocks[i] = std::move(blk);
    }
    return encoded;
}
