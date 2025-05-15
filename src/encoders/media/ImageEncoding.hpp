#ifndef IMAGE_ENCODING_HPP
#define IMAGE_ENCODING_HPP

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

/**
 * @brief Class for encoding and decoding images in various formats
 * 
 * Supports multiple image formats (PNG, JPEG, BMP) and provides
 * functionality for compression, format conversion, and basic
 * image manipulation.
 */
class ImageEncoding {
public:
    /**
     * @brief Supported image formats
     */
    enum class Format {
        PNG,
        JPEG,
        BMP,
        WEBP,
        AUTO  // Automatically detect format
    };

    /**
     * @brief Compression quality settings
     */
    struct CompressionOptions {
        int quality{90};          // 0-100, higher is better quality
        bool useCompression{true};
        int compressionLevel{6};  // 0-9, higher is more compression
    };

    /**
     * @brief Constructor
     * @param defaultFormat Default format for encoding (default: AUTO)
     */
    explicit ImageEncoding(Format defaultFormat = Format::AUTO);

    /**
     * @brief Destructor
     */
    ~ImageEncoding();

    /**
     * @brief Encodes an image file into a vector of indices
     * @param imagePath Path to the image file
     * @param format Output format (default: AUTO)
     * @param options Compression options
     * @return Vector of encoded indices
     * @throws std::runtime_error if encoding fails
     */
    std::vector<size_t> encodeImage(
        const std::string& imagePath,
        Format format,
        const CompressionOptions& options
    ) const;

    /**
     * @brief Decodes indices back into an image file
     * @param indices Vector of encoded indices
     * @param outputPath Path where to save the decoded image
     * @param format Output format
     * @throws std::runtime_error if decoding fails
     */
    void decodeIndices(
        const std::vector<size_t>& indices,
        const std::string& outputPath,
        Format format = Format::PNG
    ) const;

    /**
     * @brief Converts image format
     * @param inputPath Input image path
     * @param outputPath Output image path
     * @param format Target format
     * @param options Compression options
     * @throws std::runtime_error if conversion fails
     */
    // NOTE: No change needed here as 'format' already lacked a default.
    // The previous error message might have been slightly misleading,
    // but the core issue is the rule about default arguments order.
    // Removing the default from encodeImage's format parameter should suffice.
    // Keeping this block just for clarity that convertFormat is checked.
    void convertFormat(
        const std::string& inputPath,
        const std::string& outputPath,
        Format format,
        const CompressionOptions& options
    ) const;

    /**
     * @brief Gets the format of an image file
     * @param imagePath Path to the image file
     * @return Detected image format
     * @throws std::runtime_error if format detection fails
     */
    Format detectFormat(const std::string& imagePath) const;

    /**
     * @brief Validates if an image file is properly formatted
     * @param imagePath Path to the image file
     * @return True if valid, false otherwise
     */
    bool validateImage(const std::string& imagePath) const;

    /**
     * @brief Gets image dimensions
     * @param imagePath Path to the image file
     * @return Pair of width and height
     * @throws std::runtime_error if reading dimensions fails
     */
    std::pair<size_t, size_t> getImageDimensions(const std::string& imagePath) const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;  // Private implementation (PIMPL idiom)

    // Helper methods
    bool isValidPath(const std::string& path) const;
    std::vector<unsigned char> readImageFile(const std::string& path) const;
    void writeImageFile(const std::string& path, const std::vector<unsigned char>& data) const;
    Format defaultFormat;

    // Prevent copying and assignment
    ImageEncoding(const ImageEncoding&) = delete;
    ImageEncoding& operator=(const ImageEncoding&) = delete;
};

/**
 * @brief Custom exception class for image encoding errors
 */
class ImageEncodingError : public std::runtime_error {
public:
    explicit ImageEncodingError(const std::string& message)
        : std::runtime_error(message) {}
};

#endif // IMAGE_ENCODING_HPP
