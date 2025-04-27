#ifndef AUDIO_ENCODING_HPP
#define AUDIO_ENCODING_HPP

#include <string>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cstdint>

/**
 * @class AudioEncoding
 * @brief Handles encoding and decoding of audio content with support for multiple formats
 * 
 * This class provides functionality to encode audio data into compressed indices
 * and decode those indices back into audio samples. It supports various audio
 * formats and provides quality control options.
 */
class AudioEncoding {
public:
    /**
     * @brief Supported audio encoding formats
     */
    enum class Format {
        PCM,    ///< Uncompressed PCM audio
        MP3,    ///< MP3 compressed audio
        AAC,    ///< Advanced Audio Coding
        FLAC    ///< Free Lossless Audio Codec
    };

    /**
     * @brief Audio quality settings for compression
     */
    struct QualitySettings {
        uint32_t sampleRate{44100};     ///< Samples per second
        uint8_t bitDepth{16};           ///< Bits per sample
        uint8_t channels{2};            ///< Number of audio channels
        float compressionRatio{0.8f};   ///< Target compression ratio (0.0-1.0)
    };

    /**
     * @brief Default constructor
     */
    AudioEncoding() = default;

    /**
     * @brief Constructor with quality settings
     * @param settings Initial quality settings for encoding/decoding
     */
    explicit AudioEncoding(const QualitySettings& settings);

    /**
     * @brief Virtual destructor
     */
    virtual ~AudioEncoding() = default;

    /**
     * @brief Encodes audio samples into compressed indices
     * @param audioData Vector of float samples in range [-1.0, 1.0]
     * @param format Target encoding format
     * @return Vector of indices representing the encoded audio
     * @throws std::invalid_argument if input data is invalid
     * @throws std::runtime_error if encoding fails
     */
    virtual std::vector<size_t> encode(
        const std::vector<float>& audioData,
        Format format = Format::PCM
    ) const;

    /**
     * @brief Decodes indices back into audio samples
     * @param indices Vector of encoded indices
     * @param format Source format of the encoded data
     * @return Vector of float samples in range [-1.0, 1.0]
     * @throws std::invalid_argument if indices are invalid
     * @throws std::runtime_error if decoding fails
     */
    virtual std::vector<float> decode(
        const std::vector<size_t>& indices,
        Format format = Format::PCM
    ) const;

    /**
     * @brief Updates the quality settings
     * @param settings New quality settings to use
     * @throws std::invalid_argument if settings are invalid
     */
    void updateQualitySettings(const QualitySettings& settings);

    /**
     * @brief Gets the current quality settings
     * @return Current quality settings
     */
    QualitySettings getQualitySettings() const;

    /**
     * @brief Gets the name of a format as a string
     * @param format Format to get the name of
     * @return String representation of the format
     */
    static std::string getFormatName(Format format);

    /**
     * @brief Checks if a format is supported
     * @param format Format to check
     * @return true if format is supported, false otherwise
     */
    static bool isFormatSupported(Format format);

    // Prevent copying
    AudioEncoding(const AudioEncoding&) = delete;
    AudioEncoding& operator=(const AudioEncoding&) = delete;

    // Allow moving
    AudioEncoding(AudioEncoding&&) noexcept = default;
    AudioEncoding& operator=(AudioEncoding&&) noexcept = default;

protected:
    /**
     * @brief Validates audio content before processing
     * @param content Content to validate
     * @throws std::invalid_argument if content is invalid
     */
    virtual void validateContent(const std::vector<float>& content) const;

private:
    // Format-specific encoding implementations
    std::vector<size_t> encodePCM(const std::vector<float>& audioData) const;
    std::vector<size_t> encodeMP3(const std::vector<float>& audioData) const;
    std::vector<size_t> encodeAAC(const std::vector<float>& audioData) const;
    std::vector<size_t> encodeFLAC(const std::vector<float>& audioData) const;

    // Format-specific decoding implementations
    std::vector<float> decodePCM(const std::vector<size_t>& indices) const;
    std::vector<float> decodeMP3(const std::vector<size_t>& indices) const;
    std::vector<float> decodeAAC(const std::vector<size_t>& indices) const;
    std::vector<float> decodeFLAC(const std::vector<size_t>& indices) const;

    QualitySettings settings_; ///< Current quality settings
};

#endif // AUDIO_ENCODING_HPP
