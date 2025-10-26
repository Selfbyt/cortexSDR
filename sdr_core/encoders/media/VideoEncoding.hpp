#ifndef VIDEO_ENCODING_HPP
#define VIDEO_ENCODING_HPP

#include <string>
#include <cstdint> 
#include <vector>
#include <stdexcept>

/**
 * @class VideoEncoding
 * @brief Handles encoding and decoding of video content
 * 
 * This class provides functionality to encode video content into indices
 * and decode those indices back into video content, as well as encoding/decoding
 * between different video formats.
 */
class VideoEncoding {
public:
    /**
     * @brief Supported video encoding formats
     */
    enum class Format {
        RAW,    ///< Raw video data
        H264,   ///< H.264/AVC encoding
        H265,   ///< H.265/HEVC encoding
        VP9,    ///< VP9 encoding
        AV1     ///< AV1 encoding
    };

    /**
     * @brief Default constructor
     */
    VideoEncoding() = default;

    /**
     * @brief Virtual destructor to ensure proper cleanup of derived classes
     */
    virtual ~VideoEncoding() = default;

    /**
     * @brief Encodes raw video content into a vector of indices
     * @param videoContent Raw video content to encode
     * @param format Target encoding format
     * @return Vector of indices representing the encoded video
     * @throws std::runtime_error if the content cannot be encoded
     */
    virtual std::vector<size_t> encode(
        const std::vector<uint8_t>& videoContent,
        Format format = Format::H264
    ) const;

    /**
     * @brief Decodes a vector of indices back into video content
     * @param indices Vector of indices to decode
     * @param format Format of the encoded content
     * @return Vector of bytes representing the decoded video content
     * @throws std::runtime_error if the indices cannot be decoded
     */
    virtual std::vector<uint8_t> decode(
        const std::vector<size_t>& indices,
        Format format = Format::H264
    ) const;

    /**
     * @brief Transcodes video content from one format to another
     * @param videoContent Input video content
     * @param sourceFormat Format of the input content
     * @param targetFormat Desired output format
     * @return Vector of bytes containing the transcoded video content
     * @throws std::runtime_error if transcoding fails
     */
    virtual std::vector<uint8_t> transcode(
        const std::vector<uint8_t>& videoContent,
        Format sourceFormat,
        Format targetFormat
    ) const;

    /**
     * @brief Get the name of a format as a string
     * @param format The format to get the name of
     * @return String representation of the format
     */
    static std::string getFormatName(Format format);

    /**
     * @brief Check if transcoding between two formats is supported
     * @param sourceFormat Source format to check
     * @param targetFormat Target format to check
     * @return true if transcoding is supported, false otherwise
     */
    static bool isTranscodingSupported(Format sourceFormat, Format targetFormat);

    // Prevent copying
    VideoEncoding(const VideoEncoding&) = delete;
    VideoEncoding& operator=(const VideoEncoding&) = delete;

    // Allow moving
    VideoEncoding(VideoEncoding&&) noexcept = default;
    VideoEncoding& operator=(VideoEncoding&&) noexcept = default;

protected:
    /**
     * @brief Validates video content before processing
     * @param content Content to validate
     * @throws std::invalid_argument if content is invalid
     */
    virtual void validateContent(const std::vector<uint8_t>& content) const;

private:
    /**
     * @brief Internal helper to perform format-specific encoding
     * @param content Content to encode
     * @param format Target format
     * @return Encoded indices
     */
    std::vector<size_t> encodeFormat(
        const std::vector<uint8_t>& content,
        Format format
    ) const;

    /**
     * @brief Internal helper to perform format-specific decoding
     * @param indices Indices to decode
     * @param format Source format
     * @return Decoded content
     */
    std::vector<uint8_t> decodeFormat(
        const std::vector<size_t>& indices,
        Format format
    ) const;

    std::vector<uint8_t> applyH264Preprocessing(const std::vector<uint8_t>& content) const;
    std::vector<uint8_t> applyH264Postprocessing(const std::vector<uint8_t>& content) const;
    std::vector<uint8_t> applyH265Preprocessing(const std::vector<uint8_t>& content) const;
    std::vector<uint8_t> applyH265Postprocessing(const std::vector<uint8_t>& content) const;
};

#endif // VIDEO_ENCODING_HPP