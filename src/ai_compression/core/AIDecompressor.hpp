/**
 * @file AIDecompressor.hpp
 * @brief Public API for streaming and on-demand decompression of AI archives.
 */
#ifndef AI_DECOMPRESSOR_HPP
#define AI_DECOMPRESSOR_HPP

#include "../strategies/CompressionStrategy.hpp"
#include "ModelSegment.hpp"
#include "AICompressor.hpp" // Include for CompressedSegmentHeader
#include <string>
#include <vector>
#include <memory> // For std::unique_ptr, std::shared_ptr
#include <istream> // For reading from input stream
#include <map>     // For mapping strategy IDs back to strategies

namespace CortexAICompression {

/**
 * @brief Interface for handling decompressed segments.
 * @details Allows clients to consume segments as they are produced during
 * streaming decompression (e.g., to load into a runtime or further process).
 */
class ISegmentHandler {
public:
    virtual ~ISegmentHandler() = default;
    /**
     * @brief Called when a segment has been fully decompressed.
     * @param segment Decompressed segment; ownership transferred to the callee.
     */
    virtual void handleSegment(ModelSegment segment) = 0;
};

class AIDecompressor {
public:
    AIDecompressor();

    /** Register a decompression strategy for a specific strategy ID. */
    void registerStrategy(uint8_t strategy_id, std::shared_ptr<ICompressionStrategy> strategy);

    /**
     * @brief Stream and decompress an archive from an input stream.
     * @param inputArchiveStream Source archive stream.
     * @param handler Sink to receive decompressed segments.
     * @throws CompressionError On format/strategy errors.
     */
    void decompressModelStream(std::istream& inputArchiveStream, ISegmentHandler& handler);

    /**
     * @brief Decompress a single segment on-demand from a file path.
     * @param archivePath Path to archive.
     * @param segmentInfo Header for the desired segment.
     * @param offset Byte offset to the first segment payload.
     * @return Decompressed segment with metadata and bytes.
     */
    ModelSegment decompressSegment(const std::string& archivePath, const CompressedSegmentHeader& segmentInfo, uint64_t offset);

    /**
     * @brief Read a segment's compressed payload without decompressing.
     * @param archivePath Path to archive.
     * @param segmentInfo Header for the desired segment.
     * @param offset Byte offset to the first segment payload.
     * @return Compressed bytes for the segment payload.
     */
    std::vector<std::byte> readCompressedBytes(const std::string& archivePath, const CompressedSegmentHeader& segmentInfo, uint64_t offset);

    /**
     * @brief Read archive headers (index) without payloads.
     */
    std::vector<CompressedSegmentHeader> readArchiveHeaders(std::istream& inputArchiveStream);

    // Optional: Method to decompress a specific named segment on demand
    // ModelSegment decompressSegmentByName(std::istream& inputArchiveStream, const std::string& segmentName);

private:
    std::map<uint8_t, std::shared_ptr<ICompressionStrategy>> strategyMap_;

    /** Read the archive index from the stream. */
    std::vector<CompressedSegmentHeader> readArchiveIndex(std::istream& stream);

    /** Read and decompress a single segment from the stream. */
    ModelSegment readAndDecompressSegment(std::istream& stream, const CompressedSegmentHeader& header);
};

} // namespace CortexAICompression

#endif // AI_DECOMPRESSOR_HPP
