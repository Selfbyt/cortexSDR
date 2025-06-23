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

// Interface for handling decompressed segments (e.g., loading into memory, passing to model runtime)
class ISegmentHandler {
public:
    virtual ~ISegmentHandler() = default;
    // Called when a segment has been fully decompressed.
    // Takes ownership of the decompressed data.
    virtual void handleSegment(ModelSegment segment) = 0;
};

class AIDecompressor {
public:
    AIDecompressor();

    // Registers a decompression strategy for a specific strategy ID.
    void registerStrategy(uint8_t strategy_id, std::shared_ptr<ICompressionStrategy> strategy);

    // Reads the archive from the input stream and decompresses segments,
    // passing them to the provided handler. Designed for streaming.
    // Throws CompressionError on failure or format errors.
    void decompressModelStream(std::istream& inputArchiveStream, ISegmentHandler& handler);

    // New method for on-demand segment decompression from a file path and a known header/info
    ModelSegment decompressSegment(const std::string& archivePath, const CompressedSegmentHeader& segmentInfo, uint64_t offset);

    // Optional: Method to read just the archive index/headers first
    std::vector<CompressedSegmentHeader> readArchiveHeaders(std::istream& inputArchiveStream);

    // Optional: Method to decompress a specific named segment on demand
    // ModelSegment decompressSegmentByName(std::istream& inputArchiveStream, const std::string& segmentName);

private:
    std::map<uint8_t, std::shared_ptr<ICompressionStrategy>> strategyMap_;

    // Helper to read the archive header/index (implementation needed)
    std::vector<CompressedSegmentHeader> readArchiveIndex(std::istream& stream);

    // Helper to read and decompress a single segment based on its header
    ModelSegment readAndDecompressSegment(std::istream& stream, const CompressedSegmentHeader& header);
};

} // namespace CortexAICompression

#endif // AI_DECOMPRESSOR_HPP
