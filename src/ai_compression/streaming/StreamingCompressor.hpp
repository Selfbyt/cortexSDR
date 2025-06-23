#ifndef STREAMING_COMPRESSOR_HPP
#define STREAMING_COMPRESSOR_HPP

#include "../core/AICompressor.hpp"
#include "../core/ArchiveConstants.hpp" // Include for MAGIC and VERSION
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>
#include <utility> // For std::pair

namespace CortexAICompression {

// Handles streaming compression to a file with proper synchronization
class StreamingCompressor : public ICompressionHandler {
public:
    explicit StreamingCompressor(const std::string& outputPath);
    ~StreamingCompressor() override;

    // Implements ICompressionHandler
    void handleCompressedSegment(const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData) override;

    // Get compression statistics
    size_t getTotalCompressedSize() const { return totalCompressedSize_; }
    size_t getTotalOriginalSize() const { return totalOriginalSize_; }
    // Note: getCompressionRatio removed, calculated after finalization if needed

    // Finalize the archive writing process
    void finalizeArchive();

private:
    std::ofstream outputFile_;
    std::mutex writeMutex_; // Mutex might still be needed if handleCompressedSegment is called concurrently
    size_t totalCompressedSize_ = 0;
    size_t totalOriginalSize_ = 0;
    uint64_t currentDataOffset_ = 0;

    // Storage for segment headers and their file offsets
    std::vector<std::pair<CompressedSegmentHeader, uint64_t>> segment_headers_;
};

} // namespace CortexAICompression

#endif // STREAMING_COMPRESSOR_HPP
