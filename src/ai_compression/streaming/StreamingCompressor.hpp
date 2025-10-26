/**
 * @file StreamingCompressor.hpp
 * @brief Streaming compression handler for large neural network models
 * 
 * This header defines the StreamingCompressor class which provides efficient
 * streaming compression for large neural network models that cannot fit
 * entirely in memory. Implements thread-safe streaming with proper
 * synchronization and archive format management.
 * 
 * Key Features:
 * - Memory-efficient streaming compression for large models
 * - Thread-safe concurrent segment processing
 * - Archive format management with proper headers and indexing
 * - Real-time compression statistics and progress tracking
 * - Automatic resource management and cleanup
 * - Support for multi-gigabyte model compression
 */

#ifndef STREAMING_COMPRESSOR_HPP
#define STREAMING_COMPRESSOR_HPP

#include "../core/AICompressor.hpp"
#include "../core/ArchiveConstants.hpp" // Include for MAGIC and VERSION
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

namespace CortexAICompression {

/**
 * @brief Streaming compression handler for memory-efficient model compression
 * 
 * Implements streaming compression that processes model segments as they become
 * available without loading the entire model into memory. Provides thread-safe
 * operations and maintains compression statistics for progress monitoring.
 */
class StreamingCompressor : public ICompressionHandler {
public:
    /**
     * @brief Constructor initializing streaming compression to output file
     * @param outputPath Path to output compressed archive file
     * @throws std::runtime_error if output file cannot be created
     */
    explicit StreamingCompressor(const std::string& outputPath);
    
    /**
     * @brief Destructor ensuring proper resource cleanup and file finalization
     */
    ~StreamingCompressor() override;

    /**
     * @brief Handle compressed segment data in streaming fashion
     * @param header Compressed segment metadata and information
     * @param compressedData Compressed segment data bytes
     * 
     * Thread-safe method for processing compressed segments as they become
     * available. Writes data to output stream and updates statistics.
     */
    void handleCompressedSegment(const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData) override;

    /**
     * @brief Get total compressed data size processed so far
     * @return Total size in bytes of all compressed segments
     */
    size_t getTotalCompressedSize() const { return totalCompressedSize_; }
    
    /**
     * @brief Get total original data size before compression
     * @return Total size in bytes of all original segments
     */
    size_t getTotalOriginalSize() const { return totalOriginalSize_; }

    /**
     * @brief Finalize archive writing and close output file
     * 
     * Completes the archive format by writing segment index, updating
     * headers, and ensuring all data is properly flushed to disk.
     * Must be called after all segments have been processed.
     */
    void finalizeArchive();

private:
    std::ofstream outputFile_;                  ///< Output file stream for compressed archive
    std::mutex writeMutex_;                     ///< Thread synchronization for concurrent access
    size_t totalCompressedSize_ = 0;            ///< Running total of compressed data size
    size_t totalOriginalSize_ = 0;              ///< Running total of original data size
    uint64_t currentDataOffset_ = 0;            ///< Current write position in output file

    /**
     * @brief Storage for segment headers and their file positions
     * 
     * Maintains mapping of segment headers to their byte offsets in the
     * output file for efficient random access during decompression.
     */
    std::vector<std::pair<CompressedSegmentHeader, uint64_t>> segment_headers_;
};

} // namespace CortexAICompression

#endif // STREAMING_COMPRESSOR_HPP
