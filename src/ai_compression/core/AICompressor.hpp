#ifndef AI_COMPRESSOR_HPP
#define AI_COMPRESSOR_HPP

#include "AIModelParser.hpp"
#include "../strategies/CompressionStrategy.hpp"
#include "ModelSegment.hpp"
#include <string>
#include <vector>
#include <memory> // For std::unique_ptr, std::shared_ptr
#include <ostream> // For writing to output stream
#include <map>     // For mapping segment types to strategies
#include <list>    // For storing multiple strategies per type
#include <future>  // For async compression

namespace CortexAICompression {

// Structure to hold compression statistics
struct CompressionStats {
    size_t originalSize = 0;
    size_t compressedSize = 0;
    double compressionRatio = 0.0;
    double compressionTimeMs = 0.0;
};

// Represents the compressed archive format structure
struct CompressedSegmentHeader {
    SegmentType original_type;
    uint8_t compression_strategy_id; // ID mapping to the strategy used
    uint64_t original_size;
    uint64_t compressed_size;
    std::string name;
    std::optional<TensorMetadata> tensor_metadata; // Added tensor metadata
    std::string layer_name;                       // Added layer info
    size_t layer_index;
};

// Interface for handling compressed segments during streaming
class ICompressionHandler {
public:
    virtual ~ICompressionHandler() = default;
    virtual void handleCompressedSegment(const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData) = 0;
};

class AICompressor {
public:
    // Constructor takes the parser to use.
    // Strategies can be added or configured.
    explicit AICompressor(std::unique_ptr<IAIModelParser> parser);

    // Registers a compression strategy for a specific segment type with a given priority.
    // Lower priority values are tried first.
    // Uses shared_ptr to allow multiple segments to potentially use the same strategy instance.
    void registerStrategy(SegmentType type, int priority, uint8_t strategy_id, std::shared_ptr<ICompressionStrategy> strategy);

    // Compresses the model file and writes the bundled archive to the output stream.
    // Throws ParsingError or CompressionError on failure.
    void compressModel(const std::string& modelPath, std::ostream& outputArchiveStream);

    // New: Compress model with chunking and streaming
    void compressModelStreaming(const std::string& modelPath, ICompressionHandler& handler);

    // New: Set the number of compression threads for parallel processing
    void setCompressionThreads(size_t numThreads) { numThreads_ = numThreads; }

    // Get compression statistics
    const CompressionStats& getCompressionStats() const { return stats_; }

private:
    struct StrategyInfo {
        int priority;
        uint8_t id;
        std::shared_ptr<ICompressionStrategy> strategy;

        // Comparison operator for sorting by priority
        bool operator<(const StrategyInfo& other) const {
            return priority < other.priority;
        }
    };

    std::unique_ptr<IAIModelParser> modelParser_;
    // Store a list of strategies for each type, sorted by priority
    std::map<SegmentType, std::list<StrategyInfo>> strategyMap_;
    std::shared_ptr<ICompressionStrategy> defaultStrategy_; // e.g., Gzip for unknown/metadata types
    uint8_t defaultStrategyId_;
    size_t numThreads_ = 1;  // Default to single-threaded
    CompressionStats stats_; // Store compression statistics

    // Helper to select the list of appropriate strategies for a segment, ordered by priority
    const std::list<StrategyInfo>* selectStrategies(SegmentType type) const;

    // Helper to write the archive header/index (implementation needed)
    void writeArchiveHeader(std::ostream& stream, const std::vector<CompressedSegmentHeader>& headers);

    // Helper to write a single compressed segment (header + data)
    void writeSegment(std::ostream& stream, const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData);

    // New: Helper for parallel compression of segments
    std::vector<std::pair<CompressedSegmentHeader, std::vector<std::byte>>> 
    compressSegmentsParallel(const std::vector<ModelSegment>& segments) const;
};

} // namespace CortexAICompression

#endif // AI_COMPRESSOR_HPP
