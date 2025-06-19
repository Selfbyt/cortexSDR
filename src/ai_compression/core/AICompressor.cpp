#include "AICompressor.hpp"
#include "../strategies/CompressionStrategy.hpp"
#include "ArchiveConstants.hpp"
#include <fstream> // For file stream operations if needed, though using ostream directly
#include <stdexcept>
#include <iostream> // For potential debug/error messages
#include <vector>
#include <cstring> // For std::memcpy
#include <thread>
#include <future>
#include <list>    // Include list for strategy storage
#include <algorithm> // For std::find_if

namespace CortexAICompression {

// Helper function to write basic types to the stream
template<typename T>
void writeBasicType(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Helper function to write string (length prefixed)
void writeString(std::ostream& stream, const std::string& str) {
    uint16_t len = static_cast<uint16_t>(str.length());
    if (str.length() > UINT16_MAX) {
        throw std::runtime_error("Segment name too long for archive format.");
    }
    writeBasicType(stream, len);
    stream.write(str.data(), len);
}

AICompressor::AICompressor(std::unique_ptr<IAIModelParser> parser)
    : modelParser_(std::move(parser)), defaultStrategyId_(0) // 0 = No compression / placeholder
{
    // Optionally register a default strategy like Gzip here for metadata, etc.
    // defaultStrategy_ = std::make_shared<GzipStrategy>();
    // defaultStrategyId_ = 1; // Assuming 1 is Gzip ID
}

// Updated function signature to include priority
void AICompressor::registerStrategy(SegmentType type, int priority, uint8_t strategy_id, std::shared_ptr<ICompressionStrategy> strategy) {
    if (!strategy) {
        throw std::invalid_argument("Strategy cannot be null.");
    }
    // Use the passed 'priority' parameter
    StrategyInfo info = {priority, strategy_id, std::move(strategy)};
    auto& strategyList = strategyMap_[type];
    
    // Insert the new strategy while maintaining sorted order by priority
    auto it = std::lower_bound(strategyList.begin(), strategyList.end(), info);
    strategyList.insert(it, std::move(info));
}

// Helper to select the list of appropriate strategies for a segment, ordered by priority
const std::list<AICompressor::StrategyInfo>* AICompressor::selectStrategies(SegmentType type) const {
    auto it = strategyMap_.find(type);
    if (it != strategyMap_.end()) {
        return &(it->second); // Return pointer to the list
    }
    return nullptr; // No strategies registered for this type
}


void AICompressor::compressModel(const std::string& modelPath, std::ostream& outputArchiveStream) {
    // NOTE: This non-streaming version needs updating to use the new strategy selection logic
    // if it's intended to be used. For now, focusing on the streaming version used by the C API.
    // Consider deprecating or updating this method.
    // Reset statistics
    stats_ = CompressionStats();
    auto startTime = std::chrono::high_resolution_clock::now();
    if (!modelParser_) {
        throw std::runtime_error("AICompressor: No model parser provided.");
    }
    if (!outputArchiveStream) {
         throw std::runtime_error("AICompressor: Output stream is invalid.");
    }

    // 1. Parse the model into segments
    std::vector<ModelSegment> segments = modelParser_->parse(modelPath);
    if (segments.empty()) {
        // Handle case of empty model or parsing failure that didn't throw
        std::cerr << "Warning: Model parsing resulted in zero segments for " << modelPath << std::endl;
        // Write minimal valid archive? Or throw? For now, write minimal.
    }

    // 2. Compress each segment and gather header info
    std::vector<CompressedSegmentHeader> headers;
    std::vector<std::vector<std::byte>> compressedDatas;
    headers.reserve(segments.size());
    compressedDatas.reserve(segments.size());

    for (const auto& segment : segments) {
        CompressedSegmentHeader header;
        header.name = segment.name;
        header.original_type = segment.type;
        header.original_size = static_cast<uint64_t>(segment.original_size);

        const auto* strategies = selectStrategies(segment.type);
        std::vector<std::byte> compressedData;
        bool compressionSuccessful = false;

        if (strategies && !strategies->empty()) {
            for (const auto& stratInfo : *strategies) {
                try {
                    compressedData = stratInfo.strategy->compress(segment);
                    // If compress succeeds without throwing CompressionError, we're done for this segment
                    header.compression_strategy_id = stratInfo.id;
                    compressionSuccessful = true;
                     std::cerr << "Successfully compressed segment '" << segment.name
                               << "' with strategy ID " << static_cast<int>(stratInfo.id)
                               << " (Priority: " << stratInfo.priority << ")" << std::endl;
                    break; // Exit the strategy loop for this segment
                } catch (const CompressionError& e) {
                    std::cerr << "Warning: Compression failed for segment '" << segment.name
                              << "' with strategy ID " << static_cast<int>(stratInfo.id)
                              << " (Priority: " << stratInfo.priority << "): " << e.what()
                              << ". Trying next strategy..." << std::endl;
                    // Continue to the next strategy in the list
                }
            }
        }

        // If no strategy succeeded or no strategies were registered
        if (!compressionSuccessful) {
             std::cerr << "Warning: All strategies failed for segment '" << segment.name
                       << "'. Storing uncompressed." << std::endl;
            header.compression_strategy_id = 0; // Mark as uncompressed
            compressedData = segment.data; // Store original data
        }

        header.compressed_size = static_cast<uint64_t>(compressedData.size());
        stats_.originalSize += header.original_size;
        stats_.compressedSize += header.compressed_size;
        headers.push_back(header);
        compressedDatas.push_back(std::move(compressedData));
    }

    // 3. Write archive header and index
    // Write Magic Number & Version
    outputArchiveStream.write(ARCHIVE_MAGIC, sizeof(ARCHIVE_MAGIC));
    writeBasicType(outputArchiveStream, ARCHIVE_VERSION);

    // Write Number of Segments
    uint32_t numSegments = static_cast<uint32_t>(headers.size());
    writeBasicType(outputArchiveStream, numSegments);

    // Calculate initial offset for the first data block (after magic, version, count, and index table)
    uint64_t currentOffset = sizeof(ARCHIVE_MAGIC) + sizeof(ARCHIVE_VERSION) + sizeof(numSegments);
    // Calculate size of the index table itself
    uint64_t indexTableSize = 0;
    for (const auto& header : headers) {
        indexTableSize += sizeof(uint16_t) + header.name.length(); // Name length + name
        indexTableSize += sizeof(uint8_t); // Original Type
        indexTableSize += sizeof(uint8_t); // Strategy ID
        indexTableSize += sizeof(uint64_t); // Original Size
        indexTableSize += sizeof(uint64_t); // Compressed Size
        indexTableSize += sizeof(uint64_t); // Offset
    }
    currentOffset += indexTableSize;

    // Write Index Table Entries
    std::vector<uint64_t> dataOffsets;
    dataOffsets.reserve(headers.size());
    for (size_t i = 0; i < headers.size(); ++i) {
        dataOffsets.push_back(currentOffset);
        currentOffset += headers[i].compressed_size; // Update offset for the *next* block

        writeString(outputArchiveStream, headers[i].name);
        writeString(outputArchiveStream, segments[i].layer_type);
        uint8_t type_val = static_cast<uint8_t>(headers[i].original_type);
        writeBasicType(outputArchiveStream, type_val);
        writeBasicType(outputArchiveStream, headers[i].compression_strategy_id);
        writeBasicType(outputArchiveStream, headers[i].original_size);
        writeBasicType(outputArchiveStream, headers[i].compressed_size);
        writeBasicType(outputArchiveStream, dataOffsets[i]); // Write calculated offset
    }

    // 4. Write Compressed Data Blocks
    for (const auto& data : compressedDatas) {
        outputArchiveStream.write(reinterpret_cast<const char*>(data.data()), data.size());
    }

    // 5. Write the archive footer (if any)
    // For now, just flush the stream
    outputArchiveStream.flush();
    
    // Update compression statistics
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    stats_.compressionTimeMs = duration.count();
    stats_.compressionRatio = stats_.originalSize > 0 ? static_cast<double>(stats_.originalSize) / stats_.compressedSize : 0.0;
}

// Helper for sequential compression of a single segment
std::pair<CompressedSegmentHeader, std::vector<std::byte>>
AICompressor::compressSegment(const ModelSegment& segment) const {
    CompressedSegmentHeader header;
    header.name = segment.name;
    header.original_type = segment.type;
    header.original_size = static_cast<uint64_t>(segment.original_size);
    header.tensor_metadata = segment.tensor_metadata;
    header.layer_name = segment.layer_name;
    header.layer_index = segment.layer_index;

    const auto* strategies = selectStrategies(segment.type);
    std::vector<std::byte> compressedData;
    bool compressionSuccessful = false;
    uint8_t winningStrategyId = 0;

    if (strategies && !strategies->empty()) {
        for (const auto& stratInfo : *strategies) {
            try {
                compressedData = stratInfo.strategy->compress(segment);
                winningStrategyId = stratInfo.id;
                compressionSuccessful = true;
                // Log success
                std::cerr << "Success: Segment '" << segment.name << "', Strategy ID " 
                          << static_cast<int>(stratInfo.id) << std::endl;
                break;
            } catch (const CompressionError& e) {
                // Log failure 
                std::cerr << "Warning: Segment '" << segment.name << "', Strategy ID " 
                          << static_cast<int>(stratInfo.id) << " failed: " << e.what() << std::endl;
            }
        }
    }

    if (!compressionSuccessful) {
        std::cerr << "Warning: All strategies failed for segment '" << segment.name 
                  << "'. Storing uncompressed." << std::endl;
        winningStrategyId = 0; // Mark as uncompressed
        compressedData = segment.data; // Store original data
    }

    header.compression_strategy_id = winningStrategyId;
    header.compressed_size = static_cast<uint64_t>(compressedData.size());
    return std::make_pair(header, std::move(compressedData));
}

// New: Helper for memory-efficient parallel compression of segments
std::vector<std::pair<CompressedSegmentHeader, std::vector<std::byte>>> 
AICompressor::compressSegmentsParallel(const std::vector<ModelSegment>& segments) const {
    std::vector<std::pair<CompressedSegmentHeader, std::vector<std::byte>>> results;
    
    // Sort segments by size - process smallest segments first to reduce memory footprint
    std::vector<std::reference_wrapper<const ModelSegment>> sortedSegments;
    for (const auto& segment : segments) {
        sortedSegments.push_back(std::ref(segment));
    }
    
    std::sort(sortedSegments.begin(), sortedSegments.end(), 
              [](const ModelSegment& a, const ModelSegment& b) {
                  return a.data.size() < b.data.size();
              });
    
    // Define size thresholds for different processing approaches
    const size_t LARGE_SEGMENT_THRESHOLD = 100 * 1024 * 1024; // 100MB
    const size_t MAX_BATCH_SIZE = 5; // Max segments to process in parallel
    
    // 1. Process large segments sequentially to avoid OOM
    std::vector<std::reference_wrapper<const ModelSegment>> regularSegments;
    
    for (const auto& segmentRef : sortedSegments) {
        const ModelSegment& segment = segmentRef.get();
        
        if (segment.data.size() >= LARGE_SEGMENT_THRESHOLD) {
            std::cerr << "Processing large segment '" << segment.name << "' sequentially (" 
                      << (segment.data.size() / (1024 * 1024)) << " MB)" << std::endl;
            
            // Process large segment sequentially
            auto result = compressSegment(segment);
            results.push_back(std::move(result));
        } else {
            regularSegments.push_back(segmentRef);
        }
    }
    
    // 2. Process remaining segments in batches to control memory usage
    for (size_t i = 0; i < regularSegments.size(); i += MAX_BATCH_SIZE) {
        size_t batchEnd = std::min(i + MAX_BATCH_SIZE, regularSegments.size());
        std::vector<std::future<std::pair<CompressedSegmentHeader, std::vector<std::byte>>>> batchFutures;
        
        // Launch batch of compression tasks
        for (size_t j = i; j < batchEnd; j++) {
            const ModelSegment& segment = regularSegments[j].get();
            batchFutures.push_back(std::async(std::launch::async, 
                [this, &segment]() {

            CompressedSegmentHeader header;
            header.name = segment.name;
            header.original_type = segment.type;
            header.original_size = static_cast<uint64_t>(segment.original_size);
            header.tensor_metadata = segment.tensor_metadata;
            header.layer_name = segment.layer_name;
            header.layer_index = segment.layer_index;

            const auto* strategies = selectStrategies(segment.type);
            std::vector<std::byte> compressedData;
            bool compressionSuccessful = false;
            uint8_t winningStrategyId = 0;

            if (strategies && !strategies->empty()) {
                 for (const auto& stratInfo : *strategies) {
                     try {
                         compressedData = stratInfo.strategy->compress(segment);
                         winningStrategyId = stratInfo.id;
                         compressionSuccessful = true;
                         // Log success (optional, could be verbose in parallel)
                         // std::cerr << "Parallel Success: Segment '" << segment.name << "', Strategy ID " << static_cast<int>(stratInfo.id) << std::endl;
                         break;
                     } catch (const CompressionError& e) {
                         // Log failure (optional, could be verbose in parallel)
                         // std::cerr << "Parallel Warning: Segment '" << segment.name << "', Strategy ID " << static_cast<int>(stratInfo.id) << " failed: " << e.what() << std::endl;
                     }
                 }
            }

            if (!compressionSuccessful) {
                // std::cerr << "Parallel Warning: All strategies failed for segment '" << segment.name << "'. Storing uncompressed." << std::endl;
                winningStrategyId = 0; // Mark as uncompressed
                compressedData = segment.data; // Store original data
            }

            header.compression_strategy_id = winningStrategyId;
            header.compressed_size = static_cast<uint64_t>(compressedData.size());
            return std::make_pair(header, std::move(compressedData));
        }));
    }

            // Collect batch results
            for (auto& future : batchFutures) {
                results.push_back(future.get());
            }
            
            // Force cleanup of batch futures
            batchFutures.clear();
        }
    
    return results;
}

void AICompressor::compressModelStreaming(const std::string& modelPath, ICompressionHandler& handler) {
    // Reset statistics
    stats_ = CompressionStats();
    auto startTime = std::chrono::high_resolution_clock::now();
    if (!modelParser_) {
        throw std::runtime_error("AICompressor: No model parser provided.");
    }

    // Use model-aware chunking
    std::vector<ModelSegment> segments = modelParser_->parseWithChunking(modelPath);
    if (segments.empty()) {
        std::cerr << "Warning: Model parsing resulted in zero segments for " << modelPath << std::endl;
        return;
    }

    // Compress segments in parallel if multiple threads are configured
    if (numThreads_ > 1) {
        auto compressedSegments = compressSegmentsParallel(segments);
        for (auto& [header, data] : compressedSegments) {
            stats_.originalSize += header.original_size;
            stats_.compressedSize += header.compressed_size;
            handler.handleCompressedSegment(header, data);
        }
    } else {
        // Single-threaded compression
        for (const auto& segment : segments) {
            CompressedSegmentHeader header;
            header.name = segment.name;
            header.original_type = segment.type;
            header.original_size = static_cast<uint64_t>(segment.original_size);
            header.tensor_metadata = segment.tensor_metadata;
            header.layer_name = segment.layer_name;
            header.layer_index = segment.layer_index;

            const auto* strategies = selectStrategies(segment.type);
            std::vector<std::byte> compressedData;
            bool compressionSuccessful = false;

            if (strategies && !strategies->empty()) {
                for (const auto& stratInfo : *strategies) {
                    try {
                        compressedData = stratInfo.strategy->compress(segment);
                        // If compress succeeds without throwing CompressionError, we're done
                        header.compression_strategy_id = stratInfo.id;
                        compressionSuccessful = true;
                        std::cerr << "Successfully compressed segment '" << segment.name
                                  << "' with strategy ID " << static_cast<int>(stratInfo.id)
                                  << " (Priority: " << stratInfo.priority << ")" << std::endl;
                        break; // Exit the strategy loop
                    } catch (const CompressionError& e) {
                        std::cerr << "Warning: Compression failed for segment '" << segment.name
                                  << "' with strategy ID " << static_cast<int>(stratInfo.id)
                                  << " (Priority: " << stratInfo.priority << "): " << e.what()
                                  << ". Trying next strategy..." << std::endl;
                        // Continue to the next strategy
                    }
                }
            }

            // If no strategy succeeded or none were registered
            if (!compressionSuccessful) {
                 std::cerr << "Warning: All strategies failed for segment '" << segment.name
                           << "'. Storing uncompressed." << std::endl;
                header.compression_strategy_id = 0; // Mark as uncompressed
                compressedData = segment.data; // Store original data
            }

            header.compressed_size = static_cast<uint64_t>(compressedData.size());
            // Update statistics
            stats_.originalSize += header.original_size;
            stats_.compressedSize += header.compressed_size;
            
            handler.handleCompressedSegment(header, compressedData);
        }
    }
    
    // Update final compression statistics
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    stats_.compressionTimeMs = duration.count();
    stats_.compressionRatio = stats_.originalSize > 0 ? static_cast<double>(stats_.originalSize) / stats_.compressedSize : 0.0;
}

// Implementations for private helpers (if they were complex enough to not be inline)
// void AICompressor::writeArchiveHeader(...) { ... }
// void AICompressor::writeSegment(...) { ... }

} // namespace CortexAICompression
