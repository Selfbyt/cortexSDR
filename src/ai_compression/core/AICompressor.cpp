/**
 * @file AICompressor.cpp
 * @brief Implementation of AI model compression with multiple strategies
 * 
 * This file implements the AICompressor class which provides comprehensive
 * neural network model compression using various strategies including:
 * - Sparse Distributed Representation (SDR) compression
 * - Run-Length Encoding (RLE) for sequential data
 * - Gzip compression for general purpose reduction
 * - Quantization strategies for precision reduction
 * 
 * Key Features:
 * - Multi-strategy compression pipeline
 * - Streaming compression for large models
 * - Parallel processing for performance optimization
 * - Comprehensive error handling and validation
 * - Streaming-friendly archive layout with indexed random access
 */

#include "AICompressor.hpp"
#include "../strategies/CompressionStrategy.hpp"
#include "ArchiveConstants.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstring>
#include <thread>
#include <future>
#include <list>
#include <algorithm>
#include <sstream>
#include "../utils/sha256.h"

namespace CortexAICompression {
namespace {
std::string shapeToString(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return "[]";
    }
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

std::string segmentDiagnostics(const ModelSegment& segment) {
    std::ostringstream ss;
    ss << "name='" << segment.name << "'"
       << ", type=" << static_cast<int>(segment.type)
       << ", data_format='" << segment.data_format << "'"
       << ", layer_type='" << segment.layer_type << "'"
       << ", layer_name='" << segment.layer_name << "'"
       << ", layer_index=" << segment.layer_index
       << ", original_size=" << segment.original_size
       << ", payload_bytes=" << segment.data.size()
       << ", input_shape=" << shapeToString(segment.input_shape)
       << ", output_shape=" << shapeToString(segment.output_shape);
    return ss.str();
}
} // namespace

/**
 * @brief Write binary data of basic types to output stream
 * @tparam T Type of data to write (must be trivially copyable)
 * @param stream Output stream to write to
 * @param value Value to write in binary format
 */
template<typename T>
void writeBasicType(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

/**
 * @brief Write length-prefixed string to output stream
 * @param stream Output stream to write to
 * @param str String to write with 32-bit length prefix
 * @throws std::runtime_error if string length exceeds UINT32_MAX
 */
void writeString(std::ostream& stream, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.length());
    if (str.length() > UINT32_MAX) {
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

    // Apply skip predicate (used by CLI flags like --skip-embedding).
    if (skipPredicate_) {
        size_t kept = 0;
        size_t dropped_count = 0;
        size_t dropped_bytes = 0;
        for (size_t i = 0; i < segments.size(); ++i) {
            if (skipPredicate_(segments[i])) {
                dropped_bytes += segments[i].data.size();
                ++dropped_count;
                continue;
            }
            if (kept != i) segments[kept] = std::move(segments[i]);
            ++kept;
        }
        segments.resize(kept);
        if (dropped_count > 0) {
            std::cerr << "  Skip-predicate dropped " << dropped_count
                      << " segments (" << dropped_bytes << " source bytes)\n";
        }
    }

    // 2. Compress each segment and gather header info
    std::vector<CompressedSegmentHeader> headers;
    std::vector<std::vector<std::byte>> compressedDatas;
    headers.reserve(segments.size());
    compressedDatas.reserve(segments.size());

    for (const auto& segment : segments) {
        CompressedSegmentHeader header;
        header.name = segment.name;
        header.data_format = segment.data_format;
        header.original_type = segment.type;
        header.original_size = static_cast<uint64_t>(segment.original_size);
        header.layer_type = segment.layer_type;
        header.input_shape = segment.input_shape;
        header.output_shape = segment.output_shape;
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
                    // If compress succeeds without throwing CompressionError, we're done for this segment
                    header.compression_strategy_id = stratInfo.id;
                    compressionSuccessful = true;
                    break; // Exit the strategy loop for this segment
                } catch (const CompressionError& e) {
                    std::cerr << "Warning: Compression failed for segment '" << segment.name
                              << "' with strategy ID " << static_cast<int>(stratInfo.id)
                              << " (Priority: " << stratInfo.priority << "): " << e.what()
                              << ". Trying next strategy..." << std::endl;
                    std::cerr << "  Segment diagnostics: " << segmentDiagnostics(segment) << std::endl;
                    // Continue to the next strategy in the list
                } catch (const std::bad_alloc&) {
                    // Treat OOM like a CompressionError: the strategy can't
                    // handle this segment at its current size, so we fall
                    // through to the next priority (Gzip is usually safe
                    // because it doesn't dequantise). Without this, a
                    // single 2 GB embedding bad_alloc aborts the whole
                    // archive — happens routinely on 7B-scale models.
                    std::cerr << "Warning: Compression OOM for segment '" << segment.name
                              << "' with strategy ID " << static_cast<int>(stratInfo.id)
                              << " (Priority: " << stratInfo.priority
                              << "). Trying next strategy..." << std::endl;
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

    // 3. Write archive header placeholders matching the streaming archive format
    outputArchiveStream.write(ARCHIVE_MAGIC, sizeof(ARCHIVE_MAGIC));
    writeBasicType(outputArchiveStream, ARCHIVE_VERSION);
    uint64_t numSegments = static_cast<uint64_t>(headers.size());
    uint64_t indexOffsetPlaceholder = 0;
    writeBasicType(outputArchiveStream, numSegments);
    writeBasicType(outputArchiveStream, indexOffsetPlaceholder);

    // 4. Write compressed payloads first and record offsets
    for (size_t i = 0; i < headers.size(); ++i) {
        headers[i].data_offset = static_cast<uint64_t>(outputArchiveStream.tellp());
        outputArchiveStream.write(reinterpret_cast<const char*>(compressedDatas[i].data()), compressedDatas[i].size());
    }

    // 5. Write index table at the end, matching StreamingCompressor::finalizeArchive
    const uint64_t indexOffset = static_cast<uint64_t>(outputArchiveStream.tellp());
    for (size_t i = 0; i < headers.size(); ++i) {
        const auto& header = headers[i];

        writeString(outputArchiveStream, header.name);
        writeString(outputArchiveStream, header.data_format);
        writeString(outputArchiveStream, header.layer_type);
        writeBasicType(outputArchiveStream, static_cast<uint8_t>(header.original_type));
        writeBasicType(outputArchiveStream, header.compression_strategy_id);
        writeBasicType(outputArchiveStream, header.original_size);
        writeBasicType(outputArchiveStream, header.compressed_size);
        writeBasicType(outputArchiveStream, header.data_offset);
        writeString(outputArchiveStream, header.layer_name);
        writeBasicType(outputArchiveStream, static_cast<uint32_t>(header.layer_index));

        const bool hasMetadata = header.tensor_metadata.has_value();
        writeBasicType(outputArchiveStream, hasMetadata);
        if (hasMetadata) {
            const auto& meta = header.tensor_metadata.value();
            const uint8_t numDims = static_cast<uint8_t>(meta.dimensions.size());
            writeBasicType(outputArchiveStream, numDims);
            for (size_t dim : meta.dimensions) {
                writeBasicType(outputArchiveStream, static_cast<uint32_t>(dim));
            }
            writeBasicType(outputArchiveStream, meta.sparsity_ratio);
            writeBasicType(outputArchiveStream, meta.is_sorted);
            const bool hasScale = meta.scale.has_value();
            writeBasicType(outputArchiveStream, hasScale);
            if (hasScale) {
                writeBasicType(outputArchiveStream, meta.scale.value());
            }
            const bool hasZeroPoint = meta.zero_point.has_value();
            writeBasicType(outputArchiveStream, hasZeroPoint);
            if (hasZeroPoint) {
                writeBasicType(outputArchiveStream, meta.zero_point.value());
            }
        }

        const uint8_t hasInputShape = header.input_shape.empty() ? 0 : 1;
        writeBasicType(outputArchiveStream, hasInputShape);
        if (hasInputShape) {
            const uint8_t numIn = static_cast<uint8_t>(header.input_shape.size());
            writeBasicType(outputArchiveStream, numIn);
            for (size_t dim : header.input_shape) {
                writeBasicType(outputArchiveStream, static_cast<uint32_t>(dim));
            }
        }

        const uint8_t hasOutputShape = header.output_shape.empty() ? 0 : 1;
        writeBasicType(outputArchiveStream, hasOutputShape);
        if (hasOutputShape) {
            const uint8_t numOut = static_cast<uint8_t>(header.output_shape.size());
            writeBasicType(outputArchiveStream, numOut);
            for (size_t dim : header.output_shape) {
                writeBasicType(outputArchiveStream, static_cast<uint32_t>(dim));
            }
        }
    }

    // 6. Patch the index offset in the fixed header.
    outputArchiveStream.seekp(sizeof(ARCHIVE_MAGIC) + sizeof(ARCHIVE_VERSION) + sizeof(numSegments), std::ios::beg);
    writeBasicType(outputArchiveStream, indexOffset);

    outputArchiveStream.seekp(0, std::ios::end);
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
    header.data_format = segment.data_format;
    header.original_type = segment.type;
    header.original_size = static_cast<uint64_t>(segment.original_size);
    header.layer_type = segment.layer_type;
    header.input_shape = segment.input_shape;
    header.output_shape = segment.output_shape;
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
                break;
            } catch (const CompressionError& e) {
                // Log failure 
                std::cerr << "Warning: Segment '" << segment.name << "', Strategy ID " 
                          << static_cast<int>(stratInfo.id) << " (Priority: " << stratInfo.priority
                          << ") failed: " << e.what() << std::endl;
                std::cerr << "  Segment diagnostics: " << segmentDiagnostics(segment) << std::endl;
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
                    return this->compressSegment(segment);
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

    // Stream segments in bounded batches to avoid buffering the entire archive payload.
    if (numThreads_ > 1) {
        const size_t batch_size = std::max<size_t>(1, std::min<size_t>(numThreads_, 8));
        for (size_t i = 0; i < segments.size(); i += batch_size) {
            const size_t batch_end = std::min(i + batch_size, segments.size());
            std::vector<std::future<std::pair<CompressedSegmentHeader, std::vector<std::byte>>>> batch_futures;
            batch_futures.reserve(batch_end - i);

            for (size_t j = i; j < batch_end; ++j) {
                batch_futures.push_back(std::async(std::launch::async, [this, &segments, j]() {
                    return this->compressSegment(segments[j]);
                }));
            }

            for (size_t j = i; j < batch_end; ++j) {
                auto [header, data] = batch_futures[j - i].get();
                stats_.originalSize += header.original_size;
                stats_.compressedSize += header.compressed_size;
                handler.handleCompressedSegment(header, data);

                // Release original segment payload eagerly after it is handled.
                segments[j].data.clear();
                segments[j].data.shrink_to_fit();
            }
        }
    } else {
        // Single-threaded compression
        for (auto& segment : segments) {
            auto [header, compressedData] = compressSegment(segment);

            // Update statistics
            stats_.originalSize += header.original_size;
            stats_.compressedSize += header.compressed_size;
            
            handler.handleCompressedSegment(header, compressedData);

            // Release original segment payload eagerly after it is handled.
            segment.data.clear();
            segment.data.shrink_to_fit();
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
