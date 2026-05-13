#ifndef AI_COMPRESSOR_HPP
#define AI_COMPRESSOR_HPP

/**
 * @file AICompressor.hpp
 * @brief Public API for the AI model compressor and streaming interfaces.
 *
 * @details Declares the compressor that converts model artifacts (e.g., ONNX)
 * into a compressed archive made of typed segments. Exposes streaming hooks,
 * statistics, and a strategy registry to customize compression behavior.
 */
#include "AIModelParser.hpp"
#include "../strategies/CompressionStrategy.hpp"
#include "ModelSegment.hpp"
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <map>
#include <list>
#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <fstream>

namespace CortexAICompression {

/**
 * @brief Statistics about a compression run.
 */
struct CompressionStats {
    size_t originalSize = 0;
    size_t compressedSize = 0;
    double compressionRatio = 0.0;
    double compressionTimeMs = 0.0;
    size_t numSegments = 0;
    size_t numCompressedSegments = 0;
};

/**
 * @brief Header describing a single compressed segment in the archive.
 *
 * @details Includes original type and size information, chosen strategy ID,
 * human-readable name, layer linkage, ONNX op type, and true tensor shapes.
 */
struct CompressedSegmentHeader {
    SegmentType original_type;
    uint8_t compression_strategy_id;
    uint64_t original_size;
    uint64_t compressed_size;
    uint64_t data_offset = 0;
    std::string name;
    std::string data_format; // Tensor/storage format (e.g., "f16", "q4_0", "ONNX")
    std::optional<TensorMetadata> tensor_metadata;
    std::string layer_name;
    size_t layer_index;
    std::string layer_type; // ONNX op type (e.g., "Conv", "MatMul")
    std::vector<size_t> input_shape;  // True input tensor shape for the layer (from ONNX graph)
    std::vector<size_t> output_shape; // True output tensor shape for the layer (from ONNX graph)
};

/**
 * @brief Interface to receive compressed segments during streaming compression.
 */
class ICompressionHandler {
public:
    virtual ~ICompressionHandler() = default;
    /**
     * @brief Receive a compressed segment chunk-by-chunk or as a whole.
     * @param header Segment metadata.
     * @param compressedData Owned vector containing the compressed bytes.
     */
    virtual void handleCompressedSegment(const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData) = 0;
};

/**
 * @brief Minimal thread pool for parallel compression tasks.
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads) : stop(false) {
        for(size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        if(this->stop && this->tasks.empty()) {
                            return;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

/**
 * @brief Buffer pool to reduce heap churn during compression.
 */
class BufferPool {
public:
    explicit BufferPool(size_t maxBuffers = 100) : maxBuffers(maxBuffers) {}

    std::vector<std::byte> acquire(size_t size) {
        std::unique_lock<std::mutex> lock(mutex);
        auto it = std::find_if(buffers.begin(), buffers.end(),
            [size](const auto& buf) { return buf.size() >= size; });
        
        if (it != buffers.end()) {
            auto buffer = std::move(*it);
            buffers.erase(it);
            return buffer;
        }
        return std::vector<std::byte>(size);
    }

    void release(std::vector<std::byte>&& buffer) {
        std::unique_lock<std::mutex> lock(mutex);
        if (buffers.size() < maxBuffers) {
            buffers.push_back(std::move(buffer));
        }
    }

private:
    std::vector<std::vector<std::byte>> buffers;
    std::mutex mutex;
    size_t maxBuffers;
};

class AICompressor {
public:
    /**
     * @brief Construct a compressor with a model parser implementation.
     */
    explicit AICompressor(std::unique_ptr<IAIModelParser> parser);

    /**
     * @brief Register a compression strategy for a segment type.
     * @param type Segment type the strategy can handle.
     * @param priority Lower is tried first.
     * @param strategy_id Byte identifier persisted in the archive.
     * @param strategy Shared strategy instance.
     */
    void registerStrategy(SegmentType type, int priority, uint8_t strategy_id, std::shared_ptr<ICompressionStrategy> strategy);

    /**
     * @brief Compress a model and write the full archive to a stream.
     * @throws ParsingError On model parse failure.
     * @throws CompressionError On strategy failures or I/O errors.
     */
    void compressModel(const std::string& modelPath, std::ostream& outputArchiveStream);

    /**
     * @brief Compress model with streaming callbacks per segment.
     */
    void compressModelStreaming(const std::string& modelPath, ICompressionHandler& handler);

    /**
     * @brief Configure the number of threads for parallel compression.
     */
    void setCompressionThreads(size_t numThreads) { numThreads_ = numThreads; }

    /**
     * @brief Set a predicate to drop segments before compression.
     *
     * The predicate is called for each parsed segment; if it returns true,
     * the segment is omitted from the archive entirely. Used by CLI flags
     * like `--skip-embedding` to keep the archive small at the cost of
     * needing the source model alongside for inference-time lookups.
     */
    void setSkipPredicate(std::function<bool(const ModelSegment&)> pred) {
        skipPredicate_ = std::move(pred);
    }

    /**
     * @brief Access statistics accumulated during compression.
     */
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
    std::function<bool(const ModelSegment&)> skipPredicate_;
    std::unique_ptr<ThreadPool> threadPool_;
    std::unique_ptr<BufferPool> bufferPool_;

    // Helper to select the list of appropriate strategies for a segment, ordered by priority
    const std::list<StrategyInfo>* selectStrategies(SegmentType type) const;

    /** Write archive header/index to the stream. */
    void writeArchiveHeader(std::ostream& stream, const std::vector<CompressedSegmentHeader>& headers);

    /** Write one compressed segment (header + data) to the stream. */
    void writeSegment(std::ostream& stream, const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData);

    /** Compress one segment sequentially. */
    std::pair<CompressedSegmentHeader, std::vector<std::byte>>
    compressSegment(const ModelSegment& segment) const;
    
    /** Compress segments in parallel using the thread and buffer pools. */
    std::vector<std::pair<CompressedSegmentHeader, std::vector<std::byte>>> 
    compressSegmentsParallel(const std::vector<ModelSegment>& segments) const;

    // Optimized helpers
    std::vector<std::byte> compressSegmentWithBuffer(const ModelSegment& segment, const std::list<StrategyInfo>& strategies);
    void writeCompressedDataOptimized(std::ostream& stream, const std::vector<std::byte>& data);
    uint8_t selectBestStrategy(const ModelSegment& segment, const std::list<StrategyInfo>& strategies);
    
    // ONNX parsing helpers
    void parseONNXModelStreaming(const std::string& modelPath, ICompressionHandler& handler);
    std::vector<ModelSegment> parseONNXModelParallel(const std::string& modelPath, size_t numThreads);
};

} // namespace CortexAICompression

#endif // AI_COMPRESSOR_HPP
