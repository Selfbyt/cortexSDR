#ifndef AI_COMPRESSOR_HPP
#define AI_COMPRESSOR_HPP

#include "AIModelParser.hpp"
#include "../strategies/CompressionStrategy.hpp"
#include "ModelSegment.hpp"
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

// Structure to hold compression statistics
struct CompressionStats {
    size_t originalSize = 0;
    size_t compressedSize = 0;
    double compressionRatio = 0.0;
    double compressionTimeMs = 0.0;
    size_t numSegments = 0;
    size_t numCompressedSegments = 0;
};

// Represents the compressed archive format structure
struct CompressedSegmentHeader {
    SegmentType original_type;
    uint8_t compression_strategy_id;
    uint64_t original_size;
    uint64_t compressed_size;
    std::string name;
    std::optional<TensorMetadata> tensor_metadata;
    std::string layer_name;
    size_t layer_index;
    std::string layer_type; // ONNX op type (e.g., "Conv", "MatMul")
    std::vector<size_t> input_shape;  // True input tensor shape for the layer (from ONNX graph)
    std::vector<size_t> output_shape; // True output tensor shape for the layer (from ONNX graph)
};

// Interface for handling compressed segments during streaming
class ICompressionHandler {
public:
    virtual ~ICompressionHandler() = default;
    virtual void handleCompressedSegment(const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData) = 0;
};

// Thread pool for parallel processing
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

// Buffer pool for efficient memory management
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
    std::unique_ptr<ThreadPool> threadPool_;
    std::unique_ptr<BufferPool> bufferPool_;

    // Helper to select the list of appropriate strategies for a segment, ordered by priority
    const std::list<StrategyInfo>* selectStrategies(SegmentType type) const;

    // Helper to write the archive header/index (implementation needed)
    void writeArchiveHeader(std::ostream& stream, const std::vector<CompressedSegmentHeader>& headers);

    // Helper to write a single compressed segment (header + data)
    void writeSegment(std::ostream& stream, const CompressedSegmentHeader& header, const std::vector<std::byte>& compressedData);

    // Helper for sequential compression of a single segment
    std::pair<CompressedSegmentHeader, std::vector<std::byte>>
    compressSegment(const ModelSegment& segment) const;
    
    // Helper for memory-efficient parallel compression of segments
    std::vector<std::pair<CompressedSegmentHeader, std::vector<std::byte>>> 
    compressSegmentsParallel(const std::vector<ModelSegment>& segments) const;

    // New optimized methods
    std::vector<std::byte> compressSegmentWithBuffer(const ModelSegment& segment, const std::list<StrategyInfo>& strategies);
    void writeCompressedDataOptimized(std::ostream& stream, const std::vector<std::byte>& data);
    uint8_t selectBestStrategy(const ModelSegment& segment, const std::list<StrategyInfo>& strategies);
    
    // ONNX parsing helpers
    void parseONNXModelStreaming(const std::string& modelPath, ICompressionHandler& handler);
    std::vector<ModelSegment> parseONNXModelParallel(const std::string& modelPath, size_t numThreads);
};

} // namespace CortexAICompression

#endif // AI_COMPRESSOR_HPP
