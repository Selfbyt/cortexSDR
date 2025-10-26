/**
 * @file SparseInferenceEngine.hpp
 * @brief Sparse neural network inference engine with on-demand layer loading.
 *
 * @details This header declares the main types involved in running inference over
 * compressed model artifacts produced by the Cortex AI Compression pipeline.
 * The design emphasizes:
 * - On-demand layer loading to minimize peak memory usage
 * - A strategy-driven decompression pipeline
 * - Dynamic operation dispatch for diverse model architectures
 *
 * The public API is documented with Doxygen. Each function describes inputs,
 * outputs, side effects, and error conditions where relevant.
 */

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <future>
#include "core/ModelSegment.hpp"
#include <optional>
#include "onnx_proto/onnx.pb.h"
#include "core/AIDecompressor.hpp"
#include <functional>
#include <unordered_set>
#include <regex>
#include <mutex>

namespace CortexAICompression {

// Forward declarations
class ICompressionStrategy;

/**
 * @brief Neural network layer types for dynamic detection
 */
enum class LayerType {
    UNKNOWN,
    CONV2D,
    LINEAR,
    BATCH_NORM,
    POOLING,
    ACTIVATION
};

/**
 * @brief Container for neural network layer information and parameters.
 *
 * @details Holds all necessary data for executing a single layer including
 * weights, biases, shapes, and layer-specific properties. This is the unit of
 * computation consumed by the inference engine. Instances may be constructed
 * on-demand by the model loader and cached between calls.
 */
struct LayerInfo {
    std::string name;                       ///< Layer identifier/name
    std::string layer_type;                 ///< Layer operation type
    std::vector<size_t> active_indices;     ///< Sparse representation indices
    std::vector<std::byte> raw_data;        ///< Raw compressed layer data
    std::vector<size_t> input_shape;        ///< Expected input tensor shape
    std::vector<size_t> output_shape;       ///< Expected output tensor shape
    std::vector<float> weights;             ///< Layer weight parameters
    std::vector<float> biases;              ///< Layer bias parameters
    
    /**
     * @brief Layer-specific properties for different operation types
     */
    struct {
        std::vector<size_t> kernel_shape;    ///< Convolution/pooling kernel dimensions
        std::vector<size_t> strides;         ///< Convolution/pooling stride values
        std::vector<size_t> padding;         ///< Convolution/pooling padding values
        std::string activation_type;         ///< Activation function name
        float dropout_rate;                  ///< Dropout probability (0.0-1.0)
        bool use_batch_norm;                 ///< Enable batch normalization
        std::vector<float> bn_weights;       ///< Batch norm scale parameters
        std::vector<float> bn_biases;        ///< Batch norm shift parameters
        mutable std::vector<float> bn_running_mean;  ///< Batch norm running mean
        mutable std::vector<float> bn_running_var;   ///< Batch norm running variance
    } properties;

    // Layer type detection utilities
    bool isConvolutional() const { return layer_type == "CONV2D"; }
    bool isLinear() const { return layer_type == "LINEAR"; }
    bool isBatchNorm() const { return layer_type == "BATCH_NORM"; }
    bool isPooling() const { return layer_type == "POOLING"; }
    bool isActivation() const { return layer_type == "ACTIVATION"; }
};

/**
 * @brief Model loader with on-demand layer loading for memory efficiency
 * 
 * Provides access to compressed neural network models (.sdr files) with
 * Ollama-style on-demand loading to minimize memory usage. Only loads
 * layer data when specifically requested during inference.
 */
/**
 * @class SDRModelLoader
 * @brief Incremental loader for compressed model archives (.sdr).
 *
 * @details Provides Ollama-style on-demand access to model segments to avoid
 * loading the entire model into memory. Thread-safe read access is supported
 * through the use of futures for asynchronous layer materialization.
 */
class SDRModelLoader {
public:
    /**
     * @brief Initialize the loader for a given compressed model archive.
     * @param archive_path Absolute or relative path to a .sdr file.
     *
     * @throws std::runtime_error If the archive cannot be opened or parsed.
     * @complexity O(H) to read headers, where H is number of segment headers.
     */
    explicit SDRModelLoader(const std::string& archive_path);
    
    /**
     * @brief Get all pre-loaded layers (legacy method)
     * @return Vector of LayerInfo structures
     * @deprecated Use on-demand loading methods for better memory efficiency
     */
    const std::vector<LayerInfo>& getLayers() const;
    
    /**
     * @brief Obtain the index of compressed segments for on-demand access.
     * @return Vector of segment headers. Payloads are not loaded.
     */
    const std::vector<CompressedSegmentHeader>& getSegmentIndex() const;
    
    /**
     * @brief Materialize a single layer synchronously.
     * @param name Logical layer name to load.
     * @return Fully populated LayerInfo for computation.
     *
     * @throws std::out_of_range If the layer name does not exist.
     * @throws std::runtime_error On decompression or I/O errors.
     */
    LayerInfo loadLayerByName(const std::string& name) const;
    
    /**
     * @brief Start asynchronous materialization of a single layer.
     * @param name Logical layer name to load.
     * @return Shared future resolving to LayerInfo.
     *
     * @note Futures are cached per layer name to deduplicate concurrent requests.
     */
    std::shared_future<LayerInfo> loadLayerByNameAsync(const std::string& name) const;
    
    /**
     * @brief Evict a layer from the async cache.
     * @param name Layer name to clear from cache.
     *
     * @post Subsequent requests will trigger a fresh load.
     */
    void clearLayerFromCache(const std::string& name) const;
    
    /**
     * @brief Decode varint-encoded sparse indices from compressed data.
     * @param data Byte buffer containing varint-encoded indices.
     * @return Dense vector of decoded positions (0-based).
     *
     * @complexity O(N) where N is number of decoded indices.
     */
    static std::vector<size_t> decode_varint_indices(const std::vector<std::byte>& data);
    
#ifdef ENABLE_ONNX_PROTOBUF
    /**
     * @brief Get loaded ONNX model proto if available
     * @return Optional ONNX ModelProto structure
     */
    const std::optional<onnx::ModelProto>& getLoadedModelProto() const;
#endif
    
    /**
     * @brief Get layer lookup map for efficient access
     * @return Map of layer names to LayerInfo structures
     */
    const std::unordered_map<std::string, LayerInfo>& getLayerMap() const { return layer_map_; }
private:
    std::string archive_path_;
    std::vector<CompressedSegmentHeader> segments_;
    std::vector<LayerInfo> layers; // legacy: all layers loaded
    std::unordered_map<std::string, LayerInfo> layer_map_; // For fast lookup by name
    mutable std::unordered_map<std::string, std::shared_future<LayerInfo>> layer_cache_;
    std::unique_ptr<AIDecompressor> decompressor_;
    void loadFromArchive(const std::string& archive_path);
    void parseLayerMetadata(const std::string& metadata, LayerInfo& layer);
    std::vector<std::byte> decompressSDR(const std::vector<std::byte>& compressed_data, size_t original_size) const;
#ifdef ENABLE_ONNX_PROTOBUF
    std::optional<onnx::ModelProto> loaded_model_proto_;
#endif
};

/**
 * @brief Neural network inference engine with on-demand layer loading
 * 
 * Provides efficient neural network inference with compressed models using
 * on-demand layer loading for minimal memory footprint. Supports diverse
 * model architectures with dynamic layer type detection and fallback mechanisms.
 */
/**
 * @class SDRInferenceEngine
 * @brief Executes neural network layers provided by an SDRModelLoader.
 *
 * @details Offers whole-network and per-layer execution. The engine uses a
 * dispatch table to route layer types to specialized implementations and falls
 * back to adaptive handlers for unknown operations. Execution can be configured
 * via batch size, dropout, and training/inference mode flags.
 */
class SDRInferenceEngine {
public:
    struct LayerExecStat {
        std::string name;
        long load_ms;
        long exec_ms;
        size_t output_size;
        bool used_compressed;
        std::string op_type;
        double retained_index_ratio = 0.0; // nonzero_pairs / total_weights when streaming
    };
    struct RunStats {
        long total_ms;
        std::vector<LayerExecStat> layers;
    };
    /**
     * @brief Constructor with model loader for on-demand access
     * @param model_loader Reference to SDRModelLoader for layer access
     */
    explicit SDRInferenceEngine(SDRModelLoader& model_loader);
    
    /**
     * @brief Run end-to-end inference over the model's execution order.
     * @param input_tensor Input tensor (flattened, row-major).
     * @return Output tensor after the final layer.
     *
     * @throws std::runtime_error On layer load or execution errors.
     */
    std::vector<float> run(const std::vector<float>& input_tensor);
    const RunStats& getLastRunStats() const { return last_run_stats_; }
    
    /**
     * @brief Set batch size for inference operations
     * @param size Number of samples to process simultaneously
     */
    void setBatchSize(size_t size);
    
    /**
     * @brief Enable/disable dropout during inference
     * @param enable True to enable dropout, false to disable
     */
    void enableDropout(bool enable);
    
    /**
     * @brief Set training vs inference mode
     * @param training True for training mode, false for inference
     */
    void setInferenceMode(bool training);
    
    /**
     * @brief Prefer streaming from compressed weights when available
     * @param enable True to force compressed compute path when possible
     */
    void setForceCompressedCompute(bool enable);
    
    /**
     * @brief Execute a single layer.
     * @param layer LayerInfo including weights, biases, and attributes.
     * @param input Input tensor for the layer.
     * @return Output tensor after processing.
     */
    std::vector<float> runLayer(const LayerInfo& layer, const std::vector<float>& input);
    
    /**
     * @brief Execute an ordered sequence of layer names.
     * @param layer_names Execution order by logical names.
     * @param input Initial input tensor.
     * @return Final output tensor.
     */
    std::vector<float> runLayers(const std::vector<std::string>& layer_names, const std::vector<float>& input);
    
    /**
     * @brief Infer an execution order from segment metadata.
     * @param segments Model segment headers.
     * @return Ordered list of layer names.
     */
    std::vector<std::string> getExecutionOrder(const std::vector<CompressedSegmentHeader>& segments);
    
    /**
     * @brief Execute layers, loading each layer right before use.
     * @param layer_names Ordered layer names to execute.
     * @param input Initial input tensor.
     * @return Final output tensor.
     */
    std::vector<float> runLayersOnDemand(const std::vector<std::string>& layer_names, const std::vector<float>& input);
    
    /**
     * @brief Get set of all encountered layer types during execution
     * @return Unordered set of layer type strings
     */
    const std::unordered_set<std::string>& getEncounteredLayerTypes() const { return encountered_layer_types_; }
    
    /**
     * @brief Get set of unhandled layer types that used fallback processing
     * @return Unordered set of unhandled layer type strings
     */
    const std::unordered_set<std::string>& getUnhandledLayerTypes() const { return unhandled_layer_types_; }
    
    // Memory management for large models
    void initializeMemoryPool(size_t max_memory_mb = 8192);  ///< Initialize memory pool (default 8GB)
    void cleanupMemoryPool();                                ///< Reset memory pool
    void enableAggressiveMemoryManagement(bool enable);      ///< Enable/disable aggressive cleanup
    size_t getCurrentMemoryUsage() const;                    ///< Get current memory usage
    
private:
    SDRModelLoader& loader_;                ///< Reference to model loader for on-demand access
    
    // Inference configuration
    size_t batch_size;                      ///< Number of samples processed simultaneously
    bool dropout_enabled;                   ///< Enable dropout during inference
    bool training_mode;                     ///< Training vs inference mode flag
    bool force_compressed_compute_ = false; ///< Prefer compressed compute when possible
    bool last_layer_used_compressed_ = false; ///< Set by layer impls for benchmarking
    RunStats last_run_stats_{};               ///< Last run benchmark stats
    double last_layer_retained_ratio_ = 0.0;  ///< Set by streaming kernels
    std::unordered_map<std::string, LayerInfo> layer_map_;  ///< Fast layer lookup cache
    
    // Layer type tracking for debugging and optimization
    std::unordered_set<std::string> encountered_layer_types_;  ///< All processed layer types
    std::unordered_set<std::string> unhandled_layer_types_;    ///< Types using fallback processing
    
    // Memory management for large models
    mutable std::vector<float> memory_pool_;  ///< Pre-allocated memory pool for tensors
    mutable size_t memory_pool_offset_;       ///< Current offset in memory pool
    mutable std::mutex memory_pool_mutex_;    ///< Thread safety for memory pool
    size_t max_memory_usage_;                 ///< Maximum memory usage in bytes
    bool aggressive_memory_management_;       ///< Enable aggressive memory cleanup

    // Core neural network operations
    std::vector<float> applyLinearLayer(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> applyConvolutionalLayer(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> applyBatchNorm(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> applyActivation(const std::string& type, const std::vector<float>& input);
    std::vector<float> applyDropout(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> applyMaxPool(const std::vector<float>& input, const std::vector<size_t>& input_shape);
    std::vector<float> applyAvgPool(const std::vector<float>& input, const std::vector<size_t>& input_shape);
    
    // Utility operations
    void updateBatchNormStats(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> reshapeTensor(const std::vector<float>& input, const std::vector<size_t>& shape);
    std::vector<float> flattenTensor(const std::vector<float>& input);

    // Operation dispatch system
    std::unordered_map<std::string, std::function<std::vector<float>(const LayerInfo&, const std::vector<float>&)>> op_dispatch_;
    std::function<std::vector<float>(const LayerInfo&, const std::vector<float>&)> default_handler_;
    
    // Dynamic layer execution for diverse model architectures
    std::vector<float> executeDynamicLayer(const LayerInfo& layer, const std::vector<float>& input);
    
    // Specialized operation handlers for different layer types
    std::vector<float> executeLinearOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeConvolutionalOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeAttentionOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeNormalizationOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeActivationOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executePoolingOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeElementwiseOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeAdaptiveFallback(const LayerInfo& layer, const std::vector<float>& input);
    
    // Memory management utilities for large models (kept public above)
    std::vector<float> allocateFromPool(size_t size);        ///< Allocate tensor from pool
    void deallocateFromPool(size_t size);                    ///< Return memory to pool
    
    // Element-wise operation implementations
    std::vector<float> executeConcatOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeSliceOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeAddOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeSubOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeMulOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeDivOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeReshapeOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeTransposeOperation(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> executeFlattenOperation(const LayerInfo& layer, const std::vector<float>& input);
};

/**
 * @brief Utility function to analyze and print possible layer execution chains
 * @param layers Vector of LayerInfo structures to analyze for connectivity
 * 
 * Debugging utility that examines layer shapes to identify compatible
 * layer connections based on output/input shape matching.
 */
void print_possible_layer_chains(const std::vector<LayerInfo>& layers);

} // namespace CortexAICompression 