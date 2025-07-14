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
#include "core/ModelSegment.hpp"  // Include for SegmentType
#include <optional>
#include "onnx_proto/onnx.pb.h"
#include "core/AIDecompressor.hpp" // For AIDecompressor
#include <functional>
#include <unordered_set>

namespace CortexAICompression {

// Forward declarations
class ICompressionStrategy;

// Layer types for dynamic detection
enum class LayerType {
    UNKNOWN,
    CONV2D,
    LINEAR,
    BATCH_NORM,
    POOLING,
    ACTIVATION
};

// LayerInfo holds SDR indices and metadata for a layer
struct LayerInfo {
    std::string name;
    std::string layer_type;      // Free-form layer type string
    std::vector<size_t> active_indices;  // Decoded indices of active bits
    std::vector<std::byte> raw_data;     // Raw SDR data
    std::vector<size_t> input_shape;     // Input shape
    std::vector<size_t> output_shape;    // Output shape
    std::vector<float> weights;          // Actual weight values
    std::vector<float> biases;           // Bias values if present
    
    // Dynamic properties based on layer type
    struct {
        std::vector<size_t> kernel_shape;    // For conv/pool layers
        std::vector<size_t> strides;         // For conv/pool layers
        std::vector<size_t> padding;         // For conv/pool layers
        std::string activation_type;         // For activation layers
        float dropout_rate;                  // For dropout layers
        bool use_batch_norm;                 // For batch norm layers
        std::vector<float> bn_weights;       // Batch norm weights
        std::vector<float> bn_biases;        // Batch norm biases
        mutable std::vector<float> bn_running_mean;  // Batch norm running mean
        mutable std::vector<float> bn_running_var;   // Batch norm running variance
    } properties;

    // Helper methods for layer type detection
    bool isConvolutional() const { return layer_type == "CONV2D"; }
    bool isLinear() const { return layer_type == "LINEAR"; }
    bool isBatchNorm() const { return layer_type == "BATCH_NORM"; }
    bool isPooling() const { return layer_type == "POOLING"; }
    bool isActivation() const { return layer_type == "ACTIVATION"; }
};

class SDRModelLoader {
public:
    explicit SDRModelLoader(const std::string& archive_path);
    // Legacy: loads all layers into memory (not memory efficient for large models)
    const std::vector<LayerInfo>& getLayers() const;
    // On-demand: get segment index (headers only, no data loaded)
    const std::vector<CompressedSegmentHeader>& getSegmentIndex() const;
    // On-demand: load a single layer by name
    LayerInfo loadLayerByName(const std::string& name) const;
    // On-demand: load a single layer by name asynchronously
    std::shared_future<LayerInfo> loadLayerByNameAsync(const std::string& name) const;
    // Helper to decode varint-encoded indices from SDR data
    static std::vector<size_t> decode_varint_indices(const std::vector<std::byte>& data);
#ifdef ENABLE_ONNX_PROTOBUF
    // Getter for loaded ONNX ModelProto (if present)
    const std::optional<onnx::ModelProto>& getLoadedModelProto() const;
#endif
    // New: Get a map of layer names to LayerInfo for selective inference
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

class SDRInferenceEngine {
public:
    // Constructor now takes the loader directly for on-demand loading
    explicit SDRInferenceEngine(SDRModelLoader& model_loader);
    // Given input tensor, run inference and return output tensor
    std::vector<float> run(const std::vector<float>& input_tensor);
    
    // Additional inference options
    void setBatchSize(size_t size);
    void enableDropout(bool enable);
    void setInferenceMode(bool training);
    
    // New: Run a single layer by name
    std::vector<float> runLayer(const LayerInfo& layer, const std::vector<float>& input);
    // Run a sequence of layers by name, loading them on-demand
    std::vector<float> runLayers(const std::vector<std::string>& layer_names, const std::vector<float>& input);
    
    // New: Get all encountered layer types
    const std::unordered_set<std::string>& getEncounteredLayerTypes() const { return encountered_layer_types_; }
    // New: Get all unhandled layer types
    const std::unordered_set<std::string>& getUnhandledLayerTypes() const { return unhandled_layer_types_; }
    
private:
    // Engine no longer stores all layers. It holds a reference to the loader.
    SDRModelLoader& loader_; 

    size_t batch_size;
    bool dropout_enabled;
    bool training_mode;
    std::unordered_map<std::string, LayerInfo> layer_map_; // For fast lookup by name
    
    // New: Track all encountered and unhandled layer types
    std::unordered_set<std::string> encountered_layer_types_;
    std::unordered_set<std::string> unhandled_layer_types_;

    // Helper functions for neural network operations
    std::vector<float> applyLinearLayer(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> applyConvolutionalLayer(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> applyBatchNorm(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> applyActivation(const std::string& type, const std::vector<float>& input);
    std::vector<float> applyDropout(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> applyMaxPool(const std::vector<float>& input, const std::vector<size_t>& input_shape);
    std::vector<float> applyAvgPool(const std::vector<float>& input, const std::vector<size_t>& input_shape);
    
    // Utility functions
    void updateBatchNormStats(const LayerInfo& layer, const std::vector<float>& input);
    std::vector<float> reshapeTensor(const std::vector<float>& input, const std::vector<size_t>& shape);
    std::vector<float> flattenTensor(const std::vector<float>& input);

    // Dynamic dispatch table for op type -> handler
    std::unordered_map<std::string, std::function<std::vector<float>(const LayerInfo&, const std::vector<float>&)>> op_dispatch_;
    // Default handler for unknown ops
    std::function<std::vector<float>(const LayerInfo&, const std::vector<float>&)> default_handler_;
};

// Heuristic to get a plausible execution order by sorting layer names
void print_possible_layer_chains(const std::vector<LayerInfo>& layers);

} // namespace CortexAICompression 