#ifndef AI_MODEL_SEGMENT_HPP
#define AI_MODEL_SEGMENT_HPP

#include <vector>
#include <string>
#include <variant>
#include <cstdint>
#include <optional>
#include <array>

namespace CortexAICompression {

// Enum to identify the type of data within a segment
enum class SegmentType {
    UNKNOWN,
    WEIGHTS_FP32,
    WEIGHTS_FP16,
    WEIGHTS_INT8,
    WEIGHTS_INT4,
    SPARSE_INDICES, // For SDR-like data
    METADATA_JSON,
    METADATA_TOML,
    TOKENIZER_VOCAB,
    TOKENIZER_MODEL, // e.g., SentencePiece model data
    CONFIG,
    ATTENTION_WEIGHTS,
    FEED_FORWARD_WEIGHTS,
    EMBEDDING_WEIGHTS,
    LAYER_NORM_WEIGHTS,
    MODEL_INPUT,    // Input tensor definition
    MODEL_OUTPUT,   // Output tensor definition
    GRAPH_STRUCTURE_PROTO, // Serialized ONNX GraphProto (nodes, connections, etc.)
    // Add more types as needed
};

// Structure to hold tensor dimensions and metadata
struct TensorMetadata {
    std::vector<size_t> dimensions;  // Shape of the tensor (e.g., [batch, height, width, channels])
    float sparsity_ratio;           // Ratio of zero elements (0.0 to 1.0)
    bool is_sorted;                 // Whether indices/values are sorted
    std::optional<float> scale;     // Optional quantization scale
    std::optional<float> zero_point; // Optional quantization zero point
};

// Represents a distinct part of an AI model file
struct ModelSegment {
    SegmentType type = SegmentType::UNKNOWN;
    std::string name;               // e.g., "layer1.weights", "config.json"
    std::vector<std::byte> data;    // Raw byte data of the segment
    size_t original_size;           // Size before any processing/compression
    std::optional<TensorMetadata> tensor_metadata; // Optional tensor-specific metadata
    std::string layer_name;         // Name of the layer this segment belongs to
    size_t layer_index;            // Index of the layer in the model
    std::string data_format;       // Format of the data (e.g., "NCHW", "NHWC")
    std::string layer_type;         // layer_type is now a free-form string for extensibility

    ModelSegment(SegmentType t, std::string n, std::vector<std::byte> d)
        : type(t), name(std::move(n)), data(std::move(d)), original_size(data.size()),
          layer_index(0) {}

    ModelSegment() : original_size(0), layer_index(0) {}

    // Helper to check if this segment represents a weight tensor
    bool isWeightTensor() const {
        return type == SegmentType::WEIGHTS_FP32 ||
               type == SegmentType::WEIGHTS_FP16 ||
               type == SegmentType::WEIGHTS_INT8 ||
               type == SegmentType::WEIGHTS_INT4 ||
               type == SegmentType::ATTENTION_WEIGHTS ||
               type == SegmentType::FEED_FORWARD_WEIGHTS ||
               type == SegmentType::EMBEDDING_WEIGHTS ||
               type == SegmentType::LAYER_NORM_WEIGHTS;
    }
};

} // namespace CortexAICompression

#endif // AI_MODEL_SEGMENT_HPP
