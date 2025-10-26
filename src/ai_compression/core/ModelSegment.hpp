/**
 * @file ModelSegment.hpp
 * @brief Core data structures for neural network model segmentation and compression
 * 
 * This header defines the fundamental data structures used to represent and
 * manage segments of neural network models during compression and decompression.
 * Provides type-safe handling of different data types and metadata preservation.
 * 
 * Key Features:
 * - Type-safe segment identification and handling
 * - Support for various precision levels (FP32, FP16, INT8, INT4)
 * - Metadata preservation and tensor shape information
 * - Sparse representation support for efficiency
 * - Comprehensive model component coverage
 */

#ifndef AI_MODEL_SEGMENT_HPP
#define AI_MODEL_SEGMENT_HPP

#include <vector>
#include <string>
#include <variant>
#include <cstdint>
#include <optional>
#include <array>

namespace CortexAICompression {

/**
 * @brief Enumeration of supported neural network segment data types
 * 
 * Identifies the type of data contained within a model segment for
 * appropriate processing, compression, and reconstruction handling.
 */
enum class SegmentType {
    UNKNOWN,                    ///< Unknown or unidentified segment type
    WEIGHTS_FP32,              ///< 32-bit floating point weights
    WEIGHTS_FP16,              ///< 16-bit floating point weights  
    WEIGHTS_INT8,              ///< 8-bit integer weights (quantized)
    WEIGHTS_INT4,              ///< 4-bit integer weights (heavily quantized)
    SPARSE_INDICES,            ///< Sparse representation indices (SDR format)
    METADATA_JSON,             ///< JSON-formatted metadata
    METADATA_TOML,             ///< TOML-formatted metadata
    TOKENIZER_VOCAB,           ///< Tokenizer vocabulary data
    TOKENIZER_MODEL,           ///< Tokenizer model (e.g., SentencePiece)
    CONFIG,                    ///< Model configuration parameters
    ATTENTION_WEIGHTS,         ///< Transformer attention mechanism weights
    FEED_FORWARD_WEIGHTS,      ///< Feed-forward network weights
    EMBEDDING_WEIGHTS,         ///< Word/token embedding weights
    LAYER_NORM_WEIGHTS,        ///< Layer normalization parameters
    MODEL_INPUT,               ///< Input tensor shape and type definitions
    MODEL_OUTPUT,   // Output tensor definition
    GRAPH_STRUCTURE_PROTO, // Serialized ONNX GraphProto (nodes, connections, etc.)
    // Add more types as needed
};

/**
 * @brief Tensor shape and sparsity/quantization metadata.
 */
struct TensorMetadata {
    std::vector<size_t> dimensions;  // Shape of the tensor (e.g., [batch, height, width, channels])
    float sparsity_ratio;           // Ratio of zero elements (0.0 to 1.0)
    bool is_sorted;                 // Whether indices/values are sorted
    std::optional<float> scale;     // Optional quantization scale
    std::optional<float> zero_point; // Optional quantization zero point
};

/**
 * @brief Represents a distinct part of an AI model file.
 */
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
    std::vector<size_t> input_shape;  // True input tensor shape for the layer (from ONNX graph)
    std::vector<size_t> output_shape; // True output tensor shape for the layer (from ONNX graph)

    ModelSegment(SegmentType t, std::string n, std::vector<std::byte> d)
        : type(t), name(std::move(n)), data(std::move(d)), original_size(data.size()),
          layer_index(0) {}

    ModelSegment() : original_size(0), layer_index(0) {}

    /** Check if this segment represents a weight tensor. */
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
