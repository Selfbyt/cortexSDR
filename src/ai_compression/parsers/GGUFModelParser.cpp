/**
 * @file GGUFModelParser.cpp
 * @brief Implementation of GGUF model parsing into archive segments.
 */
#include "GGUFModelParser.hpp"
#include <stdexcept>
#include <array>
#include <regex>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <limits>
#include <filesystem>
#include <unordered_set>

namespace CortexAICompression {

namespace {

enum class GGUFMetadataType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35
};

struct GGMLTypeTraits {
    const char* name;
    uint32_t block_size;
    uint32_t bytes_per_block;
};

template <typename T>
T readBinary(std::ifstream& file, const char* field_name) {
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw ParsingError(std::string("Failed to read GGUF field: ") + field_name);
    }
    return value;
}

std::string readGGUFString(std::ifstream& file) {
    const uint64_t length = readBinary<uint64_t>(file, "string.length");
    if (length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw ParsingError("GGUF string length exceeds addressable memory");
    }

    std::string value(static_cast<size_t>(length), '\0');
    if (length > 0) {
        file.read(value.data(), static_cast<std::streamsize>(length));
        if (!file) {
            throw ParsingError("Failed to read GGUF string contents");
        }
    }
    return value;
}

uint64_t alignOffset(uint64_t offset, uint32_t alignment) {
    if (alignment == 0) {
        return offset;
    }
    const uint64_t remainder = offset % static_cast<uint64_t>(alignment);
    return remainder == 0 ? offset : (offset + static_cast<uint64_t>(alignment) - remainder);
}

std::string escapeJson(const std::string& input) {
    std::ostringstream escaped;
    for (unsigned char ch : input) {
        switch (ch) {
            case '\\': escaped << "\\\\"; break;
            case '"': escaped << "\\\""; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (ch < 0x20) {
                    escaped << "\\u"
                            << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(ch)
                            << std::dec << std::setfill(' ');
                } else {
                    escaped << static_cast<char>(ch);
                }
        }
    }
    return escaped.str();
}

GGMLTypeTraits getGGMLTypeTraits(uint32_t ggml_type_value) {
    switch (static_cast<GGMLType>(ggml_type_value)) {
        case GGMLType::F32: return {"F32", 1, 4};
        case GGMLType::F16: return {"F16", 1, 2};
        case GGMLType::Q4_0: return {"Q4_0", 32, 18};
        case GGMLType::Q4_1: return {"Q4_1", 32, 20};
        case GGMLType::Q5_0: return {"Q5_0", 32, 22};
        case GGMLType::Q5_1: return {"Q5_1", 32, 24};
        case GGMLType::Q8_0: return {"Q8_0", 32, 34};
        case GGMLType::Q8_1: return {"Q8_1", 32, 40};
        case GGMLType::Q2_K: return {"Q2_K", 256, 84};
        case GGMLType::Q3_K: return {"Q3_K", 256, 110};
        case GGMLType::Q4_K: return {"Q4_K", 256, 144};
        case GGMLType::Q5_K: return {"Q5_K", 256, 176};
        case GGMLType::Q6_K: return {"Q6_K", 256, 210};
        case GGMLType::Q8_K: return {"Q8_K", 256, 292};
        case GGMLType::IQ2_XXS: return {"IQ2_XXS", 256, 66};
        case GGMLType::IQ2_XS: return {"IQ2_XS", 256, 74};
        case GGMLType::IQ3_XXS: return {"IQ3_XXS", 256, 98};
        case GGMLType::IQ1_S: return {"IQ1_S", 256, 50};
        case GGMLType::IQ4_NL: return {"IQ4_NL", 32, 18};
        case GGMLType::IQ3_S: return {"IQ3_S", 256, 110};
        case GGMLType::IQ2_S: return {"IQ2_S", 256, 82};
        case GGMLType::IQ4_XS: return {"IQ4_XS", 256, 136};
        case GGMLType::I8: return {"I8", 1, 1};
        case GGMLType::I16: return {"I16", 1, 2};
        case GGMLType::I32: return {"I32", 1, 4};
        case GGMLType::I64: return {"I64", 1, 8};
        case GGMLType::F64: return {"F64", 1, 8};
        case GGMLType::IQ1_M: return {"IQ1_M", 256, 56};
        case GGMLType::BF16: return {"BF16", 1, 2};
        case GGMLType::Q4_0_4_4: return {"Q4_0_4_4", 128, 72};
        case GGMLType::Q4_0_4_8: return {"Q4_0_4_8", 256, 144};
        case GGMLType::Q4_0_8_8: return {"Q4_0_8_8", 256, 144};
        case GGMLType::TQ1_0: return {"TQ1_0", 256, 54};
        case GGMLType::TQ2_0: return {"TQ2_0", 256, 66};
        default: return {"UNKNOWN", 1, 0};
    }
}

uint64_t computeTensorByteSize(const std::vector<size_t>& dimensions, uint32_t ggml_type) {
    const GGMLTypeTraits traits = getGGMLTypeTraits(ggml_type);
    if (traits.bytes_per_block == 0) {
        return 0;
    }

    uint64_t element_count = 1;
    for (size_t dim : dimensions) {
        if (dim == 0) {
            return 0;
        }
        const uint64_t dim64 = static_cast<uint64_t>(dim);
        if (element_count > std::numeric_limits<uint64_t>::max() / dim64) {
            throw ParsingError("GGUF tensor element count overflow");
        }
        element_count *= dim64;
    }

    const uint64_t block_size = static_cast<uint64_t>(traits.block_size);
    const uint64_t bytes_per_block = static_cast<uint64_t>(traits.bytes_per_block);
    const uint64_t block_count = (element_count + block_size - 1) / block_size;
    return block_count * bytes_per_block;
}

bool isQuantizedGGUFType(const std::string& data_type) {
    if (data_type.empty()) {
        return false;
    }

    std::string lower_type = data_type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lower_type.rfind("q", 0) == 0 ||
           lower_type.rfind("iq", 0) == 0 ||
           lower_type.rfind("tq", 0) == 0;
}

std::string stripTensorSuffix(std::string tensor_name) {
    static const std::array<const char*, 7> suffixes = {
        ".weight", ".bias", ".scales", ".weight_scale", ".weight_bias", ".gamma", ".beta"
    };
    for (const char* suffix : suffixes) {
        const std::string suffix_str(suffix);
        if (tensor_name.size() > suffix_str.size() &&
            tensor_name.compare(tensor_name.size() - suffix_str.size(), suffix_str.size(), suffix_str) == 0) {
            tensor_name.erase(tensor_name.size() - suffix_str.size());
            break;
        }
    }
    return tensor_name;
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string classifyTensorRole(const std::string& tensor_name) {
    const std::string lower_name = toLower(tensor_name);
    if (lower_name.find("attn_q") != std::string::npos || lower_name.find("q_proj") != std::string::npos || lower_name.find(".wq") != std::string::npos) {
        return "LINEAR";
    }
    if (lower_name.find("attn_k") != std::string::npos || lower_name.find("k_proj") != std::string::npos || lower_name.find(".wk") != std::string::npos) {
        return "LINEAR";
    }
    if (lower_name.find("attn_v") != std::string::npos || lower_name.find("v_proj") != std::string::npos || lower_name.find(".wv") != std::string::npos) {
        return "LINEAR";
    }
    if (lower_name.find("attn_output") != std::string::npos || lower_name.find("o_proj") != std::string::npos || lower_name.find(".wo") != std::string::npos) {
        return "LINEAR";
    }
    if (lower_name.find("ffn_gate") != std::string::npos || lower_name.find("gate_proj") != std::string::npos || lower_name.find("w1") != std::string::npos) {
        return "LINEAR";
    }
    if (lower_name.find("ffn_down") != std::string::npos || lower_name.find("down_proj") != std::string::npos || lower_name.find("w2") != std::string::npos) {
        return "LINEAR";
    }
    if (lower_name.find("ffn_up") != std::string::npos || lower_name.find("up_proj") != std::string::npos || lower_name.find("w3") != std::string::npos) {
        return "LINEAR";
    }
    if (lower_name.find("token_embd") != std::string::npos || lower_name.find("tok_embeddings") != std::string::npos) {
        return "TOKEN_EMBEDDING";
    }
    if (lower_name.find("output") != std::string::npos) {
        return "LINEAR";
    }
    if (lower_name.find("norm") != std::string::npos) {
        return "NORM";
    }
    return "WEIGHTS";
}

std::string readMetadataValue(std::ifstream& file, GGUFMetadataType type, size_t depth = 0) {
    if (depth > 4) {
        throw ParsingError("GGUF metadata nesting is too deep");
    }

    switch (type) {
        case GGUFMetadataType::UINT8: return std::to_string(readBinary<uint8_t>(file, "metadata.uint8"));
        case GGUFMetadataType::INT8: return std::to_string(readBinary<int8_t>(file, "metadata.int8"));
        case GGUFMetadataType::UINT16: return std::to_string(readBinary<uint16_t>(file, "metadata.uint16"));
        case GGUFMetadataType::INT16: return std::to_string(readBinary<int16_t>(file, "metadata.int16"));
        case GGUFMetadataType::UINT32: return std::to_string(readBinary<uint32_t>(file, "metadata.uint32"));
        case GGUFMetadataType::INT32: return std::to_string(readBinary<int32_t>(file, "metadata.int32"));
        case GGUFMetadataType::UINT64: return std::to_string(readBinary<uint64_t>(file, "metadata.uint64"));
        case GGUFMetadataType::INT64: return std::to_string(readBinary<int64_t>(file, "metadata.int64"));
        case GGUFMetadataType::FLOAT32: {
            std::ostringstream stream;
            stream << readBinary<float>(file, "metadata.float32");
            return stream.str();
        }
        case GGUFMetadataType::FLOAT64: {
            std::ostringstream stream;
            stream << readBinary<double>(file, "metadata.float64");
            return stream.str();
        }
        case GGUFMetadataType::BOOL:
            return readBinary<uint8_t>(file, "metadata.bool") != 0 ? "true" : "false";
        case GGUFMetadataType::STRING:
            return readGGUFString(file);
        case GGUFMetadataType::ARRAY: {
            const auto element_type = static_cast<GGUFMetadataType>(readBinary<uint32_t>(file, "metadata.array.type"));
            const uint64_t element_count = readBinary<uint64_t>(file, "metadata.array.length");
            std::ostringstream stream;
            stream << "[";
            const uint64_t preview_count = std::min<uint64_t>(element_count, 16);
            for (uint64_t index = 0; index < preview_count; ++index) {
                if (index > 0) {
                    stream << ", ";
                }
                stream << readMetadataValue(file, element_type, depth + 1);
            }
            for (uint64_t index = preview_count; index < element_count; ++index) {
                (void)readMetadataValue(file, element_type, depth + 1);
            }
            if (element_count > preview_count) {
                stream << ", ...";
            }
            stream << "]";
            return stream.str();
        }
        default:
            throw ParsingError("Unsupported GGUF metadata type");
    }
}

std::vector<std::string> readMetadataStringArray(std::ifstream& file) {
    const auto element_type = static_cast<GGUFMetadataType>(readBinary<uint32_t>(file, "metadata.array.type"));
    if (element_type != GGUFMetadataType::STRING) {
        throw ParsingError("Expected GGUF string array");
    }

    const uint64_t element_count = readBinary<uint64_t>(file, "metadata.array.length");
    if (element_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw ParsingError("GGUF string array is too large");
    }

    std::vector<std::string> values;
    values.reserve(static_cast<size_t>(element_count));
    for (uint64_t index = 0; index < element_count; ++index) {
        values.push_back(readGGUFString(file));
    }
    return values;
}

std::string joinLines(const std::vector<std::string>& values) {
    std::ostringstream stream;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            stream << '\n';
        }
        stream << values[index];
    }
    return stream.str();
}

std::string summarizeStringArray(const std::vector<std::string>& values) {
    std::ostringstream stream;
    stream << "[" << values.size() << " strings";
    if (!values.empty()) {
        stream << ": ";
        const size_t preview_count = std::min<size_t>(values.size(), 4);
        for (size_t index = 0; index < preview_count; ++index) {
            if (index > 0) {
                stream << ", ";
            }
            stream << '"' << values[index] << '"';
        }
        if (values.size() > preview_count) {
            stream << ", ...";
        }
    }
    stream << "]";
    return stream.str();
}

std::vector<std::filesystem::path> discoverGGUFShards(const std::filesystem::path& model_path) {
    std::vector<std::filesystem::path> shards;
    const std::string filename = model_path.filename().string();
    // Typical split naming: model-00001-of-00002.gguf
    const std::regex shard_pattern(R"(^(.*-)(\d+)-of-(\d+)\.gguf$)", std::regex::icase);
    std::smatch matches;
    if (!std::regex_match(filename, matches, shard_pattern)) {
        shards.push_back(model_path);
        return shards;
    }

    const std::string prefix = matches[1].str();
    const std::string shard_str = matches[2].str();
    const std::string total_str = matches[3].str();
    size_t total_parts = 0;
    try {
        total_parts = static_cast<size_t>(std::stoull(total_str));
    } catch (const std::exception&) {
        shards.push_back(model_path);
        return shards;
    }
    if (total_parts == 0) {
        shards.push_back(model_path);
        return shards;
    }

    const size_t width = shard_str.size();
    shards.reserve(total_parts);
    for (size_t part = 1; part <= total_parts; ++part) {
        std::ostringstream name;
        name << prefix
             << std::setw(static_cast<int>(width))
             << std::setfill('0')
             << part
             << "-of-"
             << total_str
             << ".gguf";
        const auto shard_path = model_path.parent_path() / name.str();
        if (!std::filesystem::exists(shard_path)) {
            throw ParsingError("Missing GGUF shard: " + shard_path.string());
        }
        shards.push_back(shard_path);
    }
    return shards;
}

} // namespace

GGUFModelParser::GGUFModelParser() {}

GGUFModelParser::GGUFHeaderInfo GGUFModelParser::readHeader(std::ifstream& file) const {
    GGUFHeaderInfo header;

    const uint32_t magic = readBinary<uint32_t>(file, "header.magic");
    if (magic != GGUF_MAGIC) {
        throw ParsingError("Invalid GGUF magic number");
    }

    header.version = readBinary<uint32_t>(file, "header.version");
    if (header.version < GGUF_VERSION_MIN || header.version > GGUF_VERSION_MAX) {
        throw ParsingError("Unsupported GGUF version: " + std::to_string(header.version));
    }

    header.tensor_count = readBinary<uint64_t>(file, "header.tensor_count");
    header.metadata_count = readBinary<uint64_t>(file, "header.metadata_count");

    for (uint64_t index = 0; index < header.metadata_count; ++index) {
        const std::string key = readGGUFString(file);
        const auto type = static_cast<GGUFMetadataType>(readBinary<uint32_t>(file, "metadata.type"));

        if (type == GGUFMetadataType::ARRAY && key == "tokenizer.ggml.tokens") {
            header.tokenizer_tokens = readMetadataStringArray(file);
            header.metadata[key] = summarizeStringArray(header.tokenizer_tokens);
            continue;
        }

        const std::string value = readMetadataValue(file, type);
        header.metadata[key] = value;
        if (key == "general.architecture") {
            header.architecture = value;
        } else if (key == "general.name") {
            header.model_name = value;
        } else if (key == "tokenizer.ggml.model") {
            header.tokenizer_model = value;
        }
    }

    const auto alignment_it = header.metadata.find("general.alignment");
    if (alignment_it != header.metadata.end()) {
        try {
            const unsigned long parsed_alignment = std::stoul(alignment_it->second);
            if (parsed_alignment > 0 && parsed_alignment <= std::numeric_limits<uint32_t>::max()) {
                header.alignment = static_cast<uint32_t>(parsed_alignment);
            }
        } catch (const std::exception&) {
            header.alignment = 32;
        }
    }

    return header;
}

std::vector<GGUFModelParser::GGUFTensorInfo> GGUFModelParser::readTensorInfo(std::ifstream& file, GGUFHeaderInfo& header, size_t shard_index) const {
    std::vector<GGUFTensorInfo> tensors;
    tensors.reserve(static_cast<size_t>(header.tensor_count));

    for (uint64_t index = 0; index < header.tensor_count; ++index) {
        GGUFTensorInfo info;
        info.name = readGGUFString(file);

        const uint32_t num_dims = readBinary<uint32_t>(file, "tensor.num_dims");
        info.dimensions.reserve(num_dims);
        for (uint32_t dim_index = 0; dim_index < num_dims; ++dim_index) {
            const uint64_t dim = readBinary<uint64_t>(file, "tensor.dimension");
            if (dim > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw ParsingError("GGUF tensor dimension exceeds addressable memory");
            }
            info.dimensions.push_back(static_cast<size_t>(dim));
        }

        info.ggml_type = readBinary<uint32_t>(file, "tensor.ggml_type");
        info.data_type = getGGMLTypeTraits(info.ggml_type).name;
        info.offset = readBinary<uint64_t>(file, "tensor.offset");
        info.size = computeTensorByteSize(info.dimensions, info.ggml_type);
        info.shard_index = shard_index;
        tensors.push_back(std::move(info));
    }

    const auto current_offset = static_cast<uint64_t>(file.tellg());
    header.tensor_data_offset = alignOffset(current_offset, header.alignment);
    return tensors;
}

SegmentType GGUFModelParser::determineSegmentType(const std::string& tensorName, const std::string& dataType) const {
    const std::string lower_name = toLower(tensorName);
    const std::string lower_type = toLower(dataType);

    if (lower_name.find("token_embd") != std::string::npos ||
        lower_name.find("tok_embeddings") != std::string::npos ||
        lower_name.find("embedding") != std::string::npos ||
        lower_name.find("output.weight") != std::string::npos) {
        return SegmentType::EMBEDDING_WEIGHTS;
    }
    if (lower_name.find("attn") != std::string::npos ||
        lower_name.find("attention") != std::string::npos) {
        return SegmentType::ATTENTION_WEIGHTS;
    }
    if (lower_name.find("ffn") != std::string::npos ||
        lower_name.find("feed_forward") != std::string::npos ||
        lower_name.find("mlp") != std::string::npos) {
        return SegmentType::FEED_FORWARD_WEIGHTS;
    }
    if (lower_name.find("norm") != std::string::npos ||
        lower_name.find("ln_") != std::string::npos) {
        return SegmentType::LAYER_NORM_WEIGHTS;
    }

    if (lower_type == "f32" || lower_type == "f64") {
        return SegmentType::WEIGHTS_FP32;
    }
    if (lower_type == "f16" || lower_type == "bf16") {
        return SegmentType::WEIGHTS_FP16;
    }
    if (lower_type == "i8" || lower_type == "q8_0" || lower_type == "q8_1" || lower_type == "q8_k" || lower_type == "q6_k") {
        return SegmentType::WEIGHTS_INT8;
    }
    if (isQuantizedGGUFType(dataType)) {
        return SegmentType::WEIGHTS_INT4;
    }

    return SegmentType::UNKNOWN;
}

TensorMetadata GGUFModelParser::extractTensorMetadata(const GGUFTensorInfo& info) const {
    TensorMetadata metadata;
    metadata.dimensions = info.dimensions;
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = false;

    if (isQuantizedGGUFType(info.data_type)) {
        metadata.scale = 1.0f;
        metadata.zero_point = 0.0f;
    }

    return metadata;
}

std::string GGUFModelParser::extractLayerName(const std::string& tensorName) const {
    return stripTensorSuffix(tensorName);
}

size_t GGUFModelParser::extractLayerIndex(const std::string& tensorName) const {
    static const std::array<std::regex, 4> indexed_patterns = {
        std::regex(R"((?:^|\.)(?:blk|block|layer|layers)\.(\d+)(?:\.|$))"),
        std::regex(R"((?:^|\.)(?:h)\.(\d+)(?:\.|$))"),
        std::regex(R"((?:^|\.)(\d+)\.(?:attn|ffn|mlp|norm)(?:\.|$))"),
        std::regex(R"((?:transformer|model)\.(?:h|layers)\.(\d+)(?:\.|$))")
    };

    std::smatch matches;
    for (const auto& pattern : indexed_patterns) {
        if (std::regex_search(tensorName, matches, pattern)) {
            try {
                return static_cast<size_t>(std::stoull(matches[1].str()));
            } catch (const std::exception&) {
                return 0;
            }
        }
    }
    return 0;
}

ModelSegment GGUFModelParser::readTensor(std::ifstream& file, const GGUFHeaderInfo& header, const GGUFTensorInfo& info) const {
    ModelSegment segment;
    segment.name = info.name;
    segment.type = determineSegmentType(info.name, info.data_type);
    segment.tensor_metadata = extractTensorMetadata(info);
    segment.layer_name = extractLayerName(info.name);
    segment.layer_index = extractLayerIndex(info.name);
    segment.data_format = info.data_type;
    segment.layer_type = classifyTensorRole(info.name);

    if (info.dimensions.size() >= 2) {
        segment.input_shape = {1, info.dimensions[0]};
        segment.output_shape = {1, info.dimensions[1]};
    } else if (info.dimensions.size() == 1) {
        segment.input_shape = {1, info.dimensions[0]};
        segment.output_shape = {1, info.dimensions[0]};
    }

    if (segment.layer_type == "WEIGHTS") {
        if (segment.type == SegmentType::ATTENTION_WEIGHTS) {
            segment.layer_type = "LINEAR";
        } else if (segment.type == SegmentType::FEED_FORWARD_WEIGHTS) {
            segment.layer_type = "LINEAR";
        } else if (segment.type == SegmentType::EMBEDDING_WEIGHTS) {
            segment.layer_type = "EMBEDDING";
        } else if (segment.type == SegmentType::LAYER_NORM_WEIGHTS) {
            segment.layer_type = "NORM";
        }
    }

    const uint64_t absolute_offset = header.tensor_data_offset + info.offset;
    file.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    if (!file) {
        throw ParsingError("Failed to seek to GGUF tensor data for " + info.name);
    }

    if (info.size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw ParsingError("GGUF tensor too large to fit in memory: " + info.name);
    }
    segment.data.resize(static_cast<size_t>(info.size));
    if (!segment.data.empty()) {
        file.read(reinterpret_cast<char*>(segment.data.data()), static_cast<std::streamsize>(segment.data.size()));
        if (!file) {
            throw ParsingError("Failed to read GGUF tensor data for " + info.name);
        }
    }
    segment.original_size = segment.data.size();

    return segment;
}

ModelSegment GGUFModelParser::createMetadataSegment(const GGUFHeaderInfo& header) const {
    std::ostringstream json;
    json << "{";
    json << "\"format\":\"GGUF\",";
    json << "\"version\":" << header.version << ",";
    json << "\"tensor_count\":" << header.tensor_count << ",";
    json << "\"alignment\":" << header.alignment << ",";
    json << "\"metadata\":{";

    bool first = true;
    for (const auto& [key, value] : header.metadata) {
        if (!first) {
            json << ",";
        }
        first = false;
        json << "\"" << escapeJson(key) << "\":\"" << escapeJson(value) << "\"";
    }

    json << "}}";

    const std::string payload = json.str();
    ModelSegment segment;
    segment.name = "gguf_metadata";
    segment.type = SegmentType::METADATA_JSON;
    segment.data.assign(
        reinterpret_cast<const std::byte*>(payload.data()),
        reinterpret_cast<const std::byte*>(payload.data() + payload.size())
    );
    segment.original_size = segment.data.size();
    segment.data_format = "GGUF";
    segment.layer_type = "METADATA";
    return segment;
}

ModelSegment GGUFModelParser::createConfigSegment(const GGUFHeaderInfo& header) const {
    std::ostringstream json;
    json << "{";
    json << "\"format\":\"GGUF\",";
    json << "\"version\":" << header.version << ",";
    json << "\"architecture\":\"" << escapeJson(header.architecture) << "\",";
    json << "\"name\":\"" << escapeJson(header.model_name) << "\",";
    json << "\"alignment\":" << header.alignment << ",";
    json << "\"tensor_count\":" << header.tensor_count;
    json << "}";

    const std::string payload = json.str();
    ModelSegment segment;
    segment.name = "gguf_config";
    segment.type = SegmentType::CONFIG;
    segment.data.assign(
        reinterpret_cast<const std::byte*>(payload.data()),
        reinterpret_cast<const std::byte*>(payload.data() + payload.size())
    );
    segment.original_size = segment.data.size();
    segment.data_format = "GGUF";
    segment.layer_type = "CONFIG";
    return segment;
}

ModelSegment GGUFModelParser::createTokenizerVocabSegment(const GGUFHeaderInfo& header) const {
    const std::string payload = joinLines(header.tokenizer_tokens);
    ModelSegment segment;
    segment.name = "gguf_tokenizer_vocab";
    segment.type = SegmentType::TOKENIZER_VOCAB;
    segment.data.assign(
        reinterpret_cast<const std::byte*>(payload.data()),
        reinterpret_cast<const std::byte*>(payload.data() + payload.size())
    );
    segment.original_size = segment.data.size();
    segment.data_format = "GGUF";
    segment.layer_type = "TOKENIZER";
    return segment;
}

ModelSegment GGUFModelParser::createTokenizerModelSegment(const GGUFHeaderInfo& header) const {
    ModelSegment segment;
    segment.name = "gguf_tokenizer_model";
    segment.type = SegmentType::TOKENIZER_MODEL;
    segment.data.assign(
        reinterpret_cast<const std::byte*>(header.tokenizer_model.data()),
        reinterpret_cast<const std::byte*>(header.tokenizer_model.data() + header.tokenizer_model.size())
    );
    segment.original_size = segment.data.size();
    segment.data_format = "GGUF";
    segment.layer_type = "TOKENIZER";
    return segment;
}

std::vector<ModelSegment> GGUFModelParser::parse(const std::string& modelPath) const {
    const std::filesystem::path model_path(modelPath);
    const auto shard_paths = discoverGGUFShards(model_path);

    std::vector<std::ifstream> shard_files;
    shard_files.reserve(shard_paths.size());
    std::vector<GGUFHeaderInfo> shard_headers;
    shard_headers.reserve(shard_paths.size());
    std::vector<GGUFTensorInfo> tensor_infos;
    tensor_infos.reserve(1024);
    std::unordered_set<std::string> seen_tensor_names;
    seen_tensor_names.reserve(2048);

    for (size_t shard_index = 0; shard_index < shard_paths.size(); ++shard_index) {
        shard_files.emplace_back(shard_paths[shard_index], std::ios::binary);
        if (!shard_files.back()) {
            throw ParsingError("Failed to open model shard: " + shard_paths[shard_index].string());
        }
        GGUFHeaderInfo header = readHeader(shard_files.back());
        auto shard_tensors = readTensorInfo(shard_files.back(), header, shard_index);
        for (auto& tensor : shard_tensors) {
            // Avoid duplicate tensor descriptors across shards.
            if (seen_tensor_names.insert(tensor.name).second) {
                tensor_infos.push_back(std::move(tensor));
            }
        }
        shard_headers.push_back(std::move(header));
    }

    if (shard_headers.empty()) {
        throw ParsingError("No GGUF header information found for: " + modelPath);
    }
    const GGUFHeaderInfo& primary_header = shard_headers.front();

    std::vector<uint64_t> shard_data_sizes(shard_paths.size(), 0);
    std::vector<uint64_t> shard_data_prefix(shard_paths.size(), 0);
    uint64_t cumulative_data = 0;
    for (size_t shard_index = 0; shard_index < shard_paths.size(); ++shard_index) {
        const uint64_t file_size = static_cast<uint64_t>(std::filesystem::file_size(shard_paths[shard_index]));
        const uint64_t data_start = shard_headers[shard_index].tensor_data_offset;
        shard_data_prefix[shard_index] = cumulative_data;
        if (file_size > data_start) {
            shard_data_sizes[shard_index] = file_size - data_start;
            cumulative_data += shard_data_sizes[shard_index];
        } else {
            shard_data_sizes[shard_index] = 0;
        }
    }

    std::vector<GGUFTensorInfo> resolved_infos;
    resolved_infos.reserve(tensor_infos.size());
    const auto resolveTensorOffsetForShards =
        [&shard_data_sizes, &shard_data_prefix](GGUFTensorInfo& info) -> bool {
            if (info.shard_index < shard_data_sizes.size()) {
                const uint64_t shard_cap = shard_data_sizes[info.shard_index];
                if (info.offset <= shard_cap && info.size <= (shard_cap - info.offset)) {
                    return true;
                }
            }

            // Fallback: some split GGUF variants encode offsets on the concatenated data stream.
            for (size_t shard = 0; shard < shard_data_sizes.size(); ++shard) {
                const uint64_t prefix = shard_data_prefix[shard];
                const uint64_t cap = shard_data_sizes[shard];
                if (info.offset < prefix) {
                    continue;
                }
                const uint64_t local = info.offset - prefix;
                if (local <= cap && info.size <= (cap - local)) {
                    info.shard_index = shard;
                    info.offset = local;
                    return true;
                }
            }
            return false;
        };

    for (auto info : tensor_infos) {
        if (resolveTensorOffsetForShards(info)) {
            resolved_infos.push_back(std::move(info));
        }
    }

    std::vector<ModelSegment> segments;
    segments.reserve(resolved_infos.size() + 4);
    segments.push_back(createMetadataSegment(primary_header));
    segments.push_back(createConfigSegment(primary_header));
    if (!primary_header.tokenizer_model.empty()) {
        segments.push_back(createTokenizerModelSegment(primary_header));
    }
    if (!primary_header.tokenizer_tokens.empty()) {
        segments.push_back(createTokenizerVocabSegment(primary_header));
    }

    for (const auto& info : resolved_infos) {
        const size_t shard_index = info.shard_index;
        if (shard_index >= shard_files.size() || shard_index >= shard_headers.size()) {
            throw ParsingError("Invalid GGUF tensor shard index for tensor: " + info.name);
        }
        segments.push_back(readTensor(shard_files[shard_index], shard_headers[shard_index], info));
    }

    return segments;
}

std::vector<ModelSegment> GGUFModelParser::parseWithChunking(const std::string& modelPath) const {
    auto segments = parse(modelPath);

    std::map<size_t, std::vector<ModelSegment*>> layer_groups;
    std::vector<ModelSegment> metadata_segments;
    metadata_segments.reserve(1);

    for (auto& segment : segments) {
        if (segment.type == SegmentType::METADATA_JSON) {
            metadata_segments.push_back(segment);
            continue;
        }
        layer_groups[segment.layer_index].push_back(&segment);
    }

    for (auto& [layer, group] : layer_groups) {
        std::sort(group.begin(), group.end(), [](const ModelSegment* lhs, const ModelSegment* rhs) {
            if (lhs->layer_name != rhs->layer_name) {
                return lhs->layer_name < rhs->layer_name;
            }
            if (lhs->type != rhs->type) {
                return static_cast<int>(lhs->type) < static_cast<int>(rhs->type);
            }
            return lhs->name < rhs->name;
        });
    }

    std::vector<ModelSegment> reordered_segments;
    reordered_segments.reserve(segments.size());
    for (const auto& metadata_segment : metadata_segments) {
        reordered_segments.push_back(metadata_segment);
    }

    for (const auto& segment : segments) {
        if (segment.type != SegmentType::METADATA_JSON &&
            segment.layer_index == 0 &&
            (segment.layer_name.empty() || segment.layer_name == stripTensorSuffix(segment.name))) {
            reordered_segments.push_back(segment);
        }
    }

    for (const auto& [layer, group] : layer_groups) {
        for (const auto* segment : group) {
            const bool already_added =
                segment->layer_index == 0 &&
                (segment->layer_name.empty() || segment->layer_name == stripTensorSuffix(segment->name));
            if (!already_added) {
                reordered_segments.push_back(*segment);
            }
        }
    }

    return reordered_segments;
}

} // namespace CortexAICompression
