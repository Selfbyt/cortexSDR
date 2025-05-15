#ifndef AI_MODEL_PARSER_HPP
#define AI_MODEL_PARSER_HPP

#include "ModelSegment.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <memory> // For std::unique_ptr

namespace CortexAICompression {

// Base class for parsing errors
class ParsingError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Interface for AI model parsers
class IAIModelParser {
public:
    virtual ~IAIModelParser() = default;

    // Parses the model file at the given path and returns its segments.
    // Throws ParsingError on failure.
    virtual std::vector<ModelSegment> parse(const std::string& modelPath) const = 0;

    // Optional: Method to identify if the parser supports a given file format
    // virtual bool supportsFormat(const std::string& modelPath) const = 0;

    // New method for model-aware chunking. This allows parsers to split the model into segments at optimal boundaries
    // (e.g., per-layer, per-tensor) for better compression. The default implementation returns the same as parse().
    virtual std::vector<ModelSegment> parseWithChunking(const std::string& modelPath) const {
        return parse(modelPath);
    }
};

// Factory function to create a parser based on file type (implementation needed)
// std::unique_ptr<IAIModelParser> createParserForModel(const std::string& modelPath);

} // namespace CortexAICompression

#endif // AI_MODEL_PARSER_HPP
