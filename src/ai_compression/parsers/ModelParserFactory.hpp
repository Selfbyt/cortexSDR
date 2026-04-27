/**
 * @file ModelParserFactory.hpp
 * @brief Factory for creating appropriate model parsers based on file format
 */
#ifndef MODEL_PARSER_FACTORY_HPP
#define MODEL_PARSER_FACTORY_HPP

#include "../core/AIModelParser.hpp"
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <vector>

namespace CortexAICompression {

/**
 * @brief Factory class for creating model parsers based on file format detection
 */
class ModelParserFactory {
public:
    /**
     * @brief Create a parser for the given model file
     * @param modelPath Path to the model file
     * @return Unique pointer to the appropriate parser
     * @throws ParsingError if no suitable parser is found
     */
    static std::unique_ptr<IAIModelParser> createParser(const std::string& modelPath);

    /**
     * @brief Create a parser for an explicit format name.
     * @param format Format identifier (e.g., "onnx", "gguf")
     * @return Unique pointer to the requested parser
     * @throws ParsingError if format is unsupported
     */
    static std::unique_ptr<IAIModelParser> createParserForFormat(const std::string& format);
    
    /**
     * @brief Detect the model format from file extension and content
     * @param modelPath Path to the model file
     * @return String identifying the detected format
     */
    static std::string detectFormat(const std::string& modelPath);
    
    /**
     * @brief Check if a specific format is supported
     * @param format Format identifier (e.g., "onnx", "tensorflow", "pytorch")
     * @return True if the format is supported
     */
    static bool isFormatSupported(const std::string& format);
    
    /**
     * @brief Get list of all supported formats
     * @return Vector of supported format strings
     */
    static std::vector<std::string> getSupportedFormats();

private:
    // Format detection based on file extension
    static std::string detectFromExtension(const std::string& modelPath);
    
    // Format detection based on file content (magic numbers, etc.)
    static std::string detectFromContent(const std::string& modelPath);
    
    // Registry of available parsers
    static std::map<std::string, std::function<std::unique_ptr<IAIModelParser>()>> parserRegistry;
    
    // Initialize the parser registry
    static void initializeRegistry();
};

} // namespace CortexAICompression

#endif // MODEL_PARSER_FACTORY_HPP
