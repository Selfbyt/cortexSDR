#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept> // For exception handling
#include <cstdint>
#include "cortexSDR.hpp"

// Function to read file content into a string
std::string readFileContent(const std::string& filePath) {
    std::ifstream fileStream(filePath, std::ios::binary); // Read as binary
    if (!fileStream) {
        throw std::runtime_error("Error: Could not open file: " + filePath);
    }
    std::stringstream buffer;
    buffer << fileStream.rdbuf();
    return buffer.str();
}

// Function to write string content to a file
void writeFileContent(const std::string& filePath, const std::string& content) {
    std::ofstream fileStream(filePath, std::ios::binary); // Write as binary
    if (!fileStream) {
        throw std::runtime_error("Error: Could not open file for writing: " + filePath);
    }
    fileStream << content;
}

// Function to serialize EncodedData AND vocabulary to a file
void writeSdrFile(const std::string& filePath, const SparseDistributedRepresentation& sdr) {
    std::ofstream fileStream(filePath, std::ios::binary | std::ios::trunc);
    if (!fileStream) {
        throw std::runtime_error("Error: Could not open SDR file for writing: " + filePath);
    }

    const auto& data = sdr.getEncodedData(); // Get encoded data from SDR object
    const auto& vocabulary = sdr.getWordVocabulary(); // Get vocabulary from SDR object

    // --- Write EncodedData ---
    // Write totalSize
    fileStream.write(reinterpret_cast<const char*>(&data.totalSize), sizeof(data.totalSize));

    // Write number of active positions (compact)
    uint32_t numActivePositions = static_cast<uint32_t>(data.activePositions.size());
    fileStream.write(reinterpret_cast<const char*>(&numActivePositions), sizeof(numActivePositions));
    if (numActivePositions > 0) {
        std::vector<uint16_t> positions16;
        positions16.reserve(numActivePositions);
        for (size_t i = 0; i < numActivePositions; ++i) {
            positions16.push_back(static_cast<uint16_t>(data.activePositions[i]));
        }
        fileStream.write(reinterpret_cast<const char*>(positions16.data()), numActivePositions * sizeof(uint16_t));
    }

    // --- Write Vocabulary ---
    // Write number of words in vocabulary
    size_t vocabSize = vocabulary.size();
    fileStream.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));

    // Write each word (length followed by characters)
    for (const std::string& word : vocabulary) {
        size_t wordLength = word.length();
        fileStream.write(reinterpret_cast<const char*>(&wordLength), sizeof(wordLength));
        fileStream.write(word.c_str(), wordLength);
    }

    if (!fileStream) {
         throw std::runtime_error("Error: Failed to write to SDR file: " + filePath);
    }
}

// Function to deserialize EncodedData AND vocabulary from a file into an SDR object
void readSdrFile(const std::string& filePath, SparseDistributedRepresentation& sdr) {
    std::ifstream fileStream(filePath, std::ios::binary);
    if (!fileStream) {
        throw std::runtime_error("Error: Could not open SDR file for reading: " + filePath);
    }

    // --- Read EncodedData ---
    SparseDistributedRepresentation::EncodedData data;
    // Read totalSize
    fileStream.read(reinterpret_cast<char*>(&data.totalSize), sizeof(data.totalSize));
    if (!fileStream) throw std::runtime_error("Error: Failed to read totalSize from SDR file: " + filePath);

    // Read number of active positions (compact)
    uint32_t numActivePositions = 0;
    fileStream.read(reinterpret_cast<char*>(&numActivePositions), sizeof(numActivePositions));
    if (!fileStream) throw std::runtime_error("Error: Failed to read numActivePositions from SDR file: " + filePath);
    if (numActivePositions > 0) {
        std::vector<uint16_t> positions16(numActivePositions);
        fileStream.read(reinterpret_cast<char*>(positions16.data()), numActivePositions * sizeof(uint16_t));
        if (!fileStream) throw std::runtime_error("Error: Failed to read activePositions from SDR file: " + filePath);
        data.activePositions.resize(numActivePositions);
        for (uint32_t i = 0; i < numActivePositions; ++i) {
            data.activePositions[i] = positions16[i];
        }
    } else {
        data.activePositions.clear();
    }

    // Set the encoded data in the SDR object
    sdr.setEncoding(data);

    // --- Read Vocabulary ---
    std::vector<std::string> vocabulary;
    size_t vocabSize = 0;
    fileStream.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
    if (!fileStream) throw std::runtime_error("Error: Failed to read vocabSize from SDR file: " + filePath);

    vocabulary.reserve(vocabSize);
    for (size_t i = 0; i < vocabSize; ++i) {
        size_t wordLength = 0;
        fileStream.read(reinterpret_cast<char*>(&wordLength), sizeof(wordLength));
        if (!fileStream) throw std::runtime_error("Error: Failed to read wordLength for word " + std::to_string(i) + " from SDR file: " + filePath);

        if (wordLength > 0) {
             std::string word(wordLength, '\0');
             fileStream.read(&word[0], wordLength);
             if (!fileStream) throw std::runtime_error("Error: Failed to read word " + std::to_string(i) + " from SDR file: " + filePath);
             vocabulary.push_back(word);
        } else {
             vocabulary.push_back(""); // Handle empty strings if necessary
        }
    }

    // Set the vocabulary in the SDR object (this will also rebuild the internal map)
    sdr.setWordVocabulary(vocabulary);

    // Check for extra data (optional, basic check)
    fileStream.peek();
    if (!fileStream.eof()) {
         std::cerr << "Warning: Extra data found at the end of SDR file: " << filePath << std::endl;
    }
}


void printUsage(const char* progName) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << progName << " -c <input_file> <output_file.sdr>   (Compress)" << std::endl;
    std::cerr << "  " << progName << " -d <input_file.sdr> <output_file> (Decompress)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    std::string inputFile = argv[2];
    std::string outputFile = argv[3];

    // Initialize SDR. Vocabulary is now handled dynamically by WordEncoding.
    SparseDistributedRepresentation sdr; 

    try {
        if (mode == "-c") {
            // Compress Mode
            std::cout << "Compressing " << inputFile << " to " << outputFile << "..." << std::endl;

            std::string fileContent = readFileContent(inputFile);
            std::cout << "Original size: " << fileContent.size() << " bytes" << std::endl;

            sdr.encodeText(fileContent); // Encode text, populates sdr internal state
            writeSdrFile(outputFile, sdr); // Pass the sdr object to write data + vocabulary

            size_t encodedSize = sdr.getEncodedData().getMemorySize(); // Get size from sdr object
             std::cout << "Encoded SDR in-memory size: " << encodedSize << " bytes" << std::endl;
             // Calculate file size for more accurate ratio
             std::ifstream sdrFile(outputFile, std::ios::binary | std::ios::ate);
             size_t sdrFileSize = sdrFile.tellg();
             std::cout << "Output SDR file size: " << sdrFileSize << " bytes" << std::endl;


            if (sdrFileSize > 0) {
                 double compressionRatio = static_cast<double>(fileContent.size()) / sdrFileSize;
                 std::cout << "Compression ratio (file size): " << compressionRatio << ":1" << std::endl;
            } else {
                 std::cout << "Compression ratio: N/A (output file size is zero)" << std::endl;
            }

            std::cout << "Compression complete." << std::endl;

        } else if (mode == "-d") {
            // Decompress Mode
            std::cout << "Decompressing " << inputFile << " to " << outputFile << "..." << std::endl;

            readSdrFile(inputFile, sdr); // Read data and vocabulary directly into sdr
            // No need for sdr.setEncoding(encodedData) anymore
            std::string decodedContent = sdr.decode();

            writeFileContent(outputFile, decodedContent);

            std::cout << "Decompression complete. Output written to " << outputFile << std::endl;

        } else {
            std::cerr << "Error: Invalid mode '" << mode << "'." << std::endl; // Corrected typo here
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
