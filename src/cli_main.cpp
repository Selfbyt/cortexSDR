#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept> // For exception handling
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

// Function to serialize EncodedData to a file
void writeSdrFile(const std::string& filePath, const SparseDistributedRepresentation::EncodedData& data) {
    std::ofstream fileStream(filePath, std::ios::binary | std::ios::trunc);
    if (!fileStream) {
        throw std::runtime_error("Error: Could not open SDR file for writing: " + filePath);
    }

    // Write totalSize
    fileStream.write(reinterpret_cast<const char*>(&data.totalSize), sizeof(data.totalSize));

    // Write number of active positions
    size_t numActivePositions = data.activePositions.size();
    fileStream.write(reinterpret_cast<const char*>(&numActivePositions), sizeof(numActivePositions));

    // Write active positions
    fileStream.write(reinterpret_cast<const char*>(data.activePositions.data()), numActivePositions * sizeof(size_t));

    if (!fileStream) {
         throw std::runtime_error("Error: Failed to write to SDR file: " + filePath);
    }
}

// Function to deserialize EncodedData from a file
SparseDistributedRepresentation::EncodedData readSdrFile(const std::string& filePath) {
    std::ifstream fileStream(filePath, std::ios::binary);
    if (!fileStream) {
        throw std::runtime_error("Error: Could not open SDR file for reading: " + filePath);
    }

    SparseDistributedRepresentation::EncodedData data;

    // Read totalSize
    fileStream.read(reinterpret_cast<char*>(&data.totalSize), sizeof(data.totalSize));
    if (!fileStream) throw std::runtime_error("Error: Failed to read totalSize from SDR file: " + filePath);


    // Read number of active positions
    size_t numActivePositions = 0;
    fileStream.read(reinterpret_cast<char*>(&numActivePositions), sizeof(numActivePositions));
     if (!fileStream) throw std::runtime_error("Error: Failed to read numActivePositions from SDR file: " + filePath);


    // Read active positions
    data.activePositions.resize(numActivePositions);
    fileStream.read(reinterpret_cast<char*>(data.activePositions.data()), numActivePositions * sizeof(size_t));
    if (!fileStream) throw std::runtime_error("Error: Failed to read activePositions from SDR file: " + filePath);


    // Check for extra data (optional, basic check)
    fileStream.peek();
    if (!fileStream.eof()) {
         std::cerr << "Warning: Extra data found in SDR file: " << filePath << std::endl;
    }


    return data;
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

    // Initialize SDR (using a default or configurable vocabulary might be needed)
    // A real implementation might load/save the vocabulary with the SDR data.
    SparseDistributedRepresentation sdr({"the", "a", "is", "in", "it", "of", "and", "to",
                                         "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                                         "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
                                         "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
                                         "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua"}); // Expanded placeholder

    try {
        if (mode == "-c") {
            // Compress Mode
            std::cout << "Compressing " << inputFile << " to " << outputFile << "..." << std::endl;

            std::string fileContent = readFileContent(inputFile);
            std::cout << "Original size: " << fileContent.size() << " bytes" << std::endl;

            SparseDistributedRepresentation::EncodedData encoded = sdr.encodeText(fileContent); // Assuming text encoding for now
            writeSdrFile(outputFile, encoded);

            size_t encodedSize = encoded.getMemorySize(); // This is in-memory size, file size might differ slightly
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

            SparseDistributedRepresentation::EncodedData encodedData = readSdrFile(inputFile);
            sdr.setEncoding(encodedData); // Load the data into the SDR instance
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
