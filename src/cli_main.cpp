#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept> // For exception handling
#include <cstdint>
#include "cortexSDR.hpp"
#include "rle.hpp"
#include <zlib.h>
#include <cstdio>

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

// Function to serialize EncodedData AND vocabulary to a file (bit-packed SDR)
void writeSdrFile(const std::string& filePath, const SparseDistributedRepresentation& sdr) {
    std::ofstream fileStream(filePath, std::ios::binary | std::ios::trunc);
    if (!fileStream) {
        throw std::runtime_error("Error: Could not open SDR file for writing: " + filePath);
    }

    const auto& data = sdr.getEncodedData(); // Get encoded data from SDR object
    const auto& vocabulary = sdr.getWordVocabulary(); // Get vocabulary from SDR object

    // --- Write version byte for bit-packed format ---
    const uint8_t sdrFormatVersion = 1; // 1 = bit-packed
    fileStream.write(reinterpret_cast<const char*>(&sdrFormatVersion), sizeof(sdrFormatVersion));

    // --- Write EncodedData as bit-packed vector ---
    // Write totalSize (for compatibility)
    fileStream.write(reinterpret_cast<const char*>(&data.totalSize), sizeof(data.totalSize));

    // Bit-pack the SDR (e.g., 2000 bits -> 250 bytes)
    size_t bitCount = data.totalSize;
    size_t byteCount = (bitCount + 7) / 8;
    std::vector<uint8_t> bitPacked(byteCount, 0);
    for (size_t pos : data.activePositions) {
        if (pos < bitCount) {
            bitPacked[pos / 8] |= (1 << (pos % 8));
        }
    }
    fileStream.write(reinterpret_cast<const char*>(bitPacked.data()), byteCount);

    // --- Write Vocabulary (delta-encoded) ---
    // Sort vocabulary for better delta encoding
    std::vector<std::string> sortedVocab = vocabulary;
    std::sort(sortedVocab.begin(), sortedVocab.end());
    size_t vocabSize = sortedVocab.size();
    fileStream.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));
    std::string prevWord;
    for (size_t i = 0; i < vocabSize; ++i) {
        const std::string& word = sortedVocab[i];
        // Compute common prefix length
        size_t prefixLen = 0;
        while (prefixLen < prevWord.size() && prefixLen < word.size() && prevWord[prefixLen] == word[prefixLen]) {
            ++prefixLen;
        }
        size_t suffixLen = word.size() - prefixLen;
        fileStream.write(reinterpret_cast<const char*>(&prefixLen), sizeof(prefixLen));
        fileStream.write(reinterpret_cast<const char*>(&suffixLen), sizeof(suffixLen));
        fileStream.write(word.data() + prefixLen, suffixLen);
        prevWord = word;
    }

    // Added debug statements to log vocabulary during serialization.
    std::cout << "Serialized vocabulary: \n";
    for (const auto& word : sortedVocab) {
        std::cout << word << " ";
    }
    std::cout << std::endl;

    if (!fileStream) {
         throw std::runtime_error("Error: Failed to write to SDR file: " + filePath);
    }
}

// Function to deserialize EncodedData AND vocabulary from a file into an SDR object (bit-packed SDR)
void readSdrFile(const std::string& filePath, SparseDistributedRepresentation& sdr) {
    std::ifstream fileStream(filePath, std::ios::binary);
    if (!fileStream) {
        throw std::runtime_error("Error: Could not open SDR file for reading: " + filePath);
    }

    // --- Read version byte ---
    uint8_t sdrFormatVersion = 0;
    fileStream.read(reinterpret_cast<char*>(&sdrFormatVersion), sizeof(sdrFormatVersion));
    if (!fileStream) throw std::runtime_error("Error: Failed to read SDR format version from file: " + filePath);

    SparseDistributedRepresentation::EncodedData data;

    if (sdrFormatVersion == 1) {
        // --- Bit-packed format ---
        // Read totalSize
        fileStream.read(reinterpret_cast<char*>(&data.totalSize), sizeof(data.totalSize));
        if (!fileStream) throw std::runtime_error("Error: Failed to read totalSize from SDR file: " + filePath);
        size_t bitCount = data.totalSize;
        size_t byteCount = (bitCount + 7) / 8;
        std::vector<uint8_t> bitPacked(byteCount, 0);
        fileStream.read(reinterpret_cast<char*>(bitPacked.data()), byteCount);
        if (!fileStream) throw std::runtime_error("Error: Failed to read bit-packed SDR from file: " + filePath);
        data.activePositions.clear();
        for (size_t i = 0; i < bitCount; ++i) {
            if (bitPacked[i / 8] & (1 << (i % 8))) {
                data.activePositions.push_back(i);
            }
        }
    } else {
        // --- Legacy format fallback (positions as uint16_t) ---
        // Already read the first byte, so seek back to 0
        fileStream.seekg(0, std::ios::beg);
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
    }

    // Set the encoded data in the SDR object
    sdr.setEncoding(data);

    // --- Read Vocabulary (delta-decoded) ---
    std::vector<std::string> vocabulary;
    size_t vocabSize = 0;
    fileStream.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
    if (!fileStream) throw std::runtime_error("Error: Failed to read vocabSize from SDR file: " + filePath);
    vocabulary.reserve(vocabSize);
    std::string prevWord;
    for (size_t i = 0; i < vocabSize; ++i) {
        size_t prefixLen = 0, suffixLen = 0;
        fileStream.read(reinterpret_cast<char*>(&prefixLen), sizeof(prefixLen));
        fileStream.read(reinterpret_cast<char*>(&suffixLen), sizeof(suffixLen));
        if (!fileStream) throw std::runtime_error("Error: Failed to read prefix/suffixLen for vocab word " + std::to_string(i) + " from SDR file: " + filePath);
        std::string word = prevWord.substr(0, prefixLen);
        if (suffixLen > 0) {
            std::string suffix(suffixLen, '\0');
            fileStream.read(&suffix[0], suffixLen);
            if (!fileStream) throw std::runtime_error("Error: Failed to read vocab word suffix " + std::to_string(i) + " from SDR file: " + filePath);
            word += suffix;
        }
        vocabulary.push_back(word);
        prevWord = word;
    }

    // Set the vocabulary in the SDR object (this will also rebuild the internal map)
    sdr.setWordVocabulary(vocabulary);

    // Debug: Log the restored vocabulary
    std::cout << "Restored vocabulary: \n";
    for (const auto& word : vocabulary) {
        std::cout << word << " ";
    }
    std::cout << std::endl;

    // Check for extra data (optional, basic check)
    fileStream.peek();
    if (!fileStream.eof()) {
         std::cerr << "Warning: Extra data found at the end of SDR file: " << filePath << std::endl;
    }
}

// Utility: check if string ends with suffix
static bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Function to print usage
void printUsage(const char* progName) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << progName << " -c [--sparsity <frac>] [--rle] <input_file> <output_file.sdr[.gz]>   (Compress)" << std::endl;
    std::cerr << "  " << progName << " -d [--rle] <input_file.sdr[.gz]> <output_file> (Decompress)" << std::endl;
    std::cerr << "\nOptions:" << std::endl;
    std::cerr << "  --sparsity, -s <frac>  Set sparsity (fraction of active bits, e.g. 0.002 for 0.2%). Default: 0.002" << std::endl;
    std::cerr << "  --rle                  Enable run-length encoding before gzip (max compression, lossy)" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse args: support --sparsity/-s and --rle
    double sparsity = 0.002;
    bool useRle = false;
    std::string mode;
    std::string inputFile, outputFile;
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "-c" || arg == "-d") {
            mode = arg;
            ++i;
            break;
        } else if (arg == "--sparsity" || arg == "-s") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --sparsity requires a value." << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            sparsity = std::stod(argv[++i]);
            ++i;
        } else if (arg == "--rle") {
            useRle = true;
            ++i;
        } else {
            break;
        }
    }
    if (mode.empty() || (argc - i) < 2) {
        printUsage(argv[0]);
        return 1;
    }
    inputFile = argv[i++];
    outputFile = argv[i++];

    SparseDistributedRepresentation sdr;
    sdr.setSparsity(sparsity);

    try {
        if (mode == "-c") {
            // Compress Mode
            std::cout << "Compressing " << inputFile << " to " << outputFile << "..." << std::endl;
            std::string fileContent = readFileContent(inputFile);
            std::cout << "Original size: " << fileContent.size() << " bytes" << std::endl;
            sdr.encodeText(fileContent);

            // Bit-pack SDR
            const auto& data = sdr.getEncodedData();
            size_t bitCount = data.totalSize;
            size_t byteCount = (bitCount + 7) / 8;
            std::vector<uint8_t> bitPacked(byteCount, 0);
            for (size_t pos : data.activePositions) {
                if (pos < bitCount) {
                    bitPacked[pos / 8] |= (1 << (pos % 8));
                }
            }
            std::string bitPackedStr(reinterpret_cast<const char*>(bitPacked.data()), bitPacked.size());

            // Optionally RLE encode
            std::string toWrite;
            if (useRle) {
                auto rle = rle::encodeFromBytes(bitPackedStr);
                toWrite.assign(reinterpret_cast<const char*>(rle.data()), rle.size());
            } else {
                toWrite = bitPackedStr;
            }

            // Write to file or gzip
            bool useGzip = endsWith(outputFile, ".gz");
            std::string rawFile = useGzip ? (outputFile + ".tmp") : outputFile;
            std::ofstream fileStream(rawFile, std::ios::binary | std::ios::trunc);
            if (!fileStream) throw std::runtime_error("Error: Could not open SDR file for writing: " + rawFile);
            fileStream.write(toWrite.data(), toWrite.size());
            fileStream.close();
            if (useGzip) {
                gzFile gzfp = gzopen(outputFile.c_str(), "wb9");
                if (!gzfp) throw std::runtime_error("Error: gzopen failed for " + outputFile);
                std::ifstream in(rawFile, std::ios::binary);
                char buf[8192];
                while (in) {
                    in.read(buf, sizeof(buf));
                    std::streamsize n = in.gcount();
                    if (n > 0) gzwrite(gzfp, buf, static_cast<unsigned int>(n));
                }
                gzclose(gzfp);
                std::remove(rawFile.c_str());
            }

            size_t encodedSize = toWrite.size();
            std::cout << "Encoded SDR in-memory size: " << encodedSize << " bytes" << std::endl;
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
            std::string rawFile = inputFile;
            if (endsWith(inputFile, ".gz")) {
                rawFile = inputFile + ".tmp";
                gzFile gzfp = gzopen(inputFile.c_str(), "rb");
                if (!gzfp) throw std::runtime_error("Error: gzopen failed for " + inputFile);
                std::ofstream out(rawFile, std::ios::binary);
                char buf[8192]; int n;
                while ((n = gzread(gzfp, buf, sizeof(buf))) > 0) {
                    out.write(buf, n);
                }
                gzclose(gzfp);
                out.close();
            }
            std::ifstream fileStream(rawFile, std::ios::binary);
            if (!fileStream) throw std::runtime_error("Error: Could not open SDR file for reading: " + rawFile);
            std::vector<uint8_t> fileData((std::istreambuf_iterator<char>(fileStream)), std::istreambuf_iterator<char>());
            fileStream.close();
            if (endsWith(inputFile, ".gz")) std::remove(rawFile.c_str());
            std::string bitPackedStr;
            if (useRle) {
                // Assume full SDR size (MAX_VECTOR_SIZE bits)
                bitPackedStr = rle::decodeToBytes(fileData, EncodingRanges::MAX_VECTOR_SIZE);
            } else {
                bitPackedStr.assign(reinterpret_cast<const char*>(fileData.data()), fileData.size());
            }
            // Unpack bits to activePositions
            std::vector<size_t> activePositions;
            for (size_t i = 0; i < EncodingRanges::MAX_VECTOR_SIZE; ++i) {
                size_t byteIdx = i / 8;
                size_t bitIdx = i % 8;
                if (byteIdx < bitPackedStr.size() && ((bitPackedStr[byteIdx] >> bitIdx) & 1)) {
                    activePositions.push_back(i);
                }
            }
            SparseDistributedRepresentation::EncodedData data;
            data.activePositions = activePositions;
            data.totalSize = EncodingRanges::MAX_VECTOR_SIZE;
            sdr.setEncoding(data);
            std::string decodedContent = sdr.decode();
            writeFileContent(outputFile, decodedContent);
            std::cout << "Decompression complete. Output written to " << outputFile << std::endl;
        } else {
            std::cerr << "Error: Invalid mode '" << mode << "'." << std::endl; 
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
