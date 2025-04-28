#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <bitset>
#include <filesystem>
#include <zlib.h> // For exception handling
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

    // --- Write Vocabulary (preserving original order) ---
    // IMPORTANT: Do NOT sort vocabulary - order must be preserved for correct fingerprint matching
    size_t vocabSize = vocabulary.size();
    fileStream.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));
    
    // Write each word with its full content (no delta encoding)
    for (size_t i = 0; i < vocabSize; ++i) {
        const std::string& word = vocabulary[i];
        size_t wordLen = word.size();
        fileStream.write(reinterpret_cast<const char*>(&wordLen), sizeof(wordLen));
        fileStream.write(word.data(), wordLen);
    }

    // Added debug statements to log vocabulary during serialization.
    std::cout << "Serialized vocabulary (" << vocabSize << " words): \n";
    for (size_t i = 0; i < std::min(vocabSize, size_t(20)); ++i) {
        std::cout << i << ": '" << vocabulary[i] << "' ";
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

    // --- Read Vocabulary (preserving original order) ---
    std::vector<std::string> vocabulary;
    size_t vocabSize = 0;
    fileStream.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
    if (!fileStream) throw std::runtime_error("Error: Failed to read vocabSize from SDR file: " + filePath);
    vocabulary.reserve(vocabSize);
    
    // Read each word with its full content
    for (size_t i = 0; i < vocabSize; ++i) {
        size_t wordLen = 0;
        fileStream.read(reinterpret_cast<char*>(&wordLen), sizeof(wordLen));
        if (!fileStream) throw std::runtime_error("Error: Failed to read wordLen for vocab word " + std::to_string(i) + " from SDR file: " + filePath);
        
        std::string word(wordLen, '\0');
        if (wordLen > 0) {
            fileStream.read(&word[0], wordLen);
            if (!fileStream) throw std::runtime_error("Error: Failed to read vocab word " + std::to_string(i) + " from SDR file: " + filePath);
        }
        vocabulary.push_back(word);
    }

    // Set the vocabulary in the SDR object (this will also rebuild the internal map)
    sdr.setWordVocabulary(vocabulary);

    // Debug: Log the restored vocabulary
    std::cout << "Restored vocabulary (" << vocabSize << " words): \n";
    for (size_t i = 0; i < std::min(vocabSize, size_t(20)); ++i) {
        std::cout << i << ": '" << vocabulary[i] << "' ";
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

// Function to detect file type based on extension
enum class FileType {
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
    BINARY,
    UNKNOWN
};

FileType detectFileType(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // Image formats
    if (ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "gif" || ext == "bmp" || ext == "tiff") {
        return FileType::IMAGE;
    }
    // Audio formats
    else if (ext == "mp3" || ext == "wav" || ext == "ogg" || ext == "flac" || ext == "aac") {
        return FileType::AUDIO;
    }
    // Video formats
    else if (ext == "mp4" || ext == "avi" || ext == "mkv" || ext == "mov" || ext == "webm") {
        return FileType::VIDEO;
    }
    // Text formats
    else if (ext == "txt" || ext == "md" || ext == "html" || ext == "xml" || ext == "json" || 
             ext == "csv" || ext == "cpp" || ext == "hpp" || ext == "c" || ext == "h" || 
             ext == "py" || ext == "js" || ext == "css" || ext == "java") {
        return FileType::TEXT;
    }
    // Binary formats (not specifically handled yet)
    else if (ext == "bin" || ext == "dat" || ext == "exe" || ext == "dll" || ext == "so") {
        return FileType::BINARY;
    }
    // Unknown
    else {
        return FileType::UNKNOWN;
    }
}

// Function to directly compress an image file
bool compressImageFile(const std::string& inputFile, const std::string& outputFile, double sparsity = 0.002) {
    std::cout << "Using SDR-based image compression for file: " << inputFile << std::endl;
    std::cout << "Applying sparsity: " << (sparsity * 100) << "% (higher = more lossy compression)" << std::endl;
    
    // Read the input image file
    std::ifstream inFile(inputFile, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Could not open input file: " << inputFile << std::endl;
        return false;
    }
    
    // Read the file content into a buffer
    std::vector<unsigned char> imageData((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();
    
    // Create a compressed output file
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not create output file: " << outputFile << std::endl;
        return false;
    }
    
    // Write a magic number to identify this as a compressed image file
    const uint32_t magicNumber = 0x494D4753; // "IMGS" in ASCII
    outFile.write(reinterpret_cast<const char*>(&magicNumber), sizeof(magicNumber));
    
    // Write the original file size
    const uint32_t fileSize = static_cast<uint32_t>(imageData.size());
    outFile.write(reinterpret_cast<const char*>(&fileSize), sizeof(fileSize));
    
    // Store the first 32 bytes of the file (PNG header) uncompressed
    // This will help maintain the file format for proper identification
    uint32_t headerSize = std::min(static_cast<uint32_t>(32), fileSize);
    outFile.write(reinterpret_cast<const char*>(&headerSize), sizeof(headerSize));
    outFile.write(reinterpret_cast<const char*>(imageData.data()), headerSize);
    
    // Write the sparsity value for decompression
    outFile.write(reinterpret_cast<const char*>(&sparsity), sizeof(sparsity));
    
    // SDR-based compression: convert image to sparse distributed representation
    // This approach is designed to work well with RLE and gzip
    
    // Step 1: Create a sparse representation of the image
    // We'll only keep the most significant pixels based on sparsity
    std::vector<std::pair<uint32_t, unsigned char>> significantPixels;
    
    // Calculate how many pixels to keep based on sparsity
    size_t totalPixels = imageData.size() - headerSize;
    size_t keepCount = std::max(size_t(1), static_cast<size_t>(totalPixels * sparsity));
    
    // Find the most significant pixels (those with highest values)
    std::vector<std::pair<unsigned char, uint32_t>> pixelValues;
    for (size_t i = headerSize; i < imageData.size(); ++i) {
        pixelValues.push_back({imageData[i], static_cast<uint32_t>(i)});
    }
    
    // Sort by value (descending)
    std::sort(pixelValues.begin(), pixelValues.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Keep only the most significant pixels
    for (size_t i = 0; i < keepCount && i < pixelValues.size(); ++i) {
        significantPixels.push_back({pixelValues[i].second, pixelValues[i].first});
    }
    
    // Sort by position for better compression
    std::sort(significantPixels.begin(), significantPixels.end(), 
             [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Write the number of significant pixels
    uint32_t numPixels = static_cast<uint32_t>(significantPixels.size());
    outFile.write(reinterpret_cast<const char*>(&numPixels), sizeof(numPixels));
    
    // Step 2: Create a simple RLE-friendly format
    // Format: [position][value][position][value]...
    std::vector<unsigned char> compressed;
    
    for (const auto& pixel : significantPixels) {
        // Store position (4 bytes)
        uint32_t pos = pixel.first;
        compressed.push_back((pos >> 0) & 0xFF);
        compressed.push_back((pos >> 8) & 0xFF);
        compressed.push_back((pos >> 16) & 0xFF);
        compressed.push_back((pos >> 24) & 0xFF);
        
        // Store value (1 byte)
        compressed.push_back(pixel.second);
    }
    
    // Write the compressed data
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    
    return true;
}

// Function to directly decompress an image file
bool decompressImageFile(const std::string& inputFile, const std::string& outputFile) {
    std::cout << "Using SDR-based image decompression for file: " << inputFile << std::endl;
    
    // Open the compressed file
    std::ifstream inFile(inputFile, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Could not open input file: " << inputFile << std::endl;
        return false;
    }
    
    // Read and check the magic number
    uint32_t magicNumber = 0;
    inFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    if (magicNumber != 0x494D4753) {
        std::cerr << "Error: Not a valid compressed image file" << std::endl;
        return false;
    }
    
    // Read the original file size
    uint32_t originalFileSize = 0;
    inFile.read(reinterpret_cast<char*>(&originalFileSize), sizeof(originalFileSize));
    
    // Read the header size
    uint32_t headerSize = 0;
    inFile.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));
    
    // Read the uncompressed header
    std::vector<unsigned char> header(headerSize);
    inFile.read(reinterpret_cast<char*>(header.data()), headerSize);
    
    // Read the sparsity value
    double sparsity = 0.0;
    inFile.read(reinterpret_cast<char*>(&sparsity), sizeof(sparsity));
    std::cout << "Using sparsity: " << (sparsity * 100) << "%" << std::endl;
    
    // Read the number of significant pixels
    uint32_t numPixels = 0;
    inFile.read(reinterpret_cast<char*>(&numPixels), sizeof(numPixels));
    
    // Initialize decompressed data with zeros
    std::vector<unsigned char> decompressed(originalFileSize, 0);
    
    // Copy the header
    std::copy(header.begin(), header.end(), decompressed.begin());
    
    // Read the compressed data
    std::vector<unsigned char> compressed((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();
    
    // Process the compressed data
    size_t compressedIndex = 0;
    for (uint32_t i = 0; i < numPixels && compressedIndex + 4 < compressed.size(); ++i) {
        // Read position (4 bytes)
        uint32_t position = 
            (compressed[compressedIndex]) |
            (compressed[compressedIndex + 1] << 8) |
            (compressed[compressedIndex + 2] << 16) |
            (compressed[compressedIndex + 3] << 24);
        compressedIndex += 4;
        
        // Read value (1 byte)
        unsigned char value = compressed[compressedIndex++];
        
        // Store value if position is valid
        if (position < decompressed.size()) {
            decompressed[position] = value;
        }
    }
    
    // Write the decompressed data to the output file
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not create output file: " << outputFile << std::endl;
        return false;
    }
    
    outFile.write(reinterpret_cast<const char*>(decompressed.data()), decompressed.size());
    
    return true;
}

// Function to check if a file is a compressed image
bool isCompressedImageFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        return false;
    }
    
    uint32_t magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    
    return magicNumber == 0x494D4753; // "IMGS" in ASCII
}

// Function to encode content based on file type
SparseDistributedRepresentation::EncodedData encodeContent(
    SparseDistributedRepresentation& sdr, 
    const std::string& content, 
    FileType fileType,
    const std::string& filePath) {
    
    switch (fileType) {
        case FileType::TEXT:
            std::cout << "Using text encoder for file: " << filePath << std::endl;
            return sdr.encodeText(content);
        case FileType::IMAGE:
            std::cout << "Using image encoder for file: " << filePath << std::endl;
            // Use the dedicated image encoder
            return sdr.encodeImage(filePath);
        case FileType::AUDIO:
            std::cout << "Using audio encoder for file: " << filePath << std::endl;
            // For now, fall back to text encoding for audio
            return sdr.encodeText(content);
        case FileType::VIDEO:
            std::cout << "Using video encoder for file: " << filePath << std::endl;
            // For now, fall back to text encoding for video
            return sdr.encodeText(content);
        case FileType::BINARY:
        case FileType::UNKNOWN:
        default:
            std::cout << "Using generic binary encoder for file: " << filePath << std::endl;
            // For now, fall back to text encoding for unknown types
            return sdr.encodeText(content);
    }
}

// Function to decode content based on file type
std::string decodeContent(
    SparseDistributedRepresentation& sdr, 
    FileType fileType,
    const std::string& outputPath) {
    
    // Check if this is image data using the SDR's method
    if (sdr.isImageData()) {
        std::cout << "Detected image data, using image decoder for output: " << outputPath << std::endl;
        return sdr.decodeImage();
    } else {
        // For all other file types, use the appropriate decoder based on file type
        switch (fileType) {
            case FileType::TEXT:
                std::cout << "Using text decoder for output: " << outputPath << std::endl;
                return sdr.decode();
            case FileType::IMAGE:
                std::cout << "Using image decoder for output: " << outputPath << std::endl;
                // This should not happen if isImageData() is working correctly
                // but we'll handle it anyway
                return sdr.decode();
            case FileType::AUDIO:
                std::cout << "Using audio decoder for output: " << outputPath << std::endl;
                return sdr.decode();
            case FileType::VIDEO:
                std::cout << "Using video decoder for output: " << outputPath << std::endl;
                return sdr.decode();
            case FileType::BINARY:
            case FileType::UNKNOWN:
            default:
                std::cout << "Using generic decoder for output: " << outputPath << std::endl;
                return sdr.decode();
        }
    }
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
            
            // Detect file type and use appropriate encoder
            FileType inputFileType = detectFileType(inputFile);
            
            // Store the original file size for compression ratio calculation
            size_t originalSize = 0;
            
            if (inputFileType == FileType::IMAGE) {
                // For images, use direct image compression
                originalSize = std::filesystem::file_size(inputFile);
                std::cout << "Original size: " << originalSize << " bytes" << std::endl;
                
                if (compressImageFile(inputFile, outputFile, sparsity)) {
                    // Get the compressed file size
                    size_t compressedSize = std::filesystem::file_size(outputFile);
                    std::cout << "Output file size: " << compressedSize << " bytes" << std::endl;
                    
                    if (compressedSize > 0) {
                        double compressionRatio = static_cast<double>(originalSize) / compressedSize;
                        std::cout << "Compression ratio (file size): " << compressionRatio << ":1" << std::endl;
                    } else {
                        std::cout << "Compression ratio: N/A (output file size is zero)" << std::endl;
                    }
                    
                    std::cout << "Compression complete." << std::endl;
                    return 0; // Exit early, we've handled the image directly
                } else {
                    std::cerr << "Failed to compress image file." << std::endl;
                    return 1;
                }
            } else {
                // For text and other files, read the content and encode it
                std::string fileContent = readFileContent(inputFile);
                originalSize = fileContent.size();
                std::cout << "Original size: " << originalSize << " bytes" << std::endl;
                encodeContent(sdr, fileContent, inputFileType, inputFile);
            }

            // Serialize SDR (bit-packed + vocabulary)
            bool useGzip = endsWith(outputFile, ".gz");
            std::string rawFile = useGzip ? (outputFile + ".tmp") : outputFile;
            writeSdrFile(rawFile, sdr);
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
            std::ifstream sdrFile(useGzip ? outputFile : rawFile, std::ios::binary | std::ios::ate);
            size_t sdrFileSize = sdrFile.tellg();
            std::cout << "Output SDR file size: " << sdrFileSize << " bytes" << std::endl;
            if (sdrFileSize > 0) {
                double compressionRatio = static_cast<double>(originalSize) / sdrFileSize;
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
            // Check if this is a compressed image file
            if (isCompressedImageFile(rawFile)) {
                // For compressed images, use direct image decompression
                if (decompressImageFile(rawFile, outputFile)) {
                    if (endsWith(inputFile, ".gz")) std::remove(rawFile.c_str());
                    std::cout << "Decompression complete. Output written to " << outputFile << std::endl;
                    return 0; // Exit early, we've handled the image directly
                } else {
                    if (endsWith(inputFile, ".gz")) std::remove(rawFile.c_str());
                    std::cerr << "Failed to decompress image file." << std::endl;
                    return 1;
                }
            } else {
                // For other files, use the standard SDR decompression
                readSdrFile(rawFile, sdr);
                if (endsWith(inputFile, ".gz")) std::remove(rawFile.c_str());
                
                // Detect file type and use appropriate decoder
                FileType outputFileType = detectFileType(outputFile);
                std::string decodedContent = decodeContent(sdr, outputFileType, outputFile);
                
                writeFileContent(outputFile, decodedContent);
                std::cout << "Decompression complete. Output written to " << outputFile << std::endl;
            }
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
