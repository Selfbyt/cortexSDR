#include <iostream>
#include "cortexSDR.cpp"

void testEncodeDecodeFlow() {
    SparseDistributedRepresentation sdr{
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"
    };
    
    // Test text encoding/decoding
    std::string original = "The quick brown fox";
    auto encoded = sdr.encodeText(original);
    std::string decoded = sdr.decode();
    
    std::cout << "Original: " << original << "\n";
    std::cout << "Decoded:  " << decoded << "\n";
    
    // Test number encoding/decoding
    double number = 42.5;
    encoded = sdr.encodeNumber(number);
    decoded = sdr.decode();
    
    std::cout << "\nOriginal number: " << number << "\n";
    std::cout << "Decoded number:  " << decoded << "\n";
}

int main() {
    // Initialize SDR with a vocabulary
    SparseDistributedRepresentation sdr{
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", 
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt"
    };

    // Original text paragraph
    std::string paragraph = "The quick brown fox jumps over the lazy dog. "
                          "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
                          "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
    
    // Print original text size
    std::cout << "Original text size: " << paragraph.size() << " bytes\n\n";

    // Encode the paragraph
    std::cout << "Encoding paragraph to SDR...\n";
    auto encoded = sdr.encodeText(paragraph);
    
    // Calculate encoded size using the EncodedData structure
    size_t encodedSize = encoded.getMemorySize();
    
    std::cout << "Encoded SDR size: " << encodedSize << " bytes\n";
    std::cout << "Compression ratio: " << static_cast<float>(paragraph.size()) / encodedSize << ":1\n\n";
    
    // Print SDR stats
    std::cout << "SDR Statistics:\n";
    sdr.printStats();

    std::cout << "\nTesting encode/decode flow:\n";
    testEncodeDecodeFlow();

    return 0;
}
