#include <iostream>
#include "cortexSDR.cpp"

void testMixedContent() {
    // Initialize SDR with vocabulary
    SparseDistributedRepresentation sdr{
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"
    };
    
    // Test mixed content encoding/decoding
    std::string original = "The quick brown fox jumps over 42.5!";
    auto encoded = sdr.encodeText(original);
    std::string decoded = sdr.decode();
    
    std::cout << "\n=== Mixed Content Encoding Test ===\n";
    std::cout << "Original: " << original << "\n";
    std::cout << "Decoded:  " << decoded << "\n\n";
    
    // Print detailed statistics
    sdr.printStats();
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
    
    // Print encoded vector details
    std::cout << "Active positions: ";
    for (const auto& pos : encoded.activePositions) {
        std::cout << pos << " ";
    }
    std::cout << "\nTotal vector size: " << encoded.totalSize << "\n";
    
    // Calculate and print encoded size
    size_t encodedSize = encoded.getMemorySize();
    std::cout << "Encoded SDR size: " << encodedSize << " bytes\n";
    std::cout << "Compression ratio: " << static_cast<float>(paragraph.size()) / encodedSize << ":1\n\n";
    
    // Print SDR stats
    std::cout << "SDR Statistics:\n";
    sdr.printStats();

    std::cout << "\nTesting mixed content encoding:\n";
    testMixedContent();

    return 0;
}
