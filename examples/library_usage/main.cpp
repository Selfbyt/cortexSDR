#include "cortexSDR/cortexSDR.hpp"
#include <iostream>
#include <string>
#include <vector>

/**
 * Example application demonstrating how to use cortexSDR as a library
 */
int main() {
    std::cout << "CortexSDR Library Example" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Initialize SDR with vocabulary
    SparseDistributedRepresentation sdr{
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"
    };
    
    // Example 1: Text encoding and decoding
    std::cout << "\nExample 1: Text Encoding" << std::endl;
    std::string text = "the quick brown fox";
    auto encoded = sdr.encodeText(text);
    
    std::cout << "Original text: " << text << std::endl;
    std::cout << "Encoded size: " << encoded.size() << " bits" << std::endl;
    
    // Decode back to text
    std::string decoded = sdr.decode();
    std::cout << "Decoded text: " << decoded << std::endl;
    
    // Example 2: Number encoding and decoding
    std::cout << "\nExample 2: Number Encoding" << std::endl;
    double number = 42.5;
    encoded = sdr.encodeNumber(number);
    
    std::cout << "Original number: " << number << std::endl;
    std::cout << "Encoded size: " << encoded.size() << " bits" << std::endl;
    
    // Decode the number
    decoded = sdr.decode();
    std::cout << "Decoded value: " << decoded << std::endl;
    
    // Example 3: Compression ratio demonstration
    std::cout << "\nExample 3: Compression Ratio" << std::endl;
    std::string longText = "This is a longer text that demonstrates the compression capabilities of the cortexSDR library. "
                          "The sparse distributed representation allows for efficient storage of data with a typical "
                          "compression ratio of 5:1 as mentioned in the documentation.";
    
    // Calculate original size in bytes (1 char = 1 byte in ASCII/UTF-8 for English text)
    size_t originalSize = longText.size();
    
    // Encode the text
    encoded = sdr.encodeText(longText);
    
    // Calculate encoded size in bytes (bits / 8)
    size_t encodedSize = encoded.size() / 8;
    
    std::cout << "Original size: " << originalSize << " bytes" << std::endl;
    std::cout << "Encoded size: " << encodedSize << " bytes" << std::endl;
    std::cout << "Compression ratio: " << static_cast<float>(originalSize) / encodedSize << ":1" << std::endl;
    
    return 0;
}
