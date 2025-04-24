# Cortex SDR (Sparse Distributed Representation)

A C++ firmware implementation that enables storage devices or computers to store data at a 5:1 ratio (5 MB in 1 MB space) while also providing powerful encoding and decoding capabilities for AI applications.

## Overview

Cortex SDR is a firmware that implements a Sparse Distributed Representation system. It encodes various data types into binary vectors with minimal active bits. This approach enables:
- High-density data storage (achieving 5:1 compression)
- Efficient encoding/decoding operations for AI and machine learning
- Noise-resilient data representations
- Advanced pattern matching and similarity detection

## Features

### Core Functionality
- 2000-bit vector representation
- Multiple data type support (text, numbers, dates, special characters)
- Memory-efficient sparse storage
- Pattern matching and similarity detection
- Flexible encoding/decoding system

### Supported Data Types
- Text and words
- Numbers (range: -1000 to 1000)
- Special characters and symbols
- Date and time data

## Technical Details

### Vector Structure
- Fixed size: 2000 bits
- Region allocation:
  - 0-999: Word encodings
  - 1000-1499: Special characters
  - 1500-1999: Numbers

### Performance
- O(1) encoding/decoding for individual elements
- O(n) for text strings where n is string length
- Space complexity: O(k) where k is number of active bits
- Typical compression ratios of 5:1 attainable

## Usage

### Basic Example

```cpp
// Initialize SDR with vocabulary
SparseDistributedRepresentation sdr{
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"
};

// Encode text
std::string text = "The quick brown fox";
auto encoded = sdr.encodeText(text);

// Decode back to text
std::string decoded = sdr.decode();
```

### Number Encoding

```cpp
// Encode a number
double number = 42.5;
auto encoded = sdr.encodeNumber(number);

// Decode the number
std::string decodedNumber = sdr.decode();
```

## Installation

### Prerequisites
- C++11 or higher
- CMake 3.0 or higher

### Building from Source
1. Clone the repository
```bash
git clone https://github.com/Selfbyt/cortexSDR.gitt
cd cortex-sdr
```

2. Build the project
```bash
mkdir build
cd build
cmake ..
make
```

## Applications

### 1. Data Storage & Firmware Compression
- Store up to 5 MB of data in only 1 MB of physical space
- Improve storage density in flash memory or other storage devices

### 2. Artificial Intelligence
- Use as a preprocessor for ML models with efficient feature encoding
- Enable noise-robust pattern recognition and similarity detection

### 3. Natural Language Processing
- Text classification
- Semantic similarity detection
- Pattern matching in text
- Efficient storage of large text corpora

### 4. Time Series Data
- Temporal pattern recognition
- Anomaly detection
- Event sequence matching
- Time-based predictions

### 5. Machine Learning
- Feature encoding for ML models
- Dimensionality reduction
- Pattern recognition
- Noise-resistant data representation

### 6. Data Compression
- Efficient storage of mixed data types
- Lossy compression with semantic preservation
- Pattern-based data deduplication

## Future Enhancements

1. **Distributed Computing Support**
   - Parallel encoding/decoding
   - Distributed pattern matching

2. **Advanced Pattern Recognition**
   - Hierarchical pattern matching
   - Temporal sequence learning
   - Anomaly detection

3. **Additional Encoders**
   - Image encoding
   - Audio signal encoding
   - Geographic coordinate encoding

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Reference to LICENSE file:

```1:21:LICENSE
MIT License

Copyright (c) 2024 selfbyt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## Authors

- selfbyt

## Acknowledgments

- Inspired by research in sparse distributed representations and neural encoding
- Built on principles of efficient data representation and pattern matching

For more detailed implementation examples, see the test file:

```1:60:src/test.cpp
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
```
