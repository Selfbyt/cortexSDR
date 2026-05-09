/**
 * @file analyze_sdr_size.cpp
 * @brief Analyze .sdr file to show size breakdown and identify overhead
 * 
 * This tool helps diagnose why .sdr files are larger than expected by showing:
 * - Archive header size
 * - Index table size
 * - Per-segment breakdown (header + payload)
 * - Uncompressed vs compressed segments
 * - Total overhead
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <algorithm>

struct SegmentInfo {
    std::string name;
    uint64_t header_size;
    uint64_t payload_size;
    uint64_t original_size;
    uint8_t compression_strategy;
    bool is_compressed;
};

// Read basic types from stream
template<typename T>
bool readBasicType(std::ifstream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return stream.good();
}

// Read length-prefixed string
bool readString(std::ifstream& stream, std::string& str) {
    uint32_t len;
    if (!readBasicType(stream, len)) return false;
    if (len > 1000000) return false; // Sanity check
    
    str.resize(len);
    stream.read(&str[0], len);
    return stream.good();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file.sdr>\n";
        return 1;
    }

    std::string sdr_path = argv[1];
    std::ifstream file(sdr_path, std::ios::binary);
    
    if (!file) {
        std::cerr << "Error: Cannot open file: " << sdr_path << "\n";
        return 1;
    }

    // Get total file size
    file.seekg(0, std::ios::end);
    uint64_t total_file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::cout << "===========================================\n";
    std::cout << "SDR File Size Analysis\n";
    std::cout << "===========================================\n";
    std::cout << "File: " << sdr_path << "\n";
    std::cout << "Total file size: " << total_file_size << " bytes ("
              << (total_file_size / 1024.0 / 1024.0) << " MB)\n\n";

    // Read archive header
    char magic[8];
    file.read(magic, 8);
    if (std::memcmp(magic, "CORTEXSR", 8) != 0) {
        std::cerr << "Error: Invalid magic number\n";
        std::cerr << "Expected: CORTEXSR, Got: ";
        for (int i = 0; i < 8; i++) {
            std::cerr << magic[i];
        }
        std::cerr << "\n";
        return 1;
    }

    uint32_t version;
    uint64_t num_segments;
    uint64_t index_offset;
    
    readBasicType(file, version);
    readBasicType(file, num_segments);
    readBasicType(file, index_offset);

    uint64_t header_size = file.tellg();
    
    std::cout << "Archive Header:\n";
    std::cout << "  Size: " << header_size << " bytes\n";
    std::cout << "  Version: " << version << "\n";
    std::cout << "  Number of segments: " << num_segments << "\n";
    std::cout << "  Index offset: " << index_offset << "\n\n";

    // Jump to index table
    file.seekg(index_offset, std::ios::beg);
    
    std::vector<SegmentInfo> segments;
    uint64_t total_payload_size = 0;
    uint64_t total_original_size = 0;
    uint64_t compressed_payload_size = 0;
    uint64_t uncompressed_payload_size = 0;
    
    // Read index table
    for (uint64_t i = 0; i < num_segments; ++i) {
        SegmentInfo info;
        
        // Read segment header from index
        std::string data_format, layer_type, layer_name;
        uint8_t original_type;
        uint64_t compressed_size, data_offset;
        uint32_t layer_index;
        
        if (!readString(file, info.name)) break;
        if (!readString(file, data_format)) break;
        if (!readString(file, layer_type)) break;
        if (!readBasicType(file, original_type)) break;
        if (!readBasicType(file, info.compression_strategy)) break;
        if (!readBasicType(file, info.original_size)) break;
        if (!readBasicType(file, compressed_size)) break;
        if (!readBasicType(file, data_offset)) break;
        if (!readString(file, layer_name)) break;
        if (!readBasicType(file, layer_index)) break;
        
        // Skip tensor metadata
        bool hasMetadata;
        if (!readBasicType(file, hasMetadata)) break;
        if (hasMetadata) {
            uint8_t numDims;
            readBasicType(file, numDims);
            for (uint8_t d = 0; d < numDims; ++d) {
                uint32_t dim;
                readBasicType(file, dim);
            }
            float sparsity;
            bool is_sorted;
            readBasicType(file, sparsity);
            readBasicType(file, is_sorted);
            
            bool hasScale, hasZeroPoint;
            readBasicType(file, hasScale);
            if (hasScale) {
                float scale;
                readBasicType(file, scale);
            }
            readBasicType(file, hasZeroPoint);
            if (hasZeroPoint) {
                int32_t zp;
                readBasicType(file, zp);
            }
        }
        
        // Skip input/output shapes
        uint8_t hasInputShape, hasOutputShape;
        readBasicType(file, hasInputShape);
        if (hasInputShape) {
            uint8_t numIn;
            readBasicType(file, numIn);
            for (uint8_t d = 0; d < numIn; ++d) {
                uint32_t dim;
                readBasicType(file, dim);
            }
        }
        readBasicType(file, hasOutputShape);
        if (hasOutputShape) {
            uint8_t numOut;
            readBasicType(file, numOut);
            for (uint8_t d = 0; d < numOut; ++d) {
                uint32_t dim;
                readBasicType(file, dim);
            }
        }
        
        info.payload_size = compressed_size;
        info.is_compressed = (info.compression_strategy != 0);
        
        // Estimate header size (name + metadata in index)
        info.header_size = info.name.length() + data_format.length() + 
                          layer_type.length() + layer_name.length() + 100; // Approximate
        
        segments.push_back(info);
        total_payload_size += info.payload_size;
        total_original_size += info.original_size;
        
        if (info.is_compressed) {
            compressed_payload_size += info.payload_size;
        } else {
            uncompressed_payload_size += info.payload_size;
        }
    }
    
    uint64_t index_size = total_file_size - index_offset;
    uint64_t payload_area_size = index_offset - header_size;
    uint64_t total_overhead = total_file_size - total_payload_size;
    
    std::cout << "Size Breakdown:\n";
    std::cout << "  Archive header: " << header_size << " bytes\n";
    std::cout << "  Payload area: " << payload_area_size << " bytes\n";
    std::cout << "  Index table: " << index_size << " bytes\n";
    std::cout << "  Total overhead: " << total_overhead << " bytes ("
              << (total_overhead * 100.0 / total_file_size) << "%)\n\n";
    
    std::cout << "Payload Analysis:\n";
    std::cout << "  Total payload: " << total_payload_size << " bytes\n";
    std::cout << "  Compressed payloads: " << compressed_payload_size << " bytes\n";
    std::cout << "  Uncompressed payloads: " << uncompressed_payload_size << " bytes\n";
    std::cout << "  Original total size: " << total_original_size << " bytes\n";
    std::cout << "  Payload compression ratio: " 
              << (total_original_size / (double)total_payload_size) << ":1\n\n";
    
    std::cout << "Actual File Compression:\n";
    std::cout << "  File size: " << total_file_size << " bytes\n";
    std::cout << "  Original size: " << total_original_size << " bytes\n";
    std::cout << "  Actual compression ratio: " 
              << (total_original_size / (double)total_file_size) << ":1\n\n";
    
    // Show top 10 largest segments
    std::sort(segments.begin(), segments.end(), 
              [](const SegmentInfo& a, const SegmentInfo& b) {
                  return a.payload_size > b.payload_size;
              });
    
    std::cout << "Top 10 Largest Segments:\n";
    std::cout << std::left << std::setw(40) << "Name" 
              << std::right << std::setw(15) << "Payload (KB)"
              << std::setw(15) << "Original (KB)"
              << std::setw(10) << "Strategy"
              << std::setw(10) << "Ratio" << "\n";
    std::cout << std::string(90, '-') << "\n";
    
    for (size_t i = 0; i < std::min(size_t(10), segments.size()); ++i) {
        const auto& seg = segments[i];
        double ratio = seg.original_size / (double)seg.payload_size;
        
        std::cout << std::left << std::setw(40) << seg.name.substr(0, 39)
                  << std::right << std::setw(15) << (seg.payload_size / 1024.0)
                  << std::setw(15) << (seg.original_size / 1024.0)
                  << std::setw(10) << (int)seg.compression_strategy
                  << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "\n";
    }
    
    std::cout << "\n";
    
    // Show uncompressed segments
    uint64_t uncompressed_count = 0;
    uint64_t uncompressed_total = 0;
    for (const auto& seg : segments) {
        if (!seg.is_compressed) {
            uncompressed_count++;
            uncompressed_total += seg.payload_size;
        }
    }
    
    if (uncompressed_count > 0) {
        std::cout << "Uncompressed Segments: " << uncompressed_count << " segments, "
                  << (uncompressed_total / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  (These are stored without compression, adding to file size)\n\n";
    }
    
    std::cout << "===========================================\n";
    std::cout << "Summary:\n";
    std::cout << "===========================================\n";
    std::cout << "The difference between reported compression ratio\n";
    std::cout << "and actual file size is due to:\n";
    std::cout << "  1. Archive overhead: " << (header_size + index_size) << " bytes\n";
    std::cout << "  2. Uncompressed segments: " << uncompressed_total << " bytes\n";
    std::cout << "  3. Per-segment headers: ~" << (segments.size() * 50) << " bytes\n";
    std::cout << "  Total overhead: " << total_overhead << " bytes ("
              << (total_overhead * 100.0 / total_file_size) << "%)\n";
    
    return 0;
}
