#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <sdr_file>" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << argv[1] << std::endl;
        return 1;
    }

    // Read file header
    std::vector<char> buffer(1024);
    file.read(buffer.data(), buffer.size());
    size_t bytesRead = file.gcount();

    std::cout << "First " << bytesRead << " bytes of file:" << std::endl;
    for (size_t i = 0; i < bytesRead && i < 256; ++i) {
        if (i % 16 == 0) {
            std::cout << std::endl << std::setw(4) << std::setfill('0') << i << ": ";
        }
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (static_cast<unsigned int>(static_cast<unsigned char>(buffer[i]))) << " ";
    }
    std::cout << std::dec << std::endl;

    // Search for format flags
    std::cout << "\nSearching for format flags (0x95, 0x96, 0x88, 0xD0, 0x90, 0x0F)..." << std::endl;
    file.seekg(0);
    std::vector<char> fullBuffer(1024 * 1024); // Read first 1MB
    file.read(fullBuffer.data(), fullBuffer.size());
    bytesRead = file.gcount();

    int count_95 = 0, count_96 = 0, count_88 = 0, count_D0 = 0, count_90 = 0, count_0F = 0;
    for (size_t i = 0; i < bytesRead; ++i) {
        unsigned char byte = static_cast<unsigned char>(fullBuffer[i]);
        if (byte == 0x95) count_95++;
        if (byte == 0x96) count_96++;
        if (byte == 0x88) count_88++;
        if (byte == 0xD0) count_D0++;
        if (byte == 0x90) count_90++;
        if (byte == 0x0F) count_0F++;
    }

    std::cout << "Format flag counts in first 1MB:" << std::endl;
    std::cout << "  0x95 (weight preservation): " << count_95 << std::endl;
    std::cout << "  0x96 (non-weight with values): " << count_96 << std::endl;
    std::cout << "  0x88 (large tensor): " << count_88 << std::endl;
    std::cout << "  0xD0 (medium tensor): " << count_D0 << std::endl;
    std::cout << "  0x90 (very large tensor): " << count_90 << std::endl;
    std::cout << "  0x0F (small tensor/bias): " << count_0F << std::endl;

    return 0;
}
