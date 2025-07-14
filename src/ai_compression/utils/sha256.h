#ifndef AI_COMPRESSION_UTILS_SHA256_H
#define AI_COMPRESSION_UTILS_SHA256_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>

namespace CortexAICompression {

class SHA256 {
public:
    SHA256();
    void update(const uint8_t* data, size_t len);
    void update(const std::vector<std::byte>& data);
    void update(const std::vector<uint8_t>& data);
    void update(const std::string& data);
    std::vector<uint8_t> digest(); // 32 bytes
    std::string hexdigest(); // 64 hex chars
    void reset();
private:
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t data[64];
    size_t datalen;
    void transform(const uint8_t* chunk);
};

} // namespace CortexAICompression

#endif // AI_COMPRESSION_UTILS_SHA256_H 