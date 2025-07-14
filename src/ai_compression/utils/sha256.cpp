#include "sha256.h"
#include <cstring>
#include <sstream>
#include <iomanip>

namespace CortexAICompression {

namespace {
const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t bsig0(uint32_t x) { return rotr(x,2) ^ rotr(x,13) ^ rotr(x,22); }
inline uint32_t bsig1(uint32_t x) { return rotr(x,6) ^ rotr(x,11) ^ rotr(x,25); }
inline uint32_t ssig0(uint32_t x) { return rotr(x,7) ^ rotr(x,18) ^ (x >> 3); }
inline uint32_t ssig1(uint32_t x) { return rotr(x,17) ^ rotr(x,19) ^ (x >> 10); }
}

SHA256::SHA256() { reset(); }

void SHA256::reset() {
    state[0]=0x6a09e667; state[1]=0xbb67ae85; state[2]=0x3c6ef372; state[3]=0xa54ff53a;
    state[4]=0x510e527f; state[5]=0x9b05688c; state[6]=0x1f83d9ab; state[7]=0x5be0cd19;
    bitlen = 0; datalen = 0;
}

void SHA256::update(const uint8_t* data_, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        data[datalen] = data_[i];
        datalen++;
        if (datalen == 64) {
            transform(data);
            bitlen += 512;
            datalen = 0;
        }
    }
}
void SHA256::update(const std::vector<std::byte>& data_) {
    update(reinterpret_cast<const uint8_t*>(data_.data()), data_.size());
}
void SHA256::update(const std::vector<uint8_t>& data_) {
    update(data_.data(), data_.size());
}
void SHA256::update(const std::string& data_) {
    update(reinterpret_cast<const uint8_t*>(data_.data()), data_.size());
}

std::vector<uint8_t> SHA256::digest() {
    uint8_t hash[32];
    size_t i = datalen;

    // Pad whatever data is left in the buffer.
    if (datalen < 56) {
        data[i++] = 0x80;
        while (i < 56) data[i++] = 0x00;
    } else {
        data[i++] = 0x80;
        while (i < 64) data[i++] = 0x00;
        transform(data);
        std::memset(data, 0, 56);
    }

    // Append to the padding the total message's length in bits and transform.
    bitlen += datalen * 8;
    data[63] = bitlen;
    data[62] = bitlen >> 8;
    data[61] = bitlen >> 16;
    data[60] = bitlen >> 24;
    data[59] = bitlen >> 32;
    data[58] = bitlen >> 40;
    data[57] = bitlen >> 48;
    data[56] = bitlen >> 56;
    transform(data);

    // Since this implementation uses little endian byte ordering and SHA uses big endian,
    // reverse all the bytes when copying the final state to the output hash.
    for (i = 0; i < 8; ++i) {
        hash[i*4+0] = (state[i] >> 24) & 0xff;
        hash[i*4+1] = (state[i] >> 16) & 0xff;
        hash[i*4+2] = (state[i] >> 8) & 0xff;
        hash[i*4+3] = (state[i] >> 0) & 0xff;
    }
    reset();
    return std::vector<uint8_t>(hash, hash+32);
}

std::string SHA256::hexdigest() {
    auto hash = digest();
    std::ostringstream oss;
    for (auto b : hash) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)b;
    }
    return oss.str();
}

void SHA256::transform(const uint8_t* chunk) {
    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) {
        w[i] = (chunk[i*4+0] << 24) | (chunk[i*4+1] << 16) | (chunk[i*4+2] << 8) | (chunk[i*4+3]);
    }
    for (size_t i = 16; i < 64; ++i) {
        w[i] = ssig1(w[i-2]) + w[i-7] + ssig0(w[i-15]) + w[i-16];
    }
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for (size_t i = 0; i < 64; ++i) {
        uint32_t t1 = h + bsig1(e) + ch(e,f,g) + k[i] + w[i];
        uint32_t t2 = bsig0(a) + maj(a,b,c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

} // namespace CortexAICompression 