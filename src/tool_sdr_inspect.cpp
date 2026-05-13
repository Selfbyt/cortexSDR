/**
 * @file tool_sdr_inspect.cpp
 * @brief List per-segment compressed_size + strategy_id from a .sdr archive
 * so we can see exactly where the bytes are going.
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr char kMagic[8] = {'C', 'O', 'R', 'T', 'E', 'X', 'S', 'R'};

template <typename T>
T readPOD(std::ifstream& in) {
    T v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}

std::string readString(std::ifstream& in) {
    auto len = readPOD<uint32_t>(in);
    std::string s(len, '\0');
    in.read(s.data(), len);
    return s;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <archive.sdr>\n"; return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::cerr << "cannot open\n"; return 1; }

    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) {
        std::cerr << "bad magic\n";
        return 1;
    }
    auto version = readPOD<uint32_t>(in);
    auto numSegs = readPOD<uint64_t>(in);
    auto indexOffset = readPOD<uint64_t>(in);
    std::cout << "version=" << version << " segments=" << numSegs
              << " index_offset=" << indexOffset << "\n";

    in.seekg(static_cast<std::streamoff>(indexOffset), std::ios::beg);
    struct Row { std::string name; uint8_t strat; uint64_t orig; uint64_t comp; };
    std::vector<Row> rows;
    rows.reserve(numSegs);
    for (uint64_t i = 0; i < numSegs; ++i) {
        Row r;
        r.name = readString(in);
        auto fmt   = readString(in);
        auto lt    = readString(in);
        auto orig_type = readPOD<uint8_t>(in);
        r.strat = readPOD<uint8_t>(in);
        r.orig = readPOD<uint64_t>(in);
        r.comp = readPOD<uint64_t>(in);
        auto data_off = readPOD<uint64_t>(in);
        auto ln    = readString(in);
        auto li    = readPOD<uint32_t>(in);
        (void)fmt; (void)lt; (void)orig_type; (void)data_off; (void)ln; (void)li;

        const bool hasMeta = readPOD<uint8_t>(in) != 0;
        if (hasMeta) {
            auto nd = readPOD<uint8_t>(in);
            for (uint8_t j = 0; j < nd; ++j) readPOD<uint32_t>(in);
            readPOD<float>(in);   // sparsity
            readPOD<uint8_t>(in); // sorted
            if (readPOD<uint8_t>(in) != 0) readPOD<float>(in);  // scale
            if (readPOD<uint8_t>(in) != 0) readPOD<float>(in);  // zero_point
        }
        for (int s = 0; s < 2; ++s) {
            if (readPOD<uint8_t>(in) != 0) {
                auto n = readPOD<uint8_t>(in);
                for (uint8_t j = 0; j < n; ++j) readPOD<uint32_t>(in);
            }
        }
        rows.push_back(std::move(r));
    }

    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b){ return a.comp > b.comp; });

    uint64_t total = 0;
    std::cout << "  top 25 segments by compressed size:\n";
    std::cout << "  strat_id  comp_size      orig_size     name\n";
    for (size_t i = 0; i < std::min<size_t>(25, rows.size()); ++i) {
        std::printf("    %2u      %12llu  %12llu   %s\n",
            static_cast<unsigned>(rows[i].strat),
            static_cast<unsigned long long>(rows[i].comp),
            static_cast<unsigned long long>(rows[i].orig),
            rows[i].name.c_str());
    }
    for (auto& r : rows) total += r.comp;
    std::cout << "\n  sum of compressed_size across " << rows.size()
              << " segments = " << total << " bytes\n";
    return 0;
}
