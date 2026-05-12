/**
 * @file HierarchicalSDRStrategy.cpp
 * @brief Implementation of V4b hierarchical binary SDR weight compression.
 *
 * Mirrors research/step0/exp_a_v3_combined/combined.py::fit_hierarchical_ksvd
 * and exp_a_v2_binary/variants.py::decode_binary. Per-tensor dictionary is
 * fit at compress time via K-SVD with hierarchical greedy matching pursuit;
 * decompression is a pure dictionary lookup + signed accumulation.
 */
#include "HierarchicalSDRStrategy.hpp"
#include "../utils/fp16_convert.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace CortexAICompression {

namespace {

// --------------------------------------------------------------------------
// Serialization header
// --------------------------------------------------------------------------
// 4-byte magic ('H','S','D','R') + uint32 version + the on-disk config form.
// Kept packed and explicit so the byte stream is parser-friendly across
// platforms; field order MUST match the writer and reader.
constexpr char kHSDRMagic[4] = {'H', 'S', 'D', 'R'};
constexpr uint32_t kHSDRVersion = 2;  // v2 adds edge_bytes_count for non-aligned shapes

#pragma pack(push, 1)
struct HSDRHeader {
    char     magic[4];          // "HSDR"
    uint32_t version;           // kHSDRVersion
    uint32_t original_rows;
    uint32_t original_cols;
    uint32_t n_tiles;
    uint16_t tile_rows;
    uint16_t tile_cols;
    uint16_t n_atoms;
    uint8_t  n_stages;
    uint8_t  active_bits_per_stage;
    float    stage_decay;
    uint32_t edge_bytes_count;  // v2: FP32 bytes for the un-tileable edge strips (0 if aligned)
};
#pragma pack(pop)
static_assert(sizeof(HSDRHeader) == 4 + 4 + 4 + 4 + 4 + 2 + 2 + 2 + 1 + 1 + 4 + 4,
              "HSDRHeader size mismatch — packing broken");

// Edge-strip layout (when the tensor isn't a clean multiple of (tile_rows, tile_cols)):
//   • top-right strip:  rows [0..R_full),    cols [C_full..C),  row-major FP32
//   • bottom strip:     rows [R_full..R),    cols [0..C),       row-major FP32
//   total edge bytes = R_full*(C-C_full)*4 + (R-R_full)*C*4
// Both strips are stored verbatim. V4b only sees the aligned (R_full × C_full) block.

// Pack (atom_index, sign) into one uint16: high bit = sign (0=+, 1=-), low 15 bits = index.
// Supports K up to 32767; our designs use 256-512 so plenty of headroom.
inline uint16_t packIndexSign(uint16_t atom_index, int8_t sign) {
    uint16_t v = static_cast<uint16_t>(atom_index & 0x7FFFu);
    if (sign < 0) v |= 0x8000u;
    return v;
}
inline void unpackIndexSign(uint16_t packed, uint16_t& atom_index, int8_t& sign) {
    atom_index = static_cast<uint16_t>(packed & 0x7FFFu);
    sign = (packed & 0x8000u) ? -1 : +1;
}

// --------------------------------------------------------------------------
// Tile extraction / reassembly
// --------------------------------------------------------------------------

/**
 * Flatten an (R, C) FP32 weight matrix into (n_tiles, tile_rows*tile_cols)
 * row-major. Tiles are extracted in row-major tile order: (block_row, block_col).
 * Caller has already validated R % tile_rows == 0 and C % tile_cols == 0.
 */
std::vector<float> extractTiles(const float* weight, uint32_t R, uint32_t C,
                                 uint16_t tile_rows, uint16_t tile_cols,
                                 uint32_t& n_tiles_out) {
    const uint32_t n_row_tiles = R / tile_rows;
    const uint32_t n_col_tiles = C / tile_cols;
    n_tiles_out = n_row_tiles * n_col_tiles;
    const uint32_t tile_size = static_cast<uint32_t>(tile_rows) * tile_cols;

    std::vector<float> tiles(static_cast<size_t>(n_tiles_out) * tile_size);
    for (uint32_t br = 0; br < n_row_tiles; ++br) {
        for (uint32_t bc = 0; bc < n_col_tiles; ++bc) {
            const uint32_t tile_idx = br * n_col_tiles + bc;
            float* dst = tiles.data() + static_cast<size_t>(tile_idx) * tile_size;
            for (uint16_t r = 0; r < tile_rows; ++r) {
                const uint32_t src_row = br * tile_rows + r;
                const float* src = weight + static_cast<size_t>(src_row) * C + bc * tile_cols;
                std::memcpy(dst + r * tile_cols, src, sizeof(float) * tile_cols);
            }
        }
    }
    return tiles;
}

/** Inverse of extractTiles; writes reconstructed tiles back into a (R, C) matrix. */
void reassembleTiles(const float* tiles, uint32_t R, uint32_t C,
                     uint16_t tile_rows, uint16_t tile_cols, float* out_weight) {
    const uint32_t n_row_tiles = R / tile_rows;
    const uint32_t n_col_tiles = C / tile_cols;
    for (uint32_t br = 0; br < n_row_tiles; ++br) {
        for (uint32_t bc = 0; bc < n_col_tiles; ++bc) {
            const uint32_t tile_idx = br * n_col_tiles + bc;
            const float* src = tiles + static_cast<size_t>(tile_idx)
                                * (static_cast<size_t>(tile_rows) * tile_cols);
            for (uint16_t r = 0; r < tile_rows; ++r) {
                const uint32_t dst_row = br * tile_rows + r;
                float* dst = out_weight + static_cast<size_t>(dst_row) * C + bc * tile_cols;
                std::memcpy(dst, src + r * tile_cols, sizeof(float) * tile_cols);
            }
        }
    }
}

// --------------------------------------------------------------------------
// Hierarchical greedy matching pursuit (encode step of K-SVD)
// --------------------------------------------------------------------------

/**
 * For a single tile of dimension C, pick (n_stages × active_bits_per_stage)
 * (atom, sign) pairs that minimize the residual under
 *      tile ≈ Σ_l γ_l · Σ_k sign_{l,k} · D[idx_{l,k}].
 *
 * indices_out and signs_out are written with shape (n_stages, active_bits_per_stage)
 * in row-major contiguous form, totalling n_stages*k entries each.
 *
 * Within a stage we forbid re-selecting the same atom (set per-stage); across
 * stages the same atom may be picked again.
 */
void hierarchicalBinaryCode(const float* tile, const float* dictionary,
                            uint16_t n_atoms, uint32_t C,
                            uint8_t n_stages, uint8_t k_per_stage,
                            const float* stage_scales,
                            uint16_t* indices_out, int8_t* signs_out) {
    // Working residual
    std::vector<float> r(tile, tile + C);
    // Scratch for projection scores
    std::vector<float> proj(n_atoms);
    std::vector<uint8_t> used_this_stage(n_atoms, 0);

    for (uint8_t l = 0; l < n_stages; ++l) {
        const float gamma = stage_scales[l];
        std::fill(used_this_stage.begin(), used_this_stage.end(), 0);

        for (uint8_t step = 0; step < k_per_stage; ++step) {
            // proj[j] = <D[j], r> = sum_c D[j,c] * r[c]
            for (uint16_t j = 0; j < n_atoms; ++j) {
                const float* atom = dictionary + static_cast<size_t>(j) * C;
                float s = 0.0f;
                for (uint32_t c = 0; c < C; ++c) s += atom[c] * r[c];
                proj[j] = used_this_stage[j] ? 0.0f : s;
            }

            // argmax |proj|
            uint16_t best = 0;
            float best_abs = 0.0f;
            for (uint16_t j = 0; j < n_atoms; ++j) {
                const float a = std::fabs(proj[j]);
                if (a > best_abs) { best_abs = a; best = j; }
            }
            const int8_t sign = (proj[best] >= 0.0f) ? +1 : -1;
            const size_t out_idx = static_cast<size_t>(l) * k_per_stage + step;
            indices_out[out_idx] = best;
            signs_out[out_idx] = sign;
            used_this_stage[best] = 1;

            // r -= gamma * sign * D[best]
            const float scale = gamma * static_cast<float>(sign);
            const float* atom = dictionary + static_cast<size_t>(best) * C;
            for (uint32_t c = 0; c < C; ++c) r[c] -= scale * atom[c];
        }
    }
}

// --------------------------------------------------------------------------
// Atom update step (K-SVD) — sequential, mirrors Python combined.py.
// --------------------------------------------------------------------------

/**
 * Update each atom j to minimize the squared sum-of-stages reconstruction error
 * over the tiles that currently use atom j.
 *
 * For users (i, l, k) where indices[i,l,k] == j and sign = signs[i,l,k]:
 *   target_u = tile_i - reconstruction_without_atom_j_at_(l,k)
 *            = residual_i + sign · γ_l · D[j]_old
 * Closed-form update (weighted least squares):
 *   D[j] = D[j]_old + (Σ_u sign_u · γ_u · residual_i) / (Σ_u γ_u^2)
 *
 * Recomputes the full residual at the start of each iteration (cheap relative
 * to the matching-pursuit step).
 */
void atomUpdateStep(const float* tiles, uint32_t n_tiles, uint32_t C,
                    float* D, uint16_t n_atoms,
                    const uint16_t* indices, const int8_t* signs,
                    uint8_t n_stages, uint8_t k_per_stage,
                    const float* stage_scales,
                    std::mt19937& rng) {
    const uint32_t slots_per_tile = static_cast<uint32_t>(n_stages) * k_per_stage;

    // Full reconstruction for each tile, then residual = tile - recon
    std::vector<float> full_recon(static_cast<size_t>(n_tiles) * C, 0.0f);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        float* r = full_recon.data() + static_cast<size_t>(i) * C;
        for (uint8_t l = 0; l < n_stages; ++l) {
            const float gamma = stage_scales[l];
            for (uint8_t k = 0; k < k_per_stage; ++k) {
                const size_t slot = static_cast<size_t>(i) * slots_per_tile
                                    + static_cast<size_t>(l) * k_per_stage + k;
                const uint16_t idx = indices[slot];
                const int8_t s = signs[slot];
                if (s == 0) continue;
                const float scale = gamma * static_cast<float>(s);
                const float* atom = D + static_cast<size_t>(idx) * C;
                for (uint32_t c = 0; c < C; ++c) r[c] += scale * atom[c];
            }
        }
    }

    std::vector<float> residual(static_cast<size_t>(n_tiles) * C);
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = tiles[i] - full_recon[i];
    }

    // Per-atom: walk all (i, l, k) slots once and gather contributions.
    // Index buffer of users to avoid a second pass per atom.
    std::vector<std::vector<uint32_t>> users(n_atoms);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        for (uint8_t l = 0; l < n_stages; ++l) {
            for (uint8_t k = 0; k < k_per_stage; ++k) {
                const size_t slot = static_cast<size_t>(i) * slots_per_tile
                                    + static_cast<size_t>(l) * k_per_stage + k;
                const uint16_t j = indices[slot];
                if (signs[slot] == 0) continue;
                // Encode slot as (i << 16) | (l*k_per_stage + k) for compact storage.
                const uint32_t enc = (i << 16) | static_cast<uint32_t>(l * k_per_stage + k);
                users[j].push_back(enc);
            }
        }
    }

    std::vector<float> correction(C);
    std::uniform_int_distribution<uint32_t> tile_picker(0, n_tiles - 1);

    for (uint16_t j = 0; j < n_atoms; ++j) {
        const auto& u = users[j];
        if (u.empty()) {
            // Dead atom — re-seed with a random tile.
            const uint32_t rid = tile_picker(rng);
            std::memcpy(D + static_cast<size_t>(j) * C,
                        tiles + static_cast<size_t>(rid) * C,
                        sizeof(float) * C);
            continue;
        }

        std::fill(correction.begin(), correction.end(), 0.0f);
        float weight_sum = 0.0f;
        for (uint32_t enc : u) {
            const uint32_t i = enc >> 16;
            const uint16_t loc = static_cast<uint16_t>(enc & 0xFFFFu);
            const uint8_t l = static_cast<uint8_t>(loc / k_per_stage);
            const size_t slot = static_cast<size_t>(i) * slots_per_tile + loc;
            const float gamma = stage_scales[l];
            const float sg = static_cast<float>(signs[slot]) * gamma;
            const float* res_i = residual.data() + static_cast<size_t>(i) * C;
            for (uint32_t c = 0; c < C; ++c) correction[c] += sg * res_i[c];
            weight_sum += gamma * gamma;
        }
        if (weight_sum <= 1e-12f) continue;
        const float inv_w = 1.0f / weight_sum;
        float* atom = D + static_cast<size_t>(j) * C;
        for (uint32_t c = 0; c < C; ++c) atom[c] += correction[c] * inv_w;
    }
}

// --------------------------------------------------------------------------
// Top-level fit: dictionary + binary codes
// --------------------------------------------------------------------------
void fitHierarchicalKSVD(const float* tiles, uint32_t n_tiles, uint32_t C,
                         const HierarchicalSDRConfig& cfg,
                         std::vector<float>& D_out,
                         std::vector<uint16_t>& indices_out,
                         std::vector<int8_t>& signs_out,
                         std::vector<float>& stage_scales_out) {
    const uint16_t K = cfg.n_atoms;
    const uint8_t S = cfg.n_stages;
    const uint8_t k = cfg.active_bits_per_stage;
    const uint32_t slots_per_tile = static_cast<uint32_t>(S) * k;

    // Stage scales: γ_l = stage_decay^l.
    stage_scales_out.resize(S);
    for (uint8_t l = 0; l < S; ++l) {
        stage_scales_out[l] = std::pow(cfg.stage_decay, static_cast<float>(l));
    }

    // Init D: random tile selection, unit-normalised, then scaled so that
    // sum-of-±atoms reconstruction starts in the right magnitude range.
    std::mt19937 rng(0xC0DE);
    std::uniform_int_distribution<uint32_t> tile_picker(0, n_tiles - 1);

    D_out.assign(static_cast<size_t>(K) * C, 0.0f);
    for (uint16_t j = 0; j < K; ++j) {
        uint32_t src = tile_picker(rng);
        const float* src_tile = tiles + static_cast<size_t>(src) * C;
        float* dst_atom = D_out.data() + static_cast<size_t>(j) * C;
        std::memcpy(dst_atom, src_tile, sizeof(float) * C);
        // Normalise to unit length
        float sq = 0.0f;
        for (uint32_t c = 0; c < C; ++c) sq += dst_atom[c] * dst_atom[c];
        const float norm = std::sqrt(std::max(sq, 1e-24f));
        for (uint32_t c = 0; c < C; ++c) dst_atom[c] /= norm;
    }
    // Scale dictionary so that ±atoms reconstruct at the right tile magnitude.
    double tile_norm_sum = 0.0;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        const float* t = tiles + static_cast<size_t>(i) * C;
        double sq = 0.0;
        for (uint32_t c = 0; c < C; ++c) sq += static_cast<double>(t[c]) * t[c];
        tile_norm_sum += std::sqrt(sq);
    }
    const float tile_norm_mean = static_cast<float>(tile_norm_sum / std::max<uint32_t>(1u, n_tiles));
    const float init_scale = tile_norm_mean / std::max(1.0f, std::sqrt(static_cast<float>(slots_per_tile)));
    for (size_t i = 0; i < D_out.size(); ++i) D_out[i] *= init_scale;

    indices_out.assign(static_cast<size_t>(n_tiles) * slots_per_tile, 0);
    signs_out.assign(static_cast<size_t>(n_tiles) * slots_per_tile, 0);

    for (uint8_t iter = 0; iter < cfg.ksvd_iters; ++iter) {
        // Code step: independent per tile.
        for (uint32_t i = 0; i < n_tiles; ++i) {
            const float* tile = tiles + static_cast<size_t>(i) * C;
            uint16_t* idx_dst = indices_out.data() + static_cast<size_t>(i) * slots_per_tile;
            int8_t*   sgn_dst = signs_out.data()   + static_cast<size_t>(i) * slots_per_tile;
            hierarchicalBinaryCode(tile, D_out.data(), K, C, S, k,
                                   stage_scales_out.data(), idx_dst, sgn_dst);
        }
        // Atom update step.
        atomUpdateStep(tiles, n_tiles, C, D_out.data(), K,
                       indices_out.data(), signs_out.data(),
                       S, k, stage_scales_out.data(), rng);
    }
}

// --------------------------------------------------------------------------
// Decode (dictionary + codes → tiles → weight matrix)
// --------------------------------------------------------------------------
void decodeAllTiles(const float* D, uint16_t n_atoms, uint32_t C,
                    const uint16_t* indices, const int8_t* signs,
                    uint32_t n_tiles, uint8_t n_stages, uint8_t k_per_stage,
                    const float* stage_scales, float* out_tiles) {
    const uint32_t slots_per_tile = static_cast<uint32_t>(n_stages) * k_per_stage;
    std::fill(out_tiles, out_tiles + static_cast<size_t>(n_tiles) * C, 0.0f);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        float* t = out_tiles + static_cast<size_t>(i) * C;
        for (uint8_t l = 0; l < n_stages; ++l) {
            const float gamma = stage_scales[l];
            for (uint8_t k = 0; k < k_per_stage; ++k) {
                const size_t slot = static_cast<size_t>(i) * slots_per_tile
                                    + static_cast<size_t>(l) * k_per_stage + k;
                const uint16_t idx = indices[slot];
                const int8_t s = signs[slot];
                if (s == 0) continue;
                const float scale = gamma * static_cast<float>(s);
                const float* atom = D + static_cast<size_t>(idx) * C;
                for (uint32_t c = 0; c < C; ++c) t[c] += scale * atom[c];
            }
        }
    }
}

// --------------------------------------------------------------------------
// Helpers for byte append/read
// --------------------------------------------------------------------------
template <typename T>
void appendPOD(std::vector<std::byte>& dst, const T& value) {
    const auto* p = reinterpret_cast<const std::byte*>(&value);
    dst.insert(dst.end(), p, p + sizeof(T));
}

template <typename T>
void appendBuffer(std::vector<std::byte>& dst, const T* data, size_t count) {
    const auto* p = reinterpret_cast<const std::byte*>(data);
    dst.insert(dst.end(), p, p + sizeof(T) * count);
}

template <typename T>
void readPOD(const std::byte*& cursor, const std::byte* end, T& value) {
    if (cursor + sizeof(T) > end) {
        throw CompressionError("HSDR decompress: unexpected end-of-stream");
    }
    std::memcpy(&value, cursor, sizeof(T));
    cursor += sizeof(T);
}

template <typename T>
void readBuffer(const std::byte*& cursor, const std::byte* end, T* out, size_t count) {
    const size_t bytes = sizeof(T) * count;
    if (cursor + bytes > end) {
        throw CompressionError("HSDR decompress: truncated stream");
    }
    std::memcpy(out, cursor, bytes);
    cursor += bytes;
}

// --------------------------------------------------------------------------
// Codes-only stream header — used by the shared-dictionary path. The
// dictionary is stored externally so it's not embedded here, but everything
// else (shape, tile geometry, stage scales, packed codes) is.
// --------------------------------------------------------------------------
constexpr char kHSDRCodesMagic[4] = {'H', 'S', 'D', 'C'};
constexpr uint32_t kHSDRCodesVersion = 1;

#pragma pack(push, 1)
struct HSDRCodesHeader {
    char     magic[4];
    uint32_t version;
    uint32_t original_rows;
    uint32_t original_cols;
    uint32_t n_tiles;
    uint16_t tile_rows;
    uint16_t tile_cols;
    uint16_t n_atoms;          // dictionary size (must match external dict)
    uint8_t  n_stages;
    uint8_t  active_bits_per_stage;
    float    stage_decay;
};
#pragma pack(pop)

/// Normalise a data_format string to lowercase for comparison.
std::string lowerFormat(const std::string& fmt) {
    std::string out;
    out.reserve(fmt.size());
    for (char c : fmt) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

/// Dequantise a segment to FP32 in-memory. Supports:
///   • FP32                          — zero-copy reinterpret returned via output vector
///   • FP16 (data_format == "f16")   — SIMD-accelerated via Utils::fp16_to_fp32_array
///   • BF16                          — top-16-bit shift up to FP32
///   • INT8 with TensorMetadata scale/zero_point — linear dequant
///
/// Throws CompressionError for unsupported dtypes (e.g. GGUF block-quantised
/// formats Q4_K_M etc., which need llama.cpp-style per-block code).
///
/// Returns a vector<float> of length R*C. The HSDR fitter consumes this
/// without caring whether the original was lower-precision.
std::vector<float> dequantizeSegmentToFP32(const ModelSegment& segment) {
    using namespace Utils;
    const std::string fmt = lowerFormat(segment.data_format);

    // ---------------------------------------------------------------------
    // Dispatch by `data_format` first (more specific than `type`, e.g. BF16
    // and FP16 both ride the WEIGHTS_FP16 enum). Fall through to the type
    // enum only when the format string is empty / unknown.
    // ---------------------------------------------------------------------

    // BF16 — top 16 bits of FP32 representation. Must come before the FP16
    // type-based branch because BF16 data is often tagged as WEIGHTS_FP16.
    if (fmt == "bf16" || fmt == "bfloat16") {
        if (segment.data.size() % sizeof(uint16_t) != 0) {
            throw CompressionError("HSDR dequant: BF16 byte size not a multiple of 2");
        }
        const size_t n = segment.data.size() / sizeof(uint16_t);
        std::vector<float> out(n);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(segment.data.data());
        for (size_t i = 0; i < n; ++i) {
            uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
            std::memcpy(&out[i], &bits, sizeof(float));
        }
        return out;
    }

    // GGUF block formats (Q4_0, Q8_0) — must come before the WEIGHTS_INT8 /
    // WEIGHTS_INT4 type-based branches, because those segments often carry
    // SegmentType::WEIGHTS_INT8 even though the byte layout is block-quantised.
    if (fmt == "q4_0") {
        constexpr size_t QK = 32;
        constexpr size_t BLOCK_BYTES = sizeof(uint16_t) + QK / 2;  // 18
        static_assert(BLOCK_BYTES == 18, "Q4_0 block size");
        if (segment.data.size() % BLOCK_BYTES != 0) {
            throw CompressionError("HSDR dequant: Q4_0 byte size not a multiple of 18");
        }
        const size_t n_blocks = segment.data.size() / BLOCK_BYTES;
        std::vector<float> out(n_blocks * QK);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(segment.data.data());
        for (size_t b = 0; b < n_blocks; ++b) {
            uint16_t d_bits;
            std::memcpy(&d_bits, p, sizeof(uint16_t));
            const float d = fp16_to_fp32(d_bits);
            const uint8_t* qs = p + 2;
            for (size_t j = 0; j < QK / 2; ++j) {
                const uint8_t byte = qs[j];
                const int low  = static_cast<int>(byte & 0x0F) - 8;
                const int high = static_cast<int>(byte >> 4)   - 8;
                out[b * QK + 2 * j    ] = static_cast<float>(low)  * d;
                out[b * QK + 2 * j + 1] = static_cast<float>(high) * d;
            }
            p += BLOCK_BYTES;
        }
        return out;
    }

    if (fmt == "q8_0") {
        constexpr size_t QK = 32;
        constexpr size_t BLOCK_BYTES = sizeof(uint16_t) + QK;  // 34
        static_assert(BLOCK_BYTES == 34, "Q8_0 block size");
        if (segment.data.size() % BLOCK_BYTES != 0) {
            throw CompressionError("HSDR dequant: Q8_0 byte size not a multiple of 34");
        }
        const size_t n_blocks = segment.data.size() / BLOCK_BYTES;
        std::vector<float> out(n_blocks * QK);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(segment.data.data());
        for (size_t b = 0; b < n_blocks; ++b) {
            uint16_t d_bits;
            std::memcpy(&d_bits, p, sizeof(uint16_t));
            const float d = fp16_to_fp32(d_bits);
            const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
            for (size_t j = 0; j < QK; ++j) {
                out[b * QK + j] = static_cast<float>(qs[j]) * d;
            }
            p += BLOCK_BYTES;
        }
        return out;
    }

    // FP32 fast path — copy bytes into a float vector.
    if (segment.type == SegmentType::WEIGHTS_FP32 ||
        fmt == "f32" || fmt == "fp32" || fmt == "float32") {
        if (segment.data.size() % sizeof(float) != 0) {
            throw CompressionError("HSDR dequant: FP32 byte size not a multiple of 4");
        }
        const size_t n = segment.data.size() / sizeof(float);
        std::vector<float> out(n);
        std::memcpy(out.data(), segment.data.data(), segment.data.size());
        return out;
    }

    // FP16 — uses the existing SIMD-accelerated converter (F16C / NEON / scalar).
    if (segment.type == SegmentType::WEIGHTS_FP16 ||
        fmt == "f16" || fmt == "fp16" || fmt == "float16" || fmt == "half") {
        if (segment.data.size() % sizeof(uint16_t) != 0) {
            throw CompressionError("HSDR dequant: FP16 byte size not a multiple of 2");
        }
        const size_t n = segment.data.size() / sizeof(uint16_t);
        std::vector<float> out(n);
        fp16_to_fp32_array(reinterpret_cast<const uint16_t*>(segment.data.data()),
                            out.data(), n);
        return out;
    }

    // INT8 with uniform (scale, zero_point) — requires TensorMetadata to carry the params.
    if (segment.type == SegmentType::WEIGHTS_INT8 || fmt == "i8" || fmt == "int8") {
        if (!segment.tensor_metadata.has_value()
            || !segment.tensor_metadata.value().scale.has_value()) {
            throw CompressionError("HSDR dequant: INT8 input needs tensor_metadata.scale");
        }
        const float scale = segment.tensor_metadata.value().scale.value();
        const float zp = segment.tensor_metadata.value().zero_point.value_or(0.0f);
        const size_t n = segment.data.size();  // 1 byte per element
        std::vector<float> out(n);
        const int8_t* src = reinterpret_cast<const int8_t*>(segment.data.data());
        for (size_t i = 0; i < n; ++i) {
            out[i] = (static_cast<float>(src[i]) - zp) * scale;
        }
        return out;
    }

    // Everything else (Q4_K, Q5_K, Q6_K, Q8_K, IQ*-, AWQ, GPTQ, etc.) requires
    // format-specific block-dequant code with super-block + sub-block layouts.
    // Carry-overs to a future round; documented in STEP0 followups.
    throw CompressionError("HSDR dequant: unsupported input dtype '" + segment.data_format
                            + "' (type=" + std::to_string(static_cast<int>(segment.type))
                            + "); supported: f32, f16, bf16, i8, q4_0, q8_0");
}

/// Drive the encode step (greedy MP) for a whole tensor against an external
/// dictionary. Used by both `compressWithExternalDictionary` (single-tile-row
/// encoding) and as a helper for full segment encoding.
void encodeAllTilesAgainstDictionary(const float* tiles, uint32_t n_tiles, uint32_t C,
                                     const float* D, uint16_t K,
                                     uint8_t n_stages, uint8_t k_per_stage,
                                     const float* stage_scales,
                                     uint16_t* indices_out, int8_t* signs_out) {
    const uint32_t slots_per_tile = static_cast<uint32_t>(n_stages) * k_per_stage;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        const float* tile = tiles + static_cast<size_t>(i) * C;
        uint16_t* idx_dst = indices_out + static_cast<size_t>(i) * slots_per_tile;
        int8_t*   sgn_dst = signs_out   + static_cast<size_t>(i) * slots_per_tile;
        hierarchicalBinaryCode(tile, D, K, C, n_stages, k_per_stage,
                                stage_scales, idx_dst, sgn_dst);
    }
}

} // anonymous namespace

// --------------------------------------------------------------------------
// Strategy implementation
// --------------------------------------------------------------------------

HierarchicalSDRConfig HierarchicalSDRStrategy::configFor(const ModelSegment& segment) const {
    // Match the B.2 adaptive role-aware allocation: MLP gets the bigger config.
    switch (segment.type) {
        case SegmentType::FEED_FORWARD_WEIGHTS:
            return mlp_default_;
        case SegmentType::EMBEDDING_WEIGHTS:
        case SegmentType::ATTENTION_WEIGHTS:
        case SegmentType::LAYER_NORM_WEIGHTS:
        case SegmentType::WEIGHTS_FP32:
        default:
            break;
    }
    // Fallback: inspect name for "mlp"/"ffn" pattern.
    const std::string& n = segment.name;
    if (n.find("mlp") != std::string::npos || n.find("ffn") != std::string::npos
        || n.find("feed_forward") != std::string::npos) {
        return mlp_default_;
    }
    return attn_default_;
}

std::vector<std::byte> HierarchicalSDRStrategy::compress(const ModelSegment& segment) const {
    if (!segment.isWeightTensor()) {
        throw CompressionError("HierarchicalSDRStrategy: only weight tensors are supported");
    }
    // Hybrid FP16 protection (followup #4): a configurable predicate can mark
    // certain segments as "protected" — HSDR throws here, the AICompressor
    // chain then falls through to the next-priority strategy (Quant/Gzip).
    if (protection_ && protection_(segment)) {
        throw CompressionError("HSDR: segment '" + segment.name +
                               "' is protected; deferring to lossless fallback");
    }

    // Dequantise to FP32. Throws CompressionError for unsupported dtypes
    // (the strategy chain in AICompressor then falls through to the next
    // priority — typically QuantizedTensorStrategy or Gzip).
    std::vector<float> fp32_view = dequantizeSegmentToFP32(segment);
    const size_t num_elements = fp32_view.size();

    // Need a 2-D shape; pull from tensor metadata.
    if (!segment.tensor_metadata.has_value() ||
        segment.tensor_metadata.value().dimensions.size() != 2) {
        throw CompressionError("HSDR: tensor_metadata.dimensions must be 2-D (rows, cols)");
    }
    const auto& dims = segment.tensor_metadata.value().dimensions;
    const uint32_t R = static_cast<uint32_t>(dims[0]);
    const uint32_t C = static_cast<uint32_t>(dims[1]);
    if (static_cast<size_t>(R) * C != num_elements) {
        throw CompressionError("HSDR: tensor_metadata dims don't match data size");
    }

    const HierarchicalSDRConfig cfg = configFor(segment);
    const uint32_t R_full = (R / cfg.tile_rows) * cfg.tile_rows;
    const uint32_t C_full = (C / cfg.tile_cols) * cfg.tile_cols;
    if (R_full == 0 || C_full == 0) {
        throw CompressionError("HSDR: tensor too small to hold even one full tile");
    }
    const uint32_t n_row_tiles = R_full / cfg.tile_rows;
    const uint32_t n_col_tiles = C_full / cfg.tile_cols;
    const uint32_t n_tiles = n_row_tiles * n_col_tiles;
    const uint32_t tile_dim = cfg.tileSize();
    if (n_tiles < cfg.n_atoms) {
        throw CompressionError("HSDR: not enough tiles to fit the requested dictionary size");
    }

    // `fp32_view` holds the (possibly dequantised) FP32 weight values. This is
    // what HSDR fits — the original dtype (FP16/BF16/INT8) is invisible past
    // dequantizeSegmentToFP32.
    const float* weight = fp32_view.data();

    // Slice out the aligned (R_full × C_full) top-left block into a contiguous
    // buffer so extractTiles' stride matches. Edge strips outside this block
    // get stored as raw FP32 after the packed codes (see header comment).
    std::vector<float> aligned;
    aligned.reserve(static_cast<size_t>(R_full) * C_full);
    if (R_full == R && C_full == C) {
        aligned.assign(weight, weight + static_cast<size_t>(R) * C);
    } else {
        aligned.resize(static_cast<size_t>(R_full) * C_full);
        for (uint32_t r = 0; r < R_full; ++r) {
            std::memcpy(aligned.data() + static_cast<size_t>(r) * C_full,
                        weight + static_cast<size_t>(r) * C,
                        sizeof(float) * C_full);
        }
    }
    uint32_t n_tiles_check = 0;
    std::vector<float> tiles = extractTiles(aligned.data(), R_full, C_full,
                                             cfg.tile_rows, cfg.tile_cols, n_tiles_check);
    if (n_tiles_check != n_tiles) {
        throw CompressionError("HSDR: internal tile-count mismatch");
    }

    // Fit V4b: dictionary + per-tile (indices, signs).
    std::vector<float> D, stage_scales;
    std::vector<uint16_t> indices;
    std::vector<int8_t>   signs;
    fitHierarchicalKSVD(tiles.data(), n_tiles, tile_dim, cfg, D, indices, signs, stage_scales);

    // Build edge-strip bytes (top-right + bottom). 0 bytes if shape is aligned.
    std::vector<float> edge_buffer;
    const uint32_t top_right_floats = R_full * (C - C_full);
    const uint32_t bottom_floats    = (R - R_full) * C;
    edge_buffer.reserve(static_cast<size_t>(top_right_floats) + bottom_floats);
    for (uint32_t r = 0; r < R_full; ++r) {
        for (uint32_t c = C_full; c < C; ++c) {
            edge_buffer.push_back(weight[static_cast<size_t>(r) * C + c]);
        }
    }
    for (uint32_t r = R_full; r < R; ++r) {
        for (uint32_t c = 0; c < C; ++c) {
            edge_buffer.push_back(weight[static_cast<size_t>(r) * C + c]);
        }
    }
    const uint32_t edge_bytes_count = static_cast<uint32_t>(edge_buffer.size() * sizeof(float));

    // Serialize.
    HSDRHeader header{};
    std::memcpy(header.magic, kHSDRMagic, 4);
    header.version = kHSDRVersion;
    header.original_rows = R;
    header.original_cols = C;
    header.n_tiles = n_tiles;
    header.tile_rows = cfg.tile_rows;
    header.tile_cols = cfg.tile_cols;
    header.n_atoms = cfg.n_atoms;
    header.n_stages = cfg.n_stages;
    header.active_bits_per_stage = cfg.active_bits_per_stage;
    header.stage_decay = cfg.stage_decay;
    header.edge_bytes_count = edge_bytes_count;

    const uint32_t slots_per_tile = static_cast<uint32_t>(cfg.n_stages) * cfg.active_bits_per_stage;
    const size_t packed_codes = static_cast<size_t>(n_tiles) * slots_per_tile;

    std::vector<std::byte> out;
    out.reserve(sizeof(HSDRHeader)
                + sizeof(float) * stage_scales.size()
                + sizeof(float) * D.size()
                + sizeof(uint16_t) * packed_codes
                + edge_bytes_count);

    appendPOD(out, header);
    appendBuffer(out, stage_scales.data(), stage_scales.size());
    appendBuffer(out, D.data(), D.size());

    // Pack each (index, sign) pair into a single uint16.
    std::vector<uint16_t> packed(packed_codes);
    for (size_t i = 0; i < packed_codes; ++i) {
        packed[i] = packIndexSign(indices[i], signs[i]);
    }
    appendBuffer(out, packed.data(), packed.size());

    if (edge_bytes_count > 0) {
        appendBuffer(out, edge_buffer.data(), edge_buffer.size());
    }

    return out;
}

// --------------------------------------------------------------------------
// Fused matmul: Y = W · x directly from the compressed stream
// --------------------------------------------------------------------------
//
// For 1D row tiles (tile_rows == 1):
//   Each atom D[a] is a row vector of length C (= tile_cols = input dim).
//   Each row r of W is the sum (across stages) of signed atoms.
//   Y[r, b] = w_r · x[:, b]
//           = Σ_l γ_l · Σ_k sign · (D[idx_{r,l,k}] · x[:, b])
//   With Dx[a, b] = D[a] · x[:, b] precomputed once, per-row work collapses
//   to (n_active · batch) adds — a real speed win over dense matmul.
//
// For 2D tiles (tile_rows > 1):
//   Each atom is a (tile_rows, tile_cols) matrix. The precompute "Dx for
//   every atom × every column block" costs more FLOPs than the dense product
//   for typical LLM shapes. We refuse this path; callers must materialise
//   the weight via decompress() and use a regular dense matmul.
std::vector<float> HierarchicalSDRStrategy::matmulRowMajor(
    const std::vector<std::byte>& compressedData,
    const float* x,
    size_t batch) const
{
    const std::byte* cursor = compressedData.data();
    const std::byte* end = cursor + compressedData.size();

    HSDRHeader header{};
    readPOD(cursor, end, header);
    if (std::memcmp(header.magic, kHSDRMagic, 4) != 0) {
        throw CompressionError("HSDR matmul: bad magic — not an HSDR stream");
    }
    if (header.version != kHSDRVersion) {
        throw CompressionError("HSDR matmul: unsupported version");
    }
    if (header.tile_rows != 1) {
        throw CompressionError("HSDR matmul: fused matmul requires tile_rows == 1; "
                               "use decompress() + dense matmul for 2D tile encodings");
    }
    if (header.tile_cols != header.original_cols) {
        throw CompressionError("HSDR matmul: 1D row-tile encoding must have tile_cols == "
                               "original_cols (each tile is exactly one row of W)");
    }
    const uint32_t R = header.original_rows;
    const uint32_t C = header.original_cols;
    const uint16_t K = header.n_atoms;
    const uint8_t S = header.n_stages;
    const uint8_t k = header.active_bits_per_stage;
    const uint32_t n_tiles = header.n_tiles;
    if (n_tiles != R) {
        throw CompressionError("HSDR matmul: tile count must equal row count");
    }
    const uint32_t slots_per_tile = static_cast<uint32_t>(S) * k;

    std::vector<float> stage_scales(S);
    readBuffer(cursor, end, stage_scales.data(), S);

    std::vector<float> D(static_cast<size_t>(K) * C);
    readBuffer(cursor, end, D.data(), D.size());

    std::vector<uint16_t> packed(static_cast<size_t>(n_tiles) * slots_per_tile);
    readBuffer(cursor, end, packed.data(), packed.size());

    // Precompute Dx[a, b] = D[a] · x[:, b]   shape (K, batch)
    // K * C * batch * 2 FLOPs. This is the one place we still do real
    // multiplies in the fused path.
    std::vector<float> Dx(static_cast<size_t>(K) * batch, 0.0f);
    for (uint16_t a = 0; a < K; ++a) {
        const float* atom = D.data() + static_cast<size_t>(a) * C;
        float* out_row = Dx.data() + static_cast<size_t>(a) * batch;
        for (uint32_t c = 0; c < C; ++c) {
            const float av = atom[c];
            const float* xrow = x + static_cast<size_t>(c) * batch;
            for (size_t b = 0; b < batch; ++b) {
                out_row[b] += av * xrow[b];
            }
        }
    }

    // For each row r of the output Y, gather signed contributions from Dx.
    std::vector<float> Y(static_cast<size_t>(R) * batch, 0.0f);
    for (uint32_t r = 0; r < R; ++r) {
        float* y_row = Y.data() + static_cast<size_t>(r) * batch;
        const uint16_t* row_codes = packed.data() + static_cast<size_t>(r) * slots_per_tile;
        for (uint8_t l = 0; l < S; ++l) {
            const float gamma = stage_scales[l];
            for (uint8_t kk = 0; kk < k; ++kk) {
                uint16_t atom_idx;
                int8_t sign;
                unpackIndexSign(row_codes[l * k + kk], atom_idx, sign);
                if (sign == 0) continue;
                const float scale = gamma * static_cast<float>(sign);
                const float* dx_row = Dx.data() + static_cast<size_t>(atom_idx) * batch;
                for (size_t b = 0; b < batch; ++b) {
                    y_row[b] += scale * dx_row[b];
                }
            }
        }
    }
    return Y;
}

std::vector<std::byte> HierarchicalSDRStrategy::decompress(
    const std::vector<std::byte>& compressedData,
    SegmentType /*originalType*/,
    size_t originalSize) const
{
    const std::byte* cursor = compressedData.data();
    const std::byte* end = cursor + compressedData.size();

    HSDRHeader header{};
    readPOD(cursor, end, header);
    if (std::memcmp(header.magic, kHSDRMagic, 4) != 0) {
        throw CompressionError("HSDR decompress: bad magic — not an HSDR stream");
    }
    if (header.version != kHSDRVersion) {
        throw CompressionError("HSDR decompress: unsupported version");
    }
    const uint32_t R = header.original_rows;
    const uint32_t C = header.original_cols;
    const uint16_t K = header.n_atoms;
    const uint8_t S = header.n_stages;
    const uint8_t k = header.active_bits_per_stage;
    const uint32_t n_tiles = header.n_tiles;
    const uint32_t tile_dim = static_cast<uint32_t>(header.tile_rows) * header.tile_cols;
    const uint32_t slots_per_tile = static_cast<uint32_t>(S) * k;
    const uint32_t R_full = (R / header.tile_rows) * header.tile_rows;
    const uint32_t C_full = (C / header.tile_cols) * header.tile_cols;

    if (static_cast<size_t>(R) * C * sizeof(float) != originalSize) {
        throw CompressionError("HSDR decompress: originalSize does not match shape");
    }

    std::vector<float> stage_scales(S);
    readBuffer(cursor, end, stage_scales.data(), S);

    std::vector<float> D(static_cast<size_t>(K) * tile_dim);
    readBuffer(cursor, end, D.data(), D.size());

    std::vector<uint16_t> packed(static_cast<size_t>(n_tiles) * slots_per_tile);
    readBuffer(cursor, end, packed.data(), packed.size());

    // Read edge strips (v2). 0 bytes if shape was aligned.
    std::vector<float> edge_buffer(header.edge_bytes_count / sizeof(float));
    if (header.edge_bytes_count > 0) {
        readBuffer(cursor, end, edge_buffer.data(), edge_buffer.size());
    }
    if (cursor != end) {
        std::cerr << "[HSDR] decompress: " << (end - cursor) << " trailing bytes ignored\n";
    }

    // Unpack indices+signs.
    std::vector<uint16_t> indices(packed.size());
    std::vector<int8_t>   signs(packed.size());
    for (size_t i = 0; i < packed.size(); ++i) {
        unpackIndexSign(packed[i], indices[i], signs[i]);
    }

    // V4b-decode the aligned (R_full × C_full) block.
    std::vector<float> recon_tiles(static_cast<size_t>(n_tiles) * tile_dim);
    decodeAllTiles(D.data(), K, tile_dim, indices.data(), signs.data(),
                   n_tiles, S, k, stage_scales.data(), recon_tiles.data());

    std::vector<std::byte> out(originalSize);
    float* out_f = reinterpret_cast<float*>(out.data());

    if (R_full == R && C_full == C) {
        // Fast path: aligned, no edge strips.
        reassembleTiles(recon_tiles.data(), R, C, header.tile_rows, header.tile_cols, out_f);
    } else {
        // Reassemble the aligned block into a contiguous buffer then splice into
        // the output, with the edge strips copied back into their original
        // positions (top-right + bottom).
        std::vector<float> aligned_block(static_cast<size_t>(R_full) * C_full);
        reassembleTiles(recon_tiles.data(), R_full, C_full,
                        header.tile_rows, header.tile_cols, aligned_block.data());
        // Top-left aligned region
        for (uint32_t r = 0; r < R_full; ++r) {
            std::memcpy(out_f + static_cast<size_t>(r) * C,
                        aligned_block.data() + static_cast<size_t>(r) * C_full,
                        sizeof(float) * C_full);
        }
        // Top-right edge: rows [0..R_full), cols [C_full..C)
        size_t edge_cursor = 0;
        for (uint32_t r = 0; r < R_full; ++r) {
            for (uint32_t c = C_full; c < C; ++c) {
                out_f[static_cast<size_t>(r) * C + c] = edge_buffer[edge_cursor++];
            }
        }
        // Bottom edge: rows [R_full..R), cols [0..C)
        for (uint32_t r = R_full; r < R; ++r) {
            for (uint32_t c = 0; c < C; ++c) {
                out_f[static_cast<size_t>(r) * C + c] = edge_buffer[edge_cursor++];
            }
        }
    }
    return out;
}

// --------------------------------------------------------------------------
// Shared-dictionary path implementations.
// --------------------------------------------------------------------------

HierarchicalSDRStrategy::SharedDictionary
HierarchicalSDRStrategy::fitSharedDictionary(
    const std::vector<float>& pooled_tiles, uint32_t n_total_tiles,
    const HierarchicalSDRConfig& cfg) const
{
    const uint32_t tile_dim = cfg.tileSize();
    if (pooled_tiles.size() != static_cast<size_t>(n_total_tiles) * tile_dim) {
        throw CompressionError("fitSharedDictionary: pooled tile size mismatch");
    }
    if (n_total_tiles < cfg.n_atoms) {
        throw CompressionError("fitSharedDictionary: pooled tile count below n_atoms");
    }

    SharedDictionary shared;
    shared.config = cfg;

    // Reuse the same fitter — it doesn't care that tiles came from many tensors.
    std::vector<uint16_t> indices_throwaway;
    std::vector<int8_t>   signs_throwaway;
    fitHierarchicalKSVD(pooled_tiles.data(), n_total_tiles, tile_dim, cfg,
                        shared.atoms, indices_throwaway, signs_throwaway,
                        shared.stage_scales);
    return shared;
}

std::vector<std::byte> HierarchicalSDRStrategy::compressWithExternalDictionary(
    const ModelSegment& segment, const SharedDictionary& dict) const
{
    if (!segment.isWeightTensor()) {
        throw CompressionError("compressWithExternalDictionary: weight tensor required");
    }
    if (!segment.tensor_metadata.has_value()
        || segment.tensor_metadata.value().dimensions.size() != 2) {
        throw CompressionError("compressWithExternalDictionary: 2-D tensor_metadata required");
    }
    const auto& dims = segment.tensor_metadata.value().dimensions;
    const uint32_t R = static_cast<uint32_t>(dims[0]);
    const uint32_t C = static_cast<uint32_t>(dims[1]);
    const auto& cfg = dict.config;
    if (R % cfg.tile_rows != 0 || C % cfg.tile_cols != 0) {
        throw CompressionError("compressWithExternalDictionary: shape not divisible by tile size");
    }
    const uint32_t tile_dim = cfg.tileSize();
    if (dict.atoms.size() != static_cast<size_t>(cfg.n_atoms) * tile_dim) {
        throw CompressionError("compressWithExternalDictionary: dictionary size doesn't match config");
    }

    // Dequantise to FP32 if needed (FP16/BF16/INT8 input is now supported).
    std::vector<float> fp32_view = dequantizeSegmentToFP32(segment);
    if (fp32_view.size() != static_cast<size_t>(R) * C) {
        throw CompressionError(
            "compressWithExternalDictionary: dequantised element count doesn't match shape");
    }

    uint32_t n_tiles = 0;
    std::vector<float> tiles = extractTiles(fp32_view.data(), R, C,
                                             cfg.tile_rows, cfg.tile_cols, n_tiles);

    const uint32_t slots_per_tile = static_cast<uint32_t>(cfg.n_stages) * cfg.active_bits_per_stage;
    std::vector<uint16_t> indices(static_cast<size_t>(n_tiles) * slots_per_tile);
    std::vector<int8_t>   signs(static_cast<size_t>(n_tiles) * slots_per_tile);
    encodeAllTilesAgainstDictionary(tiles.data(), n_tiles, tile_dim,
                                     dict.atoms.data(), cfg.n_atoms,
                                     cfg.n_stages, cfg.active_bits_per_stage,
                                     dict.stage_scales.data(),
                                     indices.data(), signs.data());

    HSDRCodesHeader header{};
    std::memcpy(header.magic, kHSDRCodesMagic, 4);
    header.version = kHSDRCodesVersion;
    header.original_rows = R;
    header.original_cols = C;
    header.n_tiles = n_tiles;
    header.tile_rows = cfg.tile_rows;
    header.tile_cols = cfg.tile_cols;
    header.n_atoms = cfg.n_atoms;
    header.n_stages = cfg.n_stages;
    header.active_bits_per_stage = cfg.active_bits_per_stage;
    header.stage_decay = cfg.stage_decay;

    std::vector<std::byte> out;
    out.reserve(sizeof(header) + indices.size() * sizeof(uint16_t));
    appendPOD(out, header);

    std::vector<uint16_t> packed(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        packed[i] = packIndexSign(indices[i], signs[i]);
    }
    appendBuffer(out, packed.data(), packed.size());
    return out;
}

namespace {
/// Read a codes-only header and packed codes, validate against the supplied
/// external dictionary. Returns (header, unpacked indices, unpacked signs).
struct CodesStreamView {
    HSDRCodesHeader header;
    std::vector<uint16_t> indices;
    std::vector<int8_t>   signs;
};

CodesStreamView parseCodesStream(
    const std::vector<std::byte>& codes_bytes,
    const HierarchicalSDRStrategy::SharedDictionary& dict)
{
    const std::byte* cursor = codes_bytes.data();
    const std::byte* end = cursor + codes_bytes.size();

    CodesStreamView view{};
    readPOD(cursor, end, view.header);
    if (std::memcmp(view.header.magic, kHSDRCodesMagic, 4) != 0) {
        throw CompressionError("HSDR shared: bad codes-stream magic");
    }
    if (view.header.version != kHSDRCodesVersion) {
        throw CompressionError("HSDR shared: unsupported codes-stream version");
    }
    if (view.header.tile_rows != dict.config.tile_rows ||
        view.header.tile_cols != dict.config.tile_cols ||
        view.header.n_atoms != dict.config.n_atoms ||
        view.header.n_stages != dict.config.n_stages ||
        view.header.active_bits_per_stage != dict.config.active_bits_per_stage) {
        throw CompressionError("HSDR shared: stream geometry doesn't match dictionary config");
    }

    const uint32_t slots_per_tile = static_cast<uint32_t>(view.header.n_stages)
                                     * view.header.active_bits_per_stage;
    const size_t total_slots = static_cast<size_t>(view.header.n_tiles) * slots_per_tile;
    std::vector<uint16_t> packed(total_slots);
    readBuffer(cursor, end, packed.data(), packed.size());

    view.indices.assign(total_slots, 0);
    view.signs.assign(total_slots, 0);
    for (size_t i = 0; i < total_slots; ++i) {
        unpackIndexSign(packed[i], view.indices[i], view.signs[i]);
    }
    return view;
}
} // anonymous namespace

std::vector<std::byte> HierarchicalSDRStrategy::decompressWithExternalDictionary(
    const std::vector<std::byte>& codes_bytes,
    const SharedDictionary& dict,
    size_t originalSize) const
{
    CodesStreamView view = parseCodesStream(codes_bytes, dict);
    const uint32_t R = view.header.original_rows;
    const uint32_t C = view.header.original_cols;
    if (static_cast<size_t>(R) * C * sizeof(float) != originalSize) {
        throw CompressionError("decompressWithExternalDictionary: originalSize mismatch");
    }
    const uint32_t tile_dim = static_cast<uint32_t>(view.header.tile_rows) * view.header.tile_cols;
    const uint32_t n_tiles = view.header.n_tiles;

    std::vector<float> recon_tiles(static_cast<size_t>(n_tiles) * tile_dim);
    decodeAllTiles(dict.atoms.data(), view.header.n_atoms, tile_dim,
                   view.indices.data(), view.signs.data(),
                   n_tiles, view.header.n_stages, view.header.active_bits_per_stage,
                   dict.stage_scales.data(), recon_tiles.data());

    std::vector<std::byte> out(originalSize);
    float* out_f = reinterpret_cast<float*>(out.data());
    reassembleTiles(recon_tiles.data(), R, C, view.header.tile_rows, view.header.tile_cols, out_f);
    return out;
}

std::vector<float> HierarchicalSDRStrategy::matmulWithExternalDictionary(
    const std::vector<std::byte>& codes_bytes,
    const SharedDictionary& dict,
    const float* x,
    size_t batch) const
{
    CodesStreamView view = parseCodesStream(codes_bytes, dict);
    if (view.header.tile_rows != 1) {
        throw CompressionError("matmulWithExternalDictionary: requires 1D row tiles");
    }
    if (view.header.tile_cols != view.header.original_cols) {
        throw CompressionError("matmulWithExternalDictionary: tile_cols must == original_cols");
    }
    const uint32_t R = view.header.original_rows;
    const uint32_t C = view.header.original_cols;
    const uint16_t K = view.header.n_atoms;
    const uint8_t  S = view.header.n_stages;
    const uint8_t  k = view.header.active_bits_per_stage;
    if (view.header.n_tiles != R) {
        throw CompressionError("matmulWithExternalDictionary: tile count must equal row count");
    }
    const uint32_t slots_per_tile = static_cast<uint32_t>(S) * k;

    // Precompute Dx[a, b] = dict.atoms[a] · x[:, b]
    std::vector<float> Dx(static_cast<size_t>(K) * batch, 0.0f);
    for (uint16_t a = 0; a < K; ++a) {
        const float* atom = dict.atoms.data() + static_cast<size_t>(a) * C;
        float* out_row = Dx.data() + static_cast<size_t>(a) * batch;
        for (uint32_t c = 0; c < C; ++c) {
            const float av = atom[c];
            const float* xrow = x + static_cast<size_t>(c) * batch;
            for (size_t b = 0; b < batch; ++b) out_row[b] += av * xrow[b];
        }
    }

    std::vector<float> Y(static_cast<size_t>(R) * batch, 0.0f);
    for (uint32_t r = 0; r < R; ++r) {
        float* y_row = Y.data() + static_cast<size_t>(r) * batch;
        for (uint8_t l = 0; l < S; ++l) {
            const float gamma = dict.stage_scales[l];
            for (uint8_t kk = 0; kk < k; ++kk) {
                const size_t slot = static_cast<size_t>(r) * slots_per_tile
                                    + static_cast<size_t>(l) * k + kk;
                const uint16_t atom_idx = view.indices[slot];
                const int8_t sign = view.signs[slot];
                if (sign == 0) continue;
                const float scale = gamma * static_cast<float>(sign);
                const float* dx_row = Dx.data() + static_cast<size_t>(atom_idx) * batch;
                for (size_t b = 0; b < batch; ++b) y_row[b] += scale * dx_row[b];
            }
        }
    }
    return Y;
}

// --------------------------------------------------------------------------
// Multi-pass pipeline: compressGroupedSegments
// --------------------------------------------------------------------------

size_t HierarchicalSDRStrategy::SharedDictArchive::totalBytes() const {
    size_t total = 0;
    for (const auto& d : dictionaries) {
        total += d.atoms.size() * sizeof(float);
        total += d.stage_scales.size() * sizeof(float);
        // Plus the small config struct itself, which lives in headers when serialised.
        total += sizeof(HierarchicalSDRConfig);
    }
    for (const auto& s : segments) total += s.codes_bytes.size();
    return total;
}

// --------------------------------------------------------------------------
// Disk format for SharedDictArchive
// --------------------------------------------------------------------------
//
// All multi-byte values are written in host byte order (little-endian on the
// platforms we target). The format is self-describing enough to be safely
// versioned, but is NOT the same as the main .sdr archive format yet —
// integration with AICompressor / SDRModelLoader comes in a follow-up round.
//
//   [Top-level header]
//      magic[4]            = 'H','S','D','A'
//      version (uint32)
//      num_dictionaries (uint32)
//      num_segments (uint32)
//
//   For each dictionary:
//      config (HierarchicalSDRConfig, raw POD, 12 bytes)
//      n_stage_scales (uint32)   -- redundant with config.n_stages, but explicit
//      stage_scales[n_stage_scales]
//      atoms_count (uint32)      -- equals n_atoms * tile_rows * tile_cols
//      atoms[atoms_count]        (float32)
//
//   For each segment entry:
//      name_length (uint32)
//      name[name_length]         (char, not null-terminated)
//      dict_index (uint32)
//      original_size (uint64)
//      original_type (uint8)
//      codes_length (uint32)
//      codes_bytes[codes_length]
//
// Trailing padding/footer: none. The reader knows totals from the top-level
// header.

namespace {

constexpr char kHSDAMagic[4] = {'H', 'S', 'D', 'A'};
constexpr uint32_t kHSDAVersion = 1;

template <typename T>
void writePOD(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw CompressionError("HSDA write: stream error");
}

template <typename T>
void writeArray(std::ostream& out, const T* data, size_t count) {
    if (count == 0) return;
    out.write(reinterpret_cast<const char*>(data), sizeof(T) * count);
    if (!out) throw CompressionError("HSDA write: stream error");
}

template <typename T>
void readPODStream(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(T))) {
        throw CompressionError("HSDA read: truncated stream");
    }
}

template <typename T>
void readArrayStream(std::istream& in, T* data, size_t count) {
    if (count == 0) return;
    in.read(reinterpret_cast<char*>(data), sizeof(T) * count);
    if (in.gcount() != static_cast<std::streamsize>(sizeof(T) * count)) {
        throw CompressionError("HSDA read: truncated array");
    }
}

}  // anonymous namespace

void HierarchicalSDRStrategy::SharedDictArchive::writeToStream(std::ostream& out) const {
    // Top-level header.
    out.write(kHSDAMagic, sizeof(kHSDAMagic));
    if (!out) throw CompressionError("HSDA write: cannot write magic");
    writePOD<uint32_t>(out, kHSDAVersion);
    writePOD<uint32_t>(out, static_cast<uint32_t>(dictionaries.size()));
    writePOD<uint32_t>(out, static_cast<uint32_t>(segments.size()));

    // Dictionaries.
    for (const auto& d : dictionaries) {
        writePOD<HierarchicalSDRConfig>(out, d.config);
        writePOD<uint32_t>(out, static_cast<uint32_t>(d.stage_scales.size()));
        writeArray<float>(out, d.stage_scales.data(), d.stage_scales.size());
        writePOD<uint32_t>(out, static_cast<uint32_t>(d.atoms.size()));
        writeArray<float>(out, d.atoms.data(), d.atoms.size());
    }

    // Segments.
    for (const auto& s : segments) {
        writePOD<uint32_t>(out, static_cast<uint32_t>(s.name.size()));
        if (!s.name.empty()) {
            out.write(s.name.data(), static_cast<std::streamsize>(s.name.size()));
            if (!out) throw CompressionError("HSDA write: name write failed");
        }
        writePOD<uint32_t>(out, static_cast<uint32_t>(s.dict_index));
        writePOD<uint64_t>(out, static_cast<uint64_t>(s.original_size));
        writePOD<uint8_t>(out, static_cast<uint8_t>(s.original_type));
        writePOD<uint32_t>(out, static_cast<uint32_t>(s.codes_bytes.size()));
        writeArray<std::byte>(out, s.codes_bytes.data(), s.codes_bytes.size());
    }
}

HierarchicalSDRStrategy::SharedDictArchive
HierarchicalSDRStrategy::SharedDictArchive::readFromStream(std::istream& in) {
    char magic[4];
    in.read(magic, sizeof(magic));
    if (in.gcount() != 4 || std::memcmp(magic, kHSDAMagic, 4) != 0) {
        throw CompressionError("HSDA read: bad magic — not an HSDA stream");
    }
    uint32_t version = 0, n_dicts = 0, n_segs = 0;
    readPODStream(in, version);
    if (version != kHSDAVersion) {
        throw CompressionError("HSDA read: unsupported version");
    }
    readPODStream(in, n_dicts);
    readPODStream(in, n_segs);

    SharedDictArchive archive;
    archive.dictionaries.resize(n_dicts);
    for (uint32_t i = 0; i < n_dicts; ++i) {
        auto& d = archive.dictionaries[i];
        readPODStream(in, d.config);
        uint32_t n_scales = 0;
        readPODStream(in, n_scales);
        d.stage_scales.resize(n_scales);
        readArrayStream(in, d.stage_scales.data(), d.stage_scales.size());
        uint32_t n_atoms_floats = 0;
        readPODStream(in, n_atoms_floats);
        d.atoms.resize(n_atoms_floats);
        readArrayStream(in, d.atoms.data(), d.atoms.size());
    }

    archive.segments.resize(n_segs);
    for (uint32_t i = 0; i < n_segs; ++i) {
        auto& s = archive.segments[i];
        uint32_t name_len = 0;
        readPODStream(in, name_len);
        s.name.resize(name_len);
        if (name_len > 0) {
            in.read(s.name.data(), name_len);
            if (in.gcount() != static_cast<std::streamsize>(name_len)) {
                throw CompressionError("HSDA read: name truncated");
            }
        }
        uint32_t dict_idx = 0;
        readPODStream(in, dict_idx);
        if (dict_idx >= archive.dictionaries.size()) {
            throw CompressionError("HSDA read: dict_index out of range");
        }
        s.dict_index = dict_idx;
        uint64_t orig_size = 0;
        readPODStream(in, orig_size);
        s.original_size = static_cast<size_t>(orig_size);
        uint8_t orig_type = 0;
        readPODStream(in, orig_type);
        s.original_type = static_cast<SegmentType>(orig_type);
        uint32_t codes_len = 0;
        readPODStream(in, codes_len);
        s.codes_bytes.resize(codes_len);
        readArrayStream(in, s.codes_bytes.data(), s.codes_bytes.size());
    }
    return archive;
}

void HierarchicalSDRStrategy::SharedDictArchive::writeToFile(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw CompressionError("HSDA writeToFile: cannot open " + path);
    }
    writeToStream(out);
}

HierarchicalSDRStrategy::SharedDictArchive
HierarchicalSDRStrategy::SharedDictArchive::readFromFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw CompressionError("HSDA readFromFile: cannot open " + path);
    }
    return readFromStream(in);
}

namespace {

/// Key used to group segments that can share one dictionary.
/// Two segments may share if their compression configs match in every
/// dimension that the shared dictionary serializes.
struct DictGroupKey {
    uint16_t tile_rows;
    uint16_t tile_cols;
    uint16_t n_atoms;
    uint8_t  n_stages;
    uint8_t  active_bits_per_stage;
    float    stage_decay;

    bool operator==(const DictGroupKey& o) const {
        return tile_rows == o.tile_rows && tile_cols == o.tile_cols
            && n_atoms == o.n_atoms && n_stages == o.n_stages
            && active_bits_per_stage == o.active_bits_per_stage
            && stage_decay == o.stage_decay;
    }
};

struct DictGroupKeyHash {
    size_t operator()(const DictGroupKey& k) const {
        size_t h = std::hash<uint32_t>{}(
            (uint32_t(k.tile_rows) << 16) | k.tile_cols);
        h ^= std::hash<uint32_t>{}(
            (uint32_t(k.n_atoms) << 16) | (uint32_t(k.n_stages) << 8) | k.active_bits_per_stage)
            + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(k.stage_decay)
            + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

DictGroupKey keyForConfig(const HierarchicalSDRConfig& cfg) {
    return DictGroupKey{cfg.tile_rows, cfg.tile_cols, cfg.n_atoms,
                        cfg.n_stages, cfg.active_bits_per_stage, cfg.stage_decay};
}

/// Validate that a segment is acceptable for HSDR (FP32 weight, 2-D, shape fits).
/// Returns true if OK; false otherwise (caller skips it).
/// Lightweight check of whether `dequantizeSegmentToFP32` can handle this segment.
/// Cheap (just inspects type + data_format), unlike actually running the dequant.
bool segmentDtypeSupported(const ModelSegment& seg) {
    if (seg.type == SegmentType::WEIGHTS_FP32
        || seg.type == SegmentType::WEIGHTS_FP16
        || seg.type == SegmentType::WEIGHTS_INT8) {
        return true;
    }
    const std::string fmt = lowerFormat(seg.data_format);
    return fmt == "f32" || fmt == "fp32" || fmt == "float32"
        || fmt == "f16" || fmt == "fp16" || fmt == "float16" || fmt == "half"
        || fmt == "bf16" || fmt == "bfloat16"
        || fmt == "i8" || fmt == "int8"
        || fmt == "q4_0" || fmt == "q8_0";
}

bool segmentSuitable(const ModelSegment& seg, const HierarchicalSDRConfig& cfg,
                     std::string* reason = nullptr) {
    if (!seg.isWeightTensor()) {
        if (reason) *reason = "not a weight tensor";
        return false;
    }
    if (!seg.tensor_metadata.has_value() ||
        seg.tensor_metadata.value().dimensions.size() != 2) {
        if (reason) *reason = "missing 2-D tensor_metadata";
        return false;
    }
    if (!segmentDtypeSupported(seg)) {
        if (reason) *reason = "unsupported dtype (need f32/f16/bf16/i8)";
        return false;
    }
    const auto& dims = seg.tensor_metadata.value().dimensions;
    const size_t R = dims[0];
    const size_t C = dims[1];
    // Each supported dtype has a fixed byte-per-element width. The dequant
    // helper does precise validation; here we just want to catch obviously
    // mismatched shapes without doing the full conversion.
    size_t expected_bytes = 0;
    if (seg.type == SegmentType::WEIGHTS_FP32) expected_bytes = R * C * 4;
    else if (seg.type == SegmentType::WEIGHTS_FP16) expected_bytes = R * C * 2;
    else if (seg.type == SegmentType::WEIGHTS_INT8) expected_bytes = R * C;
    else {
        const std::string fmt = lowerFormat(seg.data_format);
        if (fmt == "f32" || fmt == "fp32" || fmt == "float32") expected_bytes = R * C * 4;
        else if (fmt == "f16" || fmt == "fp16" || fmt == "float16" || fmt == "half"
                 || fmt == "bf16" || fmt == "bfloat16") expected_bytes = R * C * 2;
        else if (fmt == "i8" || fmt == "int8") expected_bytes = R * C;
    }
    if (expected_bytes != 0 && expected_bytes != seg.data.size()) {
        if (reason) *reason = "shape/data size mismatch";
        return false;
    }
    // For shared-dict path we require exact divisibility (edge handling lives in
    // the per-tensor compress() path, not the shared one).
    if (R % cfg.tile_rows != 0 || C % cfg.tile_cols != 0) {
        if (reason) *reason = "shape not divisible by tile size";
        return false;
    }
    return true;
}

}  // anonymous namespace

HierarchicalSDRStrategy::SharedDictArchive
HierarchicalSDRStrategy::compressGroupedSegments(
    const std::vector<ModelSegment>& segments,
    std::vector<std::string>* out_skipped_names) const
{
    // 1. Bucket segments by their effective compression config.
    std::unordered_map<DictGroupKey, std::vector<size_t>, DictGroupKeyHash> buckets;
    std::vector<HierarchicalSDRConfig> per_segment_cfg(segments.size());

    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& seg = segments[i];
        const HierarchicalSDRConfig cfg = configFor(seg);
        per_segment_cfg[i] = cfg;

        if (protection_ && protection_(seg)) {
            if (out_skipped_names) out_skipped_names->push_back(seg.name);
            continue;
        }
        std::string why;
        if (!segmentSuitable(seg, cfg, &why)) {
            if (out_skipped_names) out_skipped_names->push_back(seg.name);
            continue;
        }
        buckets[keyForConfig(cfg)].push_back(i);
    }

    SharedDictArchive archive;
    archive.dictionaries.reserve(buckets.size());
    archive.segments.reserve(segments.size());

    // 2. For each bucket: pool tiles, fit a shared dictionary, encode each segment.
    for (auto& kv : buckets) {
        const std::vector<size_t>& seg_indices = kv.second;
        if (seg_indices.empty()) continue;
        const HierarchicalSDRConfig& cfg = per_segment_cfg[seg_indices.front()];

        // 2a. Pool tiles. Per-segment tile extraction.
        std::vector<float> pooled;
        uint32_t pooled_count = 0;
        const uint32_t tile_dim = cfg.tileSize();

        // First pass: total size to reserve.
        size_t total_tiles_floats = 0;
        for (size_t idx : seg_indices) {
            const auto& s = segments[idx];
            const auto& dims = s.tensor_metadata.value().dimensions;
            const uint32_t R = static_cast<uint32_t>(dims[0]);
            const uint32_t C = static_cast<uint32_t>(dims[1]);
            const uint32_t nt = (R / cfg.tile_rows) * (C / cfg.tile_cols);
            total_tiles_floats += static_cast<size_t>(nt) * tile_dim;
            pooled_count += nt;
        }
        pooled.reserve(total_tiles_floats);

        // Second pass: actually extract. Dequantise each segment to FP32 (no-op
        // for already-FP32) before tile extraction so the pool is dtype-uniform.
        for (size_t idx : seg_indices) {
            const auto& s = segments[idx];
            const auto& dims = s.tensor_metadata.value().dimensions;
            const uint32_t R = static_cast<uint32_t>(dims[0]);
            const uint32_t C = static_cast<uint32_t>(dims[1]);
            std::vector<float> fp32_view = dequantizeSegmentToFP32(s);
            if (fp32_view.size() != static_cast<size_t>(R) * C) {
                throw CompressionError("compressGroupedSegments: dequant size mismatch on '"
                                       + s.name + "'");
            }
            uint32_t nt_check = 0;
            std::vector<float> tiles = extractTiles(fp32_view.data(), R, C,
                                                     cfg.tile_rows, cfg.tile_cols, nt_check);
            pooled.insert(pooled.end(), tiles.begin(), tiles.end());
        }

        if (pooled_count < cfg.n_atoms) {
            // Pool too small to fit the requested dictionary — skip the whole bucket
            // and mark every member as skipped.
            for (size_t idx : seg_indices) {
                if (out_skipped_names) out_skipped_names->push_back(segments[idx].name);
            }
            continue;
        }

        // 2b. Fit shared dictionary on the pool.
        SharedDictionary shared = fitSharedDictionary(pooled, pooled_count, cfg);
        const size_t dict_index = archive.dictionaries.size();
        archive.dictionaries.push_back(std::move(shared));

        // 2c. Encode each segment in the bucket against the new shared dict.
        const SharedDictionary& dict_ref = archive.dictionaries[dict_index];
        for (size_t idx : seg_indices) {
            const auto& s = segments[idx];
            SharedDictArchive::SegmentEntry entry;
            entry.name = s.name;
            entry.dict_index = dict_index;
            entry.codes_bytes = compressWithExternalDictionary(s, dict_ref);
            entry.original_size = s.data.size();
            entry.original_type = s.type;
            archive.segments.push_back(std::move(entry));
        }
    }

    return archive;
}

// --------------------------------------------------------------------------
// HSDAReader implementation
// --------------------------------------------------------------------------

HSDAReader::HSDAReader(HierarchicalSDRStrategy::SharedDictArchive archive)
    : archive_(std::move(archive))
{
    name_index_.reserve(archive_.segments.size());
    for (size_t i = 0; i < archive_.segments.size(); ++i) {
        name_index_[archive_.segments[i].name] = i;
    }
}

HSDAReader HSDAReader::fromFile(const std::string& path) {
    return HSDAReader(HierarchicalSDRStrategy::SharedDictArchive::readFromFile(path));
}

bool HSDAReader::hasSegment(const std::string& name) const {
    return name_index_.find(name) != name_index_.end();
}

std::vector<std::byte> HSDAReader::decompress(const std::string& name) const {
    auto it = name_index_.find(name);
    if (it == name_index_.end()) {
        throw std::runtime_error("HSDAReader::decompress: segment not found: " + name);
    }
    const auto& entry = archive_.segments[it->second];
    if (entry.dict_index >= archive_.dictionaries.size()) {
        throw std::runtime_error("HSDAReader::decompress: dangling dict_index for '" + name + "'");
    }
    const auto& dict = archive_.dictionaries[entry.dict_index];
    return strat_.decompressWithExternalDictionary(entry.codes_bytes, dict, entry.original_size);
}

std::vector<float> HSDAReader::matmul(const std::string& name,
                                      const float* x,
                                      size_t batch) const
{
    auto it = name_index_.find(name);
    if (it == name_index_.end()) {
        throw std::runtime_error("HSDAReader::matmul: segment not found: " + name);
    }
    const auto& entry = archive_.segments[it->second];
    if (entry.dict_index >= archive_.dictionaries.size()) {
        throw std::runtime_error("HSDAReader::matmul: dangling dict_index for '" + name + "'");
    }
    const auto& dict = archive_.dictionaries[entry.dict_index];
    return strat_.matmulWithExternalDictionary(entry.codes_bytes, dict, x, batch);
}

// --------------------------------------------------------------------------
// ProtectionPolicies — composable predicates for hybrid FP16 orchestration.
// --------------------------------------------------------------------------
namespace ProtectionPolicies {

HierarchicalSDRStrategy::ProtectionPredicate boundaryMLPs(
    size_t n_boundary, size_t total_layers)
{
    return [n_boundary, total_layers](const ModelSegment& seg) -> bool {
        // Identify MLP segments. Trust SegmentType first; fall back to name.
        bool is_mlp = (seg.type == SegmentType::FEED_FORWARD_WEIGHTS);
        if (!is_mlp) {
            const std::string& n = seg.name;
            is_mlp = (n.find("mlp") != std::string::npos
                      || n.find("ffn") != std::string::npos
                      || n.find("feed_forward") != std::string::npos);
        }
        if (!is_mlp) return false;

        const size_t depth = seg.layer_index;
        const bool in_early = (depth < n_boundary);
        const bool in_late  = (total_layers > n_boundary && depth >= total_layers - n_boundary);
        return in_early || in_late;
    };
}

HierarchicalSDRStrategy::ProtectionPredicate byName(std::unordered_set<std::string> names) {
    return [names = std::move(names)](const ModelSegment& seg) -> bool {
        return names.count(seg.name) > 0;
    };
}

}  // namespace ProtectionPolicies

} // namespace CortexAICompression
