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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
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
constexpr uint32_t kHSDRVersion = 1;

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
};
#pragma pack(pop)
static_assert(sizeof(HSDRHeader) == 4 + 4 + 4 + 4 + 4 + 2 + 2 + 2 + 1 + 1 + 4,
              "HSDRHeader size mismatch — packing broken");

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

    // Only FP32 input. Other dtypes fall through to the lower-priority strategy.
    bool is_fp32 = (segment.type == SegmentType::WEIGHTS_FP32);
    if (!is_fp32) {
        std::string fmt;
        fmt.reserve(segment.data_format.size());
        for (char c : segment.data_format) {
            fmt.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (fmt == "f32" || fmt == "float32" || fmt == "fp32") is_fp32 = true;
    }
    if (!is_fp32) {
        throw CompressionError("HierarchicalSDRStrategy: only FP32 weights supported");
    }

    if (segment.data.size() % sizeof(float) != 0) {
        throw CompressionError("HSDR: byte size not a multiple of sizeof(float)");
    }
    const size_t num_elements = segment.data.size() / sizeof(float);

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
    if (R % cfg.tile_rows != 0 || C % cfg.tile_cols != 0) {
        throw CompressionError("HSDR: tensor shape not divisible by tile size "
                               "(edge handling not implemented yet)");
    }
    const uint32_t n_row_tiles = R / cfg.tile_rows;
    const uint32_t n_col_tiles = C / cfg.tile_cols;
    const uint32_t n_tiles = n_row_tiles * n_col_tiles;
    const uint32_t tile_dim = cfg.tileSize();
    if (n_tiles < cfg.n_atoms) {
        throw CompressionError("HSDR: not enough tiles to fit the requested dictionary size");
    }

    // Extract tiles into a contiguous (n_tiles, tile_dim) array.
    const float* weight = reinterpret_cast<const float*>(segment.data.data());
    uint32_t n_tiles_check = 0;
    std::vector<float> tiles = extractTiles(weight, R, C, cfg.tile_rows, cfg.tile_cols, n_tiles_check);
    if (n_tiles_check != n_tiles) {
        throw CompressionError("HSDR: internal tile-count mismatch");
    }

    // Fit V4b: dictionary + per-tile (indices, signs).
    std::vector<float> D, stage_scales;
    std::vector<uint16_t> indices;
    std::vector<int8_t>   signs;
    fitHierarchicalKSVD(tiles.data(), n_tiles, tile_dim, cfg, D, indices, signs, stage_scales);

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

    const uint32_t slots_per_tile = static_cast<uint32_t>(cfg.n_stages) * cfg.active_bits_per_stage;
    const size_t packed_codes = static_cast<size_t>(n_tiles) * slots_per_tile;

    std::vector<std::byte> out;
    out.reserve(sizeof(HSDRHeader)
                + sizeof(float) * stage_scales.size()
                + sizeof(float) * D.size()
                + sizeof(uint16_t) * packed_codes);

    appendPOD(out, header);
    appendBuffer(out, stage_scales.data(), stage_scales.size());
    appendBuffer(out, D.data(), D.size());

    // Pack each (index, sign) pair into a single uint16.
    std::vector<uint16_t> packed(packed_codes);
    for (size_t i = 0; i < packed_codes; ++i) {
        packed[i] = packIndexSign(indices[i], signs[i]);
    }
    appendBuffer(out, packed.data(), packed.size());

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

    if (static_cast<size_t>(R) * C * sizeof(float) != originalSize) {
        throw CompressionError("HSDR decompress: originalSize does not match shape");
    }

    std::vector<float> stage_scales(S);
    readBuffer(cursor, end, stage_scales.data(), S);

    std::vector<float> D(static_cast<size_t>(K) * tile_dim);
    readBuffer(cursor, end, D.data(), D.size());

    std::vector<uint16_t> packed(static_cast<size_t>(n_tiles) * slots_per_tile);
    readBuffer(cursor, end, packed.data(), packed.size());

    if (cursor != end) {
        // Not fatal — just diagnostic.
        std::cerr << "[HSDR] decompress: " << (end - cursor) << " trailing bytes ignored\n";
    }

    // Unpack indices+signs.
    std::vector<uint16_t> indices(packed.size());
    std::vector<int8_t>   signs(packed.size());
    for (size_t i = 0; i < packed.size(); ++i) {
        unpackIndexSign(packed[i], indices[i], signs[i]);
    }

    // Decode all tiles, then reassemble into the original matrix.
    std::vector<float> recon_tiles(static_cast<size_t>(n_tiles) * tile_dim);
    decodeAllTiles(D.data(), K, tile_dim, indices.data(), signs.data(),
                   n_tiles, S, k, stage_scales.data(), recon_tiles.data());

    std::vector<std::byte> out(originalSize);
    float* out_f = reinterpret_cast<float*>(out.data());
    reassembleTiles(recon_tiles.data(), R, C, header.tile_rows, header.tile_cols, out_f);
    return out;
}

} // namespace CortexAICompression
