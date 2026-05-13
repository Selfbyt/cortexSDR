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

#if defined(__AVX2__) || defined(CORTEXSDR_SIMD_AVX2)
#include <immintrin.h>
#define CORTEXSDR_HSDR_HAVE_AVX2 1
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace CortexAICompression {

// Forward declaration so member functions defined before the helper itself
// can still call it. Definition (inline) lives further down in this file.
std::pair<size_t, size_t> effective2DShape_impl(const std::vector<size_t>& dims);

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
// --------------------------------------------------------------------------
// SIMD helpers — AVX2 if available, scalar fallback otherwise. These two
// kernels are the hottest in the whole encoder; everything else is fanout
// over tiles + atoms.
// --------------------------------------------------------------------------
static inline float dot_product_simd(const float* a, const float* b, uint32_t n) {
#ifdef CORTEXSDR_HSDR_HAVE_AVX2
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t c = 0;
    // Unroll x2 for IPC. Each iteration consumes 16 floats.
    for (; c + 16 <= n; c += 16) {
        __m256 av0 = _mm256_loadu_ps(a + c);
        __m256 bv0 = _mm256_loadu_ps(b + c);
        __m256 av1 = _mm256_loadu_ps(a + c + 8);
        __m256 bv1 = _mm256_loadu_ps(b + c + 8);
        acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
        acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
    }
    for (; c + 8 <= n; c += 8) {
        __m256 av = _mm256_loadu_ps(a + c);
        __m256 bv = _mm256_loadu_ps(b + c);
        acc0 = _mm256_fmadd_ps(av, bv, acc0);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    // Horizontal sum.
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float s = _mm_cvtss_f32(sum);
    for (; c < n; ++c) s += a[c] * b[c];
    return s;
#else
    float s = 0.0f;
    for (uint32_t c = 0; c < n; ++c) s += a[c] * b[c];
    return s;
#endif
}

static inline void axpy_subtract_simd(float* r, const float* atom, float scale, uint32_t n) {
#ifdef CORTEXSDR_HSDR_HAVE_AVX2
    const __m256 vs = _mm256_set1_ps(scale);
    uint32_t c = 0;
    for (; c + 16 <= n; c += 16) {
        __m256 rv0 = _mm256_loadu_ps(r + c);
        __m256 av0 = _mm256_loadu_ps(atom + c);
        __m256 rv1 = _mm256_loadu_ps(r + c + 8);
        __m256 av1 = _mm256_loadu_ps(atom + c + 8);
        rv0 = _mm256_fnmadd_ps(av0, vs, rv0);  // r -= atom * scale
        rv1 = _mm256_fnmadd_ps(av1, vs, rv1);
        _mm256_storeu_ps(r + c, rv0);
        _mm256_storeu_ps(r + c + 8, rv1);
    }
    for (; c + 8 <= n; c += 8) {
        __m256 rv = _mm256_loadu_ps(r + c);
        __m256 av = _mm256_loadu_ps(atom + c);
        rv = _mm256_fnmadd_ps(av, vs, rv);
        _mm256_storeu_ps(r + c, rv);
    }
    for (; c < n; ++c) r[c] -= scale * atom[c];
#else
    for (uint32_t c = 0; c < n; ++c) r[c] -= scale * atom[c];
#endif
}

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
            // proj[j] = <D[j], r>. SIMD inner loop; outer loop is small enough
            // to leave serial (we want OpenMP parallelism at the tile level).
            for (uint16_t j = 0; j < n_atoms; ++j) {
                const float* atom = dictionary + static_cast<size_t>(j) * C;
                const float s = dot_product_simd(atom, r.data(), C);
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
            axpy_subtract_simd(r.data(), atom, scale, C);
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
    const char* dbg = std::getenv("CORTEXSDR_HSDR_DEBUG");
    const bool log_step = dbg && std::string_view(dbg) == std::string_view("1");
    const uint32_t slots_per_tile = static_cast<uint32_t>(n_stages) * k_per_stage;

    // Full reconstruction for each tile, then residual = tile - recon.
    // Tiles are independent — parallelise across cores.
    std::vector<float> full_recon(static_cast<size_t>(n_tiles) * C, 0.0f);
    const int32_t nt_for_recon = static_cast<int32_t>(n_tiles);
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < nt_for_recon; ++i) {
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
    const int64_t rsz = static_cast<int64_t>(residual.size());
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rsz; ++i) {
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

    // Accumulators are float64 to match the Python reference: with wide tiles
    // (e.g. 3584/18944-element row tiles in LLM weights) the per-atom user
    // sum runs over tens of thousands of contributions; float32 accumulation
    // catastrophically loses precision. Each parallel atom-update iteration
    // declares its own thread-local `correction` buffer below.

    // Snapshot pre-update atom + residual magnitudes for debugging.
    if (log_step) {
        double rmax = 0.0, rsum = 0.0;
        size_t rbad = 0;
        for (size_t i = 0; i < residual.size(); ++i) {
            const float v = residual[i];
            if (!std::isfinite(v)) { ++rbad; continue; }
            rmax = std::max(rmax, std::fabs(static_cast<double>(v)));
            rsum += std::fabs(static_cast<double>(v));
        }
        double amax = 0.0, asum = 0.0;
        size_t abad = 0;
        for (size_t i = 0; i < static_cast<size_t>(n_atoms) * C; ++i) {
            const float v = D[i];
            if (!std::isfinite(v)) { ++abad; continue; }
            amax = std::max(amax, std::fabs(static_cast<double>(v)));
            asum += std::fabs(static_cast<double>(v));
        }
        const size_t total_a = static_cast<size_t>(n_atoms) * C;
        std::cerr << "  [step] pre-update  residual: max=" << rmax << " mean="
                  << (rsum / residual.size()) << " bad=" << rbad
                  << "  D: max=" << amax << " mean=" << (asum / total_a)
                  << " bad=" << abad << "\n";
    }

    // Compute a tile-magnitude cap used to detect/recover from runaway atoms.
    // Binary-MP with fixed γ_l + greedy ±1 codes has no inherent step-size
    // control: if an atom drifts past the tile magnitude range, the encode
    // step's residual update overshoots, the next atom-update overshoots
    // further, and within a handful of iterations every atom is 1e+07 or NaN.
    // Capping at a multiple of the mean tile norm keeps the loop stable.
    double tile_norm_sum = 0.0;
    const int32_t nt_norm = static_cast<int32_t>(n_tiles);
    #pragma omp parallel for reduction(+:tile_norm_sum) schedule(static)
    for (int32_t i = 0; i < nt_norm; ++i) {
        const float* t = tiles + static_cast<size_t>(i) * C;
        double sq = 0.0;
        for (uint32_t c = 0; c < C; ++c) sq += static_cast<double>(t[c]) * t[c];
        tile_norm_sum += std::sqrt(sq);
    }
    const double tile_norm_mean = tile_norm_sum / std::max<uint32_t>(1u, n_tiles);
    // Target atom magnitude. With ±1 binary signs + fixed stage scales, the
    // sum of slots_per_tile ±atoms reconstructs each tile, so each atom's
    // contribution should sit around tile_norm / √slots — same scale as
    // init_scale at line ~333 in fitHierarchicalKSVD. We cap to this scale
    // rather than to several × tile_norm_mean; tighter cap = stable encode
    // step (no MP overshoot regardless of input distribution).
    const double atom_target_norm = std::max(1e-6, tile_norm_mean
                                                    / std::sqrt(static_cast<double>(slots_per_tile)));
    const double atom_norm_cap = 1.5 * atom_target_norm;

    // Damped step. K-SVD's textbook update is the SVD of the residual
    // restricted to users; the ±1-binary variant we use here doesn't satisfy
    // SVD's optimality bounds, so a step size < 1 prevents the LMS-style
    // increment from overshooting on tiles with sparse spikes.
    constexpr double kStep = 0.5;

    // Per-atom seed for thread-safe re-seeding. We mix the iteration's rng
    // state once, then derive a per-atom seed; this keeps the parallel loop
    // deterministic and free of shared rng access.
    const uint64_t base_seed = static_cast<uint64_t>(rng()) ^ 0xA0B1C2D3u;

    const int32_t na = static_cast<int32_t>(n_atoms);
    #pragma omp parallel for schedule(static)
    for (int32_t j = 0; j < na; ++j) {
        std::vector<double> correction(C, 0.0);  // thread-local
        const auto& u = users[j];
        float* atom = D + static_cast<size_t>(j) * C;
        if (u.empty()) {
            // Dead atom — re-seed with a random tile (rescaled to target).
            std::mt19937 local_rng(static_cast<uint32_t>(base_seed ^ (j * 0x9E3779B1u)));
            std::uniform_int_distribution<uint32_t> picker(0, n_tiles - 1);
            const uint32_t rid = picker(local_rng);
            const float* src = tiles + static_cast<size_t>(rid) * C;
            double sq = 0.0;
            for (uint32_t c = 0; c < C; ++c) sq += static_cast<double>(src[c]) * src[c];
            const double n = std::sqrt(std::max(sq, 1e-30));
            const double s = atom_target_norm / n;
            for (uint32_t c = 0; c < C; ++c) atom[c] = static_cast<float>(src[c] * s);
            continue;
        }

        double weight_sum = 0.0;
        for (uint32_t enc : u) {
            const uint32_t i = enc >> 16;
            const uint16_t loc = static_cast<uint16_t>(enc & 0xFFFFu);
            const uint8_t l = static_cast<uint8_t>(loc / k_per_stage);
            const size_t slot = static_cast<size_t>(i) * slots_per_tile + loc;
            const double gamma = static_cast<double>(stage_scales[l]);
            const double sg = static_cast<double>(signs[slot]) * gamma;
            const float* res_i = residual.data() + static_cast<size_t>(i) * C;
            for (uint32_t c = 0; c < C; ++c) {
                correction[c] += sg * static_cast<double>(res_i[c]);
            }
            weight_sum += gamma * gamma;
        }
        if (weight_sum <= 1e-12) continue;
        const double inv_w = kStep / weight_sum;

        // Apply correction and compute new magnitude. NaN/Inf-safe at TWO
        // levels: (a) the double-precision update can overflow → NaN/Inf;
        // (b) even if the double is finite, |v| may exceed FLT_MAX (3.4e+38)
        // so static_cast<float>(v) silently stores +Inf. We must check the
        // float we actually wrote, not the pre-cast double, or renormalize
        // can't recover the atom (Inf * anything = Inf).
        double new_sq = 0.0;
        bool any_nonfinite = false;
        for (uint32_t c = 0; c < C; ++c) {
            const double v = static_cast<double>(atom[c]) + correction[c] * inv_w;
            const float vf = static_cast<float>(v);
            if (!std::isfinite(v) || !std::isfinite(vf)) {
                any_nonfinite = true;
                break;
            }
            atom[c] = vf;
            new_sq += static_cast<double>(vf) * static_cast<double>(vf);
        }

        if (any_nonfinite || !std::isfinite(new_sq)) {
            // Re-seed from a random tile (rescaled to target). Thread-safe
            // per-atom rng.
            std::mt19937 local_rng(static_cast<uint32_t>(base_seed ^ (j * 0xB7E15163u)));
            std::uniform_int_distribution<uint32_t> picker(0, n_tiles - 1);
            const uint32_t rid = picker(local_rng);
            const float* src = tiles + static_cast<size_t>(rid) * C;
            double sq = 0.0;
            for (uint32_t c = 0; c < C; ++c) sq += static_cast<double>(src[c]) * src[c];
            const double nrm = std::sqrt(std::max(sq, 1e-30));
            const double s = atom_target_norm / nrm;
            for (uint32_t c = 0; c < C; ++c) atom[c] = static_cast<float>(src[c] * s);
            continue;
        }

        const double new_norm = std::sqrt(new_sq);
        // Renormalize if atom drifted past the cap. Preserves the direction
        // K-SVD has been refining instead of throwing it away (re-seed loses
        // progress; renormalize keeps it). This is the key change that makes
        // the loop converge on any input distribution: binding the magnitude
        // breaks the encode-overshoot → bigger-residual → bigger-atom loop.
        if (new_norm > atom_norm_cap) {
            const double s = atom_target_norm / new_norm;
            for (uint32_t c = 0; c < C; ++c) {
                atom[c] = static_cast<float>(static_cast<double>(atom[c]) * s);
            }
        }
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

    // Optional per-iter logging — enable by setting CORTEXSDR_HSDR_DEBUG=1
    // in the environment. Prints atom-magnitude / NaN stats so we can see
    // divergence (or convergence) in real time without ad-hoc instrumentation.
    const char* dbg = std::getenv("CORTEXSDR_HSDR_DEBUG");
    const bool log_iter = dbg && std::string_view(dbg) == std::string_view("1");

    // Always run one initial encode against the random-tile-init dictionary,
    // then optionally refine via (atom update + re-encode) iterations.
    // ksvd_iters=0 means "skip the K-SVD refinement entirely" — useful when the
    // refinement cost dominates and we want a fast-compress-only mode.
    auto encode_all = [&]() {
        // Tiles are encoded independently — parallelise across cores. With
        // OpenMP this is ~Ncores× speedup. The thread-local scratch inside
        // hierarchicalBinaryCode (residual, projection scores) keeps each
        // iteration self-contained, so no shared state to guard.
        const int32_t nt = static_cast<int32_t>(n_tiles);
        #pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < nt; ++i) {
            const float* tile = tiles + static_cast<size_t>(i) * C;
            uint16_t* idx_dst = indices_out.data() + static_cast<size_t>(i) * slots_per_tile;
            int8_t* sgn_dst = signs_out.data() + static_cast<size_t>(i) * slots_per_tile;
            hierarchicalBinaryCode(tile, D_out.data(), K, C, S, k,
                                   stage_scales_out.data(), idx_dst, sgn_dst);
        }
    };
    encode_all();

    for (uint8_t iter = 0; iter < cfg.ksvd_iters; ++iter) {
        atomUpdateStep(tiles, n_tiles, C, D_out.data(), K,
                       indices_out.data(), signs_out.data(),
                       S, k, stage_scales_out.data(), rng);
        encode_all();

        if (log_iter) {
            double max_abs = 0.0, sum_abs = 0.0;
            size_t nan_n = 0;
            for (float v : D_out) {
                if (std::isnan(v) || std::isinf(v)) { ++nan_n; continue; }
                const double a = std::fabs(static_cast<double>(v));
                max_abs = std::max(max_abs, a);
                sum_abs += a;
            }
            std::cerr << "[hsdr-debug] fit C=" << C << " K=" << K
                      << " iter " << static_cast<int>(iter)
                      << "/" << static_cast<int>(cfg.ksvd_iters)
                      << "  max|a|=" << max_abs
                      << "  mean|a|=" << (sum_abs / D_out.size())
                      << "  non_finite=" << nan_n << "\n";
        }
    }

    // Final sanitization. atomUpdateStep already re-seeds on NaN, but if any
    // non-finite value somehow survives (e.g. via an encoding path we haven't
    // instrumented), zero it out rather than ship a poisoned dictionary that
    // produces NaN matmul outputs downstream. This is a safety net, not an
    // expected code path — if it ever fires, audit atomUpdateStep.
    size_t sanitized = 0;
    for (size_t i = 0; i < D_out.size(); ++i) {
        if (!std::isfinite(D_out[i])) {
            D_out[i] = 0.0f;
            ++sanitized;
        }
    }
    (void)sanitized;  // hook for future logging if needed
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

    // -----------------------------------------------------------------------
    // GGUF k-quants: super-block formats. QK_K = 256 elements per super-block.
    // Layouts implemented from the public GGUF spec
    // (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). No
    // llama.cpp source is imported. Bit layouts are sensitive — see the
    // hand-computed test cases in test_hsdr_kquants for validation.
    // -----------------------------------------------------------------------

    // Helper: unpack the 12-byte packed scales/mins block used by Q4_K and Q5_K
    // into 8 6-bit scales and 8 6-bit mins.
    auto unpack_k_scales_mins = [](const uint8_t* scales12, uint8_t* sc_out, uint8_t* m_out) {
        for (int is = 0; is < 4; ++is) {
            sc_out[is] = scales12[is] & 0x3F;
            m_out[is]  = scales12[is + 4] & 0x3F;
        }
        for (int is = 4; is < 8; ++is) {
            // High 4 bits of sc/m come from the top 2 bits of scales[is-4]/scales[is].
            sc_out[is] = (scales12[is + 4] & 0x0F)
                         | ((scales12[is - 4] >> 6) << 4);
            m_out[is]  = (scales12[is + 4] >> 4)
                         | ((scales12[is]     >> 6) << 4);
        }
    };

    if (fmt == "q4_k") {
        constexpr size_t QK_K = 256;
        constexpr size_t BLOCK_BYTES = 2 + 2 + 12 + QK_K / 2;  // 144
        static_assert(BLOCK_BYTES == 144, "Q4_K block size");
        if (segment.data.size() % BLOCK_BYTES != 0) {
            throw CompressionError("HSDR dequant: Q4_K byte size not a multiple of 144");
        }
        const size_t n_blocks = segment.data.size() / BLOCK_BYTES;
        std::vector<float> out(n_blocks * QK_K);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(segment.data.data());
        uint8_t sc[8], mn[8];
        for (size_t b = 0; b < n_blocks; ++b) {
            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits,    p,     2);
            std::memcpy(&dmin_bits, p + 2, 2);
            const float d    = fp16_to_fp32(d_bits);
            const float dmin = fp16_to_fp32(dmin_bits);
            const uint8_t* scales12 = p + 4;
            const uint8_t* qs       = p + 16;  // 4 + 12
            unpack_k_scales_mins(scales12, sc, mn);
            for (int is = 0; is < 8; ++is) {
                const float ds = d * static_cast<float>(sc[is]);
                const float dm = dmin * static_cast<float>(mn[is]);
                for (int j = 0; j < 32; ++j) {
                    const int idx = is * 32 + j;
                    const uint8_t byte = qs[idx / 2];
                    const uint8_t q = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
                    out[b * QK_K + idx] = static_cast<float>(q) * ds - dm;
                }
            }
            p += BLOCK_BYTES;
        }
        return out;
    }

    if (fmt == "q5_k") {
        constexpr size_t QK_K = 256;
        constexpr size_t BLOCK_BYTES = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;  // 176
        static_assert(BLOCK_BYTES == 176, "Q5_K block size");
        if (segment.data.size() % BLOCK_BYTES != 0) {
            throw CompressionError("HSDR dequant: Q5_K byte size not a multiple of 176");
        }
        const size_t n_blocks = segment.data.size() / BLOCK_BYTES;
        std::vector<float> out(n_blocks * QK_K);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(segment.data.data());
        uint8_t sc[8], mn[8];
        for (size_t b = 0; b < n_blocks; ++b) {
            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits,    p,     2);
            std::memcpy(&dmin_bits, p + 2, 2);
            const float d    = fp16_to_fp32(d_bits);
            const float dmin = fp16_to_fp32(dmin_bits);
            const uint8_t* scales12 = p + 4;
            const uint8_t* qh       = p + 16;          // 32 bytes — 1 high bit per element
            const uint8_t* qs       = qh + QK_K / 8;   // 128 bytes — 4 low bits per element
            unpack_k_scales_mins(scales12, sc, mn);
            for (int is = 0; is < 8; ++is) {
                const float ds = d * static_cast<float>(sc[is]);
                const float dm = dmin * static_cast<float>(mn[is]);
                for (int j = 0; j < 32; ++j) {
                    const int idx = is * 32 + j;
                    const uint8_t byte = qs[idx / 2];
                    const uint8_t q_lo = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
                    const uint8_t q_hi = (qh[idx / 8] >> (idx % 8)) & 1;
                    const uint8_t q = (q_hi << 4) | q_lo;  // 5-bit unsigned [0..31]
                    out[b * QK_K + idx] = static_cast<float>(q) * ds - dm;
                }
            }
            p += BLOCK_BYTES;
        }
        return out;
    }

    if (fmt == "q6_k") {
        constexpr size_t QK_K = 256;
        // Layout (per spec):
        //   ql[QK_K/2]      — 128 bytes (lower 4 bits, 2 per byte)
        //   qh[QK_K/4]      —  64 bytes (upper 2 bits, 4 per byte)
        //   scales[QK_K/16] —  16 bytes (one signed int8 per 16-element sub-block)
        //   d (fp16)        —   2 bytes (super-scale, at the END of the struct)
        // total = 210 bytes
        constexpr size_t BLOCK_BYTES = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
        static_assert(BLOCK_BYTES == 210, "Q6_K block size");
        if (segment.data.size() % BLOCK_BYTES != 0) {
            throw CompressionError("HSDR dequant: Q6_K byte size not a multiple of 210");
        }
        const size_t n_blocks = segment.data.size() / BLOCK_BYTES;
        std::vector<float> out(n_blocks * QK_K);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(segment.data.data());
        for (size_t b = 0; b < n_blocks; ++b) {
            const uint8_t* ql     = p;
            const uint8_t* qh     = p + QK_K / 2;
            const int8_t*  scales = reinterpret_cast<const int8_t*>(p + QK_K / 2 + QK_K / 4);
            uint16_t d_bits;
            std::memcpy(&d_bits, p + QK_K / 2 + QK_K / 4 + QK_K / 16, 2);
            const float d = fp16_to_fp32(d_bits);

            // 16 sub-blocks of 16 elements each. ql/qh are laid out in two halves
            // of 128 elements; within each half, ql packs 2 nibbles per byte and
            // qh packs 4 2-bit pairs per byte, both indexed by the within-half
            // position of the element.
            for (int half = 0; half < 2; ++half) {
                const int half_base = half * 128;
                const uint8_t* ql_h = ql + half * 64;
                const uint8_t* qh_h = qh + half * 32;
                for (int sub = 0; sub < 8; ++sub) {
                    const int8_t scale = scales[half * 8 + sub];
                    const float ds = d * static_cast<float>(scale);
                    for (int j = 0; j < 16; ++j) {
                        const int elem_in_half = sub * 16 + j;
                        const int idx = half_base + elem_in_half;
                        const uint8_t ql_byte = ql_h[elem_in_half / 2];
                        const uint8_t q_lo = (elem_in_half & 1) ? (ql_byte >> 4) : (ql_byte & 0x0F);
                        const uint8_t qh_byte = qh_h[elem_in_half / 4];
                        const uint8_t q_hi = (qh_byte >> ((elem_in_half % 4) * 2)) & 0x03;
                        const int q_unsigned = (q_hi << 4) | q_lo;  // [0..63]
                        const int q_signed = q_unsigned - 32;        // [-32..31]
                        out[b * QK_K + idx] = static_cast<float>(q_signed) * ds;
                    }
                }
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

    // Everything else (Q8_K, IQ*-, AWQ, GPTQ, etc.) requires format-specific
    // block-dequant code. Carry-overs to future rounds.
    throw CompressionError("HSDR dequant: unsupported input dtype '" + segment.data_format
                            + "' (type=" + std::to_string(static_cast<int>(segment.type))
                            + "); supported: f32, f16, bf16, i8, q4_0, q8_0, q4_k, q5_k, q6_k");
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
    const int32_t nt = static_cast<int32_t>(n_tiles);
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < nt; ++i) {
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

std::vector<float> HierarchicalSDRStrategy::dequantizeToFP32(const ModelSegment& segment) {
    // Public passthrough so tests / external callers can validate the
    // dequant in isolation from V4b fitting. Implementation lives in the
    // anonymous namespace at the top of this file.
    return dequantizeSegmentToFP32(segment);
}

HierarchicalSDRConfig HierarchicalSDRStrategy::configFor(const ModelSegment& segment) const {
    // Helper: in 1-D row-tile mode (tile_rows == 1) the row width is the
    // segment's actual column count, not a fixed knob. Adjust the config so
    // each shape is grouped/encoded against a dictionary sized to its real
    // input dim. Segments with different C end up in different buckets in
    // compressGroupedSegments — exactly what we want.
    //
    // For N-D tensors (e.g. ONNX conv weights of shape (O, I, kH, kW)), we
    // flatten to 2-D by treating dims[0] as rows and the product of remaining
    // dims as columns. This keeps each output channel as a single tile.
    auto adjustRowWidth = [&](HierarchicalSDRConfig cfg) -> HierarchicalSDRConfig {
        if (cfg.tile_rows == 1 && segment.tensor_metadata.has_value()) {
            const auto& dims = segment.tensor_metadata.value().dimensions;
            if (dims.size() >= 2) {
                size_t C = 1;
                for (size_t i = 1; i < dims.size(); ++i) C *= dims[i];
                if (C > 0 && C <= 0xFFFFu) {
                    cfg.tile_cols = static_cast<uint16_t>(C);
                }
            }
        }
        return cfg;
    };

    // Match the B.2 adaptive role-aware allocation: MLP gets the bigger config.
    switch (segment.type) {
        case SegmentType::FEED_FORWARD_WEIGHTS:
            return adjustRowWidth(mlp_default_);
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
        return adjustRowWidth(mlp_default_);
    }
    return adjustRowWidth(attn_default_);
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

    // Route very-large Q4_K segments (embeddings, lm_heads) through the
    // streaming path. The default compress() materialises the full FP32
    // dequantisation which OOMs at ~2 GB on 150k-vocab embeddings. Threshold
    // chosen at 256 MB FP32 — well below typical heap headroom but enough
    // to catch every "this should never have materialised" tensor.
    if (segment.tensor_metadata.has_value()) {
        const auto& dims = segment.tensor_metadata.value().dimensions;
        if (dims.size() >= 2) {
            const auto rc = effective2DShape_impl(dims);
            const size_t fp32_bytes = rc.first * rc.second * sizeof(float);
            // 1 GB threshold so only embedding-scale tensors (typically 2+ GB
            // FP32 for 100k+-vocab models) trigger streaming. Smaller MLPs
            // (~250-500 MB FP32) stay on the standard path which respects
            // the strategy's stored config.
            constexpr size_t kStreamThreshold = 1024ULL * 1024 * 1024;
            std::string fmt = segment.data_format;
            for (auto& c : fmt) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            const bool is_q4k = (fmt == "q4_k");
            if (fp32_bytes >= kStreamThreshold && is_q4k) {
                return compressStreaming(segment);
            }
        }
    }

    // Dequantise to FP32. Throws CompressionError for unsupported dtypes
    // (the strategy chain in AICompressor then falls through to the next
    // priority — typically QuantizedTensorStrategy or Gzip). std::bad_alloc
    // can fire on very large tensors (e.g. 150k-vocab embeddings →  2+ GB
    // FP32); we translate it to CompressionError so the chain falls through
    // gracefully instead of aborting the whole archive.
    std::vector<float> fp32_view;
    try {
        fp32_view = dequantizeSegmentToFP32(segment);
    } catch (const std::bad_alloc&) {
        throw CompressionError("HSDR: segment '" + segment.name
            + "' dequant exhausted memory (likely an embedding-scale tensor); "
              "deferring to lossless fallback");
    }
    const size_t num_elements = fp32_view.size();

    // Pull dimensions; flatten rank ≥ 2 to a 2-D (rows, cols) view.
    if (!segment.tensor_metadata.has_value() ||
        segment.tensor_metadata.value().dimensions.size() < 2) {
        throw CompressionError("HSDR: tensor_metadata.dimensions must have rank ≥ 2");
    }
    const auto& dims = segment.tensor_metadata.value().dimensions;
    const auto rc = effective2DShape_impl(dims);
    const uint32_t R = static_cast<uint32_t>(rc.first);
    const uint32_t C = static_cast<uint32_t>(rc.second);
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

    // originalSize comes from the archive header which records the SOURCE
    // dtype size (e.g. Q4_K_M-packed bytes), not the FP32 reconstruction
    // size. Accept either: the FP32 size we'll actually emit, OR a size
    // smaller than that (we trust the embedded header for the real shape).
    const size_t fp32_expected = static_cast<size_t>(R) * C * sizeof(float);
    if (originalSize != 0 && originalSize > fp32_expected) {
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

namespace {
using Utils::fp16_to_fp32;
// Streaming Q4_K dequant: decode one row at a time into the caller-provided
// FP32 buffer (sized C floats). Q4_K layout = 144 bytes / 256 elements per
// block; one row of C elements consumes C/256 blocks. Block layout matches
// the dequantizeSegmentToFP32 implementation at the top of this file.
struct Q4KRowStreamer {
    const uint8_t* base;
    size_t bytes_per_row;
    size_t blocks_per_row;
    uint32_t C;

    Q4KRowStreamer(const std::byte* data, uint32_t cols)
        : base(reinterpret_cast<const uint8_t*>(data)),
          C(cols)
    {
        constexpr size_t QK_K = 256;
        constexpr size_t BLOCK_BYTES = 144;
        if (cols % QK_K != 0) {
            throw CompressionError("Q4KRowStreamer: row dim not multiple of 256");
        }
        blocks_per_row = cols / QK_K;
        bytes_per_row = blocks_per_row * BLOCK_BYTES;
    }

    void decodeRow(size_t row_idx, float* out) const {
        constexpr size_t QK_K = 256;
        constexpr size_t BLOCK_BYTES = 144;
        const uint8_t* p = base + row_idx * bytes_per_row;
        uint8_t sc[8], mn[8];
        for (size_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = p + b * BLOCK_BYTES;
            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits, blk, 2);
            std::memcpy(&dmin_bits, blk + 2, 2);
            const float d    = fp16_to_fp32(d_bits);
            const float dmin = fp16_to_fp32(dmin_bits);
            // Unpack 6-bit scales + 6-bit mins from the 12-byte packed block.
            const uint8_t* scales12 = blk + 4;
            for (int is = 0; is < 4; ++is) {
                sc[is] = scales12[is] & 0x3F;
                mn[is] = scales12[is + 4] & 0x3F;
            }
            for (int is = 4; is < 8; ++is) {
                sc[is] = (scales12[is + 4] & 0x0F) | ((scales12[is - 4] >> 6) << 4);
                mn[is] = (scales12[is + 4] >> 4) | ((scales12[is] >> 6) << 4);
            }
            const uint8_t* qs = blk + 16;
            float* dst = out + b * QK_K;
            for (int is = 0; is < 8; ++is) {
                const float ds = d * static_cast<float>(sc[is]);
                const float dm = dmin * static_cast<float>(mn[is]);
                for (int j = 0; j < 32; ++j) {
                    const int idx = is * 32 + j;
                    const uint8_t byte = qs[idx / 2];
                    const uint8_t q = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
                    dst[idx] = static_cast<float>(q) * ds - dm;
                }
            }
        }
    }
};

bool isQ4KSegment(const ModelSegment& seg) {
    std::string fmt = seg.data_format;
    for (auto& c : fmt) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return fmt == "q4_k";
}
}  // anonymous namespace

std::vector<std::byte> HierarchicalSDRStrategy::compressStreaming(const ModelSegment& segment) const {
    if (!segment.isWeightTensor()) {
        throw CompressionError("compressStreaming: weight tensor required");
    }
    if (!segment.tensor_metadata.has_value()
        || segment.tensor_metadata.value().dimensions.size() < 2) {
        throw CompressionError("compressStreaming: rank-<2 tensor not supported");
    }
    if (!isQ4KSegment(segment)) {
        // Fall back to the regular path for non-Q4_K inputs. The whole point
        // of streaming is to avoid the 2 GB FP32 materialisation, which only
        // matters for very large quantised tensors; small/non-Q4_K segments
        // are happier on the standard path.
        return compress(segment);
    }
    const auto& dims = segment.tensor_metadata.value().dimensions;
    auto rc = effective2DShape_impl(dims);
    // GGUF stores tensor dims with the fastest-varying axis first (ggml
    // convention). For embeddings that gives dims=[hidden, vocab], which
    // effective2DShape reads as R=hidden, C=vocab. We want the opposite:
    // C should be the embedding dim (fits in uint16, what HSDR can tile),
    // R the number of token rows. Swap if C exceeds uint16 and R fits.
    if (rc.second > 0xFFFFu && rc.first <= 0xFFFFu) {
        std::swap(rc.first, rc.second);
    }
    const uint32_t R = static_cast<uint32_t>(rc.first);
    const uint32_t C = static_cast<uint32_t>(rc.second);
    if (C == 0 || R == 0) {
        throw CompressionError("compressStreaming: zero-sized shape");
    }
    if (C > 0xFFFFu) {
        throw CompressionError("compressStreaming: row dim exceeds uint16 tile_cols");
    }
    if (C % 256 != 0) {
        throw CompressionError("compressStreaming: Q4_K row dim must be multiple of 256");
    }

    // Start from the strategy's stored config (so --n-atoms, --ksvd-iters,
    // etc. from the CLI are respected), then force 1D row-tile mode with
    // tile_cols == C since streaming is only meaningful for that geometry.
    HierarchicalSDRConfig cfg = configFor(segment);
    cfg.tile_rows = 1;
    cfg.tile_cols = static_cast<uint16_t>(C);
    if (cfg.n_atoms == 0) {
        cfg.n_atoms = 32;  // sentinel safety
    }

    const uint32_t pool_capacity =
        cfg.max_tiles_for_fit > 0 ? std::min<uint32_t>(cfg.max_tiles_for_fit, R) : R;
    Q4KRowStreamer stream(segment.data.data(), C);

    // 1. Reservoir-sample R rows into a pool of pool_capacity rows, dequanting
    //    row-by-row. Memory peak: pool_capacity × C × 4 bytes (e.g. 4096 ×
    //    3584 × 4 = 56 MB for a Qwen embedding).
    std::vector<float> pool(static_cast<size_t>(pool_capacity) * C);
    uint64_t seen = 0;
    uint32_t pool_fill = 0;
    std::mt19937 rng(0xC0DECAFE);
    std::vector<float> row(C);
    for (uint32_t r = 0; r < R; ++r) {
        stream.decodeRow(r, row.data());
        if (pool_fill < pool_capacity) {
            std::memcpy(pool.data() + static_cast<size_t>(pool_fill) * C,
                        row.data(), C * sizeof(float));
            ++pool_fill;
        } else {
            std::uniform_int_distribution<uint64_t> dist(0, seen);
            const uint64_t j = dist(rng);
            if (j < pool_capacity) {
                std::memcpy(pool.data() + j * C, row.data(), C * sizeof(float));
            }
        }
        ++seen;
    }
    if (pool_fill < cfg.n_atoms) {
        throw CompressionError("compressStreaming: too few rows to fit dictionary");
    }

    // 2. Fit the dictionary on the pool.
    SharedDictionary dict;
    dict = fitSharedDictionary(pool, pool_fill, cfg);

    // 3. Stream rows again, encode each against the dict, pack codes.
    const uint32_t slots_per_tile = static_cast<uint32_t>(cfg.n_stages) * cfg.active_bits_per_stage;
    std::vector<uint16_t> indices(static_cast<size_t>(R) * slots_per_tile);
    std::vector<int8_t> signs(static_cast<size_t>(R) * slots_per_tile);
    for (uint32_t r = 0; r < R; ++r) {
        stream.decodeRow(r, row.data());
        hierarchicalBinaryCode(row.data(), dict.atoms.data(),
                                cfg.n_atoms, C, cfg.n_stages, cfg.active_bits_per_stage,
                                dict.stage_scales.data(),
                                indices.data() + static_cast<size_t>(r) * slots_per_tile,
                                signs.data()   + static_cast<size_t>(r) * slots_per_tile);
    }

    // 4. Serialise: per-tensor HSDR header + embedded dictionary + packed codes.
    //    Same format as compress()'s output so the existing decompress() path
    //    works unchanged.
    HSDRHeader header{};
    std::memcpy(header.magic, kHSDRMagic, 4);
    header.version = kHSDRVersion;
    header.original_rows = R;
    header.original_cols = C;
    header.n_tiles = R;
    header.tile_rows = cfg.tile_rows;
    header.tile_cols = cfg.tile_cols;
    header.n_atoms = cfg.n_atoms;
    header.n_stages = cfg.n_stages;
    header.active_bits_per_stage = cfg.active_bits_per_stage;
    header.stage_decay = cfg.stage_decay;
    header.edge_bytes_count = 0;

    // Same byte order as compress(): header, stage_scales, atoms, packed codes.
    std::vector<std::byte> out;
    out.reserve(sizeof(header) + dict.stage_scales.size() * sizeof(float)
                + dict.atoms.size() * sizeof(float)
                + indices.size() * sizeof(uint16_t));
    appendPOD(out, header);
    appendBuffer(out, dict.stage_scales.data(), dict.stage_scales.size());
    appendBuffer(out, dict.atoms.data(), dict.atoms.size());
    std::vector<uint16_t> packed(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) packed[i] = packIndexSign(indices[i], signs[i]);
    appendBuffer(out, packed.data(), packed.size());
    return out;
}

std::vector<std::byte> HierarchicalSDRStrategy::compressWithExternalDictionary(
    const ModelSegment& segment, const SharedDictionary& dict) const
{
    if (!segment.isWeightTensor()) {
        throw CompressionError("compressWithExternalDictionary: weight tensor required");
    }
    if (!segment.tensor_metadata.has_value()
        || segment.tensor_metadata.value().dimensions.size() < 2) {
        throw CompressionError("compressWithExternalDictionary: rank-<2 tensor not supported");
    }
    const auto& dims = segment.tensor_metadata.value().dimensions;
    const auto rc = effective2DShape_impl(dims);
    const uint32_t R = static_cast<uint32_t>(rc.first);
    const uint32_t C = static_cast<uint32_t>(rc.second);
    // Dequantise to FP32 if needed; delegate to the FP32 variant.
    std::vector<float> fp32_view = dequantizeSegmentToFP32(segment);
    if (fp32_view.size() != static_cast<size_t>(R) * C) {
        throw CompressionError(
            "compressWithExternalDictionary: dequantised element count doesn't match shape");
    }
    return compressFP32WithExternalDictionary(fp32_view.data(), R, C, dict,
                                              segment.type, segment.name);
}

std::vector<std::byte> HierarchicalSDRStrategy::compressFP32WithExternalDictionary(
    const float* fp32, uint32_t R, uint32_t C,
    const SharedDictionary& dict,
    SegmentType /*original_type*/,
    const std::string& name) const
{
    const auto& cfg = dict.config;
    if (R % cfg.tile_rows != 0 || C % cfg.tile_cols != 0) {
        throw CompressionError("compressFP32WithExternalDictionary: '" + name
                               + "' shape not divisible by tile size");
    }
    const uint32_t tile_dim = cfg.tileSize();
    if (dict.atoms.size() != static_cast<size_t>(cfg.n_atoms) * tile_dim) {
        throw CompressionError("compressFP32WithExternalDictionary: dictionary size doesn't match config");
    }

    // 1D row-tile fast path: the FP32 view is already laid out as (R) tiles
    // of width C — no extractTiles copy required. 2D mode still pays the
    // extractTiles cost; that's the rare path for big-LLM weight shapes.
    const bool is_1d_full_row = (cfg.tile_rows == 1) && (cfg.tile_cols == C);

    uint32_t n_tiles = 0;
    std::vector<float> tiles_storage;
    const float* tile_base = nullptr;
    if (is_1d_full_row) {
        n_tiles = R;
        tile_base = fp32;
    } else {
        tiles_storage = extractTiles(fp32, R, C, cfg.tile_rows, cfg.tile_cols, n_tiles);
        tile_base = tiles_storage.data();
    }

    const uint32_t slots_per_tile = static_cast<uint32_t>(cfg.n_stages) * cfg.active_bits_per_stage;
    std::vector<uint16_t> indices(static_cast<size_t>(n_tiles) * slots_per_tile);
    std::vector<int8_t>   signs(static_cast<size_t>(n_tiles) * slots_per_tile);
    encodeAllTilesAgainstDictionary(tile_base, n_tiles, tile_dim,
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
        || fmt == "q4_0" || fmt == "q8_0"
        || fmt == "q4_k" || fmt == "q5_k" || fmt == "q6_k";
}

inline std::pair<size_t, size_t> effective2DShape(const std::vector<size_t>& dims) {
    if (dims.size() < 2) return {0, 0};
    size_t cols = 1;
    for (size_t i = 1; i < dims.size(); ++i) cols *= dims[i];
    return {dims[0], cols};
}

bool segmentSuitable(const ModelSegment& seg, const HierarchicalSDRConfig& cfg,
                     std::string* reason = nullptr) {
    if (!seg.isWeightTensor()) {
        if (reason) *reason = "not a weight tensor";
        return false;
    }
    if (!seg.tensor_metadata.has_value() ||
        seg.tensor_metadata.value().dimensions.size() < 2) {
        if (reason) *reason = "missing tensor_metadata or rank < 2";
        return false;
    }
    if (!segmentDtypeSupported(seg)) {
        if (reason) *reason = "unsupported dtype (need f32/f16/bf16/i8)";
        return false;
    }
    const auto& dims = seg.tensor_metadata.value().dimensions;
    const auto [R, C] = effective2DShape(dims);
    if (R == 0 || C == 0) {
        if (reason) *reason = "zero-sized effective 2-D shape";
        return false;
    }
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

// Flatten an N-D weight to a 2-D (rows, cols) view. Rule: dims[0] becomes
// rows; remaining dims collapse into cols. This is the conventional flatten
// for conv weights (out_ch, in_ch, kH, kW) → (out_ch, in_ch*kH*kW). For 2-D
// weights it's a no-op. For 1-D / 0-D it returns (0,0) — treated as
// unsuitable upstream. Public-namespace version of the inline helper used
// inside the anonymous namespace above.
inline std::pair<size_t, size_t> effective2DShape_impl(const std::vector<size_t>& dims) {
    if (dims.size() < 2) return {0, 0};
    size_t cols = 1;
    for (size_t i = 1; i < dims.size(); ++i) cols *= dims[i];
    return {dims[0], cols};
}

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

        // 2a. Pool tiles. Per-segment tile extraction with optional reservoir
        // sampling, so the in-memory pool is bounded by max_tiles_for_fit
        // regardless of total tile count. Crucial for big models — without it,
        // a bucket of large MLP tensors easily allocates many GB before fit.
        const uint32_t tile_dim = cfg.tileSize();

        // First pass: total tile count (no allocation).
        uint64_t total_tiles_64 = 0;
        for (size_t idx : seg_indices) {
            const auto& s = segments[idx];
            const auto& dims = s.tensor_metadata.value().dimensions;
            const auto rc = effective2DShape_impl(dims);
            const uint64_t R = static_cast<uint64_t>(rc.first);
            const uint64_t C = static_cast<uint64_t>(rc.second);
            total_tiles_64 += (R / cfg.tile_rows) * (C / cfg.tile_cols);
        }
        const uint32_t pooled_count = (total_tiles_64 > 0xFFFFFFFFu)
                                       ? 0xFFFFFFFFu
                                       : static_cast<uint32_t>(total_tiles_64);

        // Capacity for the pool (in tiles): unbounded → use total; capped →
        // use the cap. Encoding still runs on every tile in step 2d.
        const uint32_t pool_capacity_tiles =
            (cfg.max_tiles_for_fit > 0 && total_tiles_64 > cfg.max_tiles_for_fit)
                ? cfg.max_tiles_for_fit
                : pooled_count;

        std::vector<float> pooled(static_cast<size_t>(pool_capacity_tiles) * tile_dim);
        bool use_reservoir = (pool_capacity_tiles < pooled_count);

        // Second pass: dequantise per-segment (one segment's FP32 in memory
        // at a time), iterate its tiles, and either append (no cap) or run
        // reservoir sampling (cap).
        std::mt19937 res_rng(0xC0DECAFE ^ pool_capacity_tiles);
        uint64_t seen_tiles = 0;     // global counter for reservoir indexing
        uint32_t pool_fill = 0;      // tiles currently in the pool (≤ capacity)

        // Track which seg_indices were successfully pooled, so we only try to
        // encode those in step 2d. (If a dequant OOMs we just skip the segment.)
        std::vector<size_t> pooled_seg_indices;
        pooled_seg_indices.reserve(seg_indices.size());
        // Cache the dequantised FP32 view per pooled segment so step 2d doesn't
        // re-dequantise. Parallel to pooled_seg_indices. Memory cost: the sum
        // of all bucket weights as FP32 (bounded by the bucket's source size
        // times the dtype expansion factor — typically 4-8x for Q4_K_M).
        std::vector<std::vector<float>> cached_fp32;
        cached_fp32.reserve(seg_indices.size());

        for (size_t idx : seg_indices) {
            const auto& s = segments[idx];
            const auto& dims = s.tensor_metadata.value().dimensions;
            const auto rc = effective2DShape_impl(dims);
            const uint32_t R = static_cast<uint32_t>(rc.first);
            const uint32_t C = static_cast<uint32_t>(rc.second);

            // For 1D row-tile mode (tile_rows == 1, tile_cols == C) the segment's
            // FP32 view is already laid out as (R) tiles of width C — no second
            // tile array is needed. For 2D mode we still call extractTiles, but
            // big LLM tensors almost always hit the 1D path under --fast, so the
            // peak heap is just one fp32_view rather than fp32_view + tiles.
            const bool is_1d_full_row =
                (cfg.tile_rows == 1) && (cfg.tile_cols == static_cast<uint32_t>(C));

            // Guard against single-segment OOM (e.g. 150k-vocab embedding
            // dequanting to 2+ GB FP32). The whole bucket survives a single
            // bad-alloc — the offending segment is reported in out_skipped_names.
            std::vector<float> fp32_view;
            try {
                fp32_view = dequantizeSegmentToFP32(s);
            } catch (const std::bad_alloc&) {
                if (out_skipped_names) out_skipped_names->push_back(s.name);
                continue;
            } catch (const std::exception&) {
                if (out_skipped_names) out_skipped_names->push_back(s.name);
                continue;
            }
            if (fp32_view.size() != static_cast<size_t>(R) * C) {
                throw CompressionError("compressGroupedSegments: dequant size mismatch on '"
                                       + s.name + "'");
            }

            // Per-segment NaN/Inf gate. Real-world quantised tensors occasionally
            // have degenerate block scales (e.g. fp16 super-scale with exponent=31
            // → Inf, or upstream parser bugs reading wrong bytes for a shard).
            // Even a small fraction of NaN tiles corrupts the K-SVD fit because
            // residuals propagate NaN to every atom. Sanitize NaN/Inf to zero
            // and skip the whole segment if it's mostly bad.
            //
            // Two-stage check: first a sparse probe (1-in-256) bailing
            // immediately for clean tensors (the common case). Only fall back
            // to a full parallel sweep when the probe hits a non-finite value.
            {
                const size_t n = fp32_view.size();
                bool any_bad = false;
                constexpr size_t kProbeStride = 256;
                for (size_t i = 0; i < n; i += kProbeStride) {
                    if (!std::isfinite(fp32_view[i])) { any_bad = true; break; }
                }
                if (any_bad) {
                    // Full parallel sweep: count + sanitise non-finite values.
                    size_t bad = 0;
                    const int64_t ns = static_cast<int64_t>(n);
                    #pragma omp parallel for reduction(+:bad) schedule(static)
                    for (int64_t i = 0; i < ns; ++i) {
                        if (!std::isfinite(fp32_view[i])) {
                            fp32_view[i] = 0.0f;
                            ++bad;
                        }
                    }
                    const double bad_ratio = static_cast<double>(bad) / std::max<size_t>(1, n);
                    if (bad_ratio > 1e-3) {
                        if (out_skipped_names) out_skipped_names->push_back(s.name);
                        continue;
                    }
                }
            }
            uint32_t nt_check = 0;
            std::vector<float> tiles_storage;
            const float* tile_base = nullptr;
            if (is_1d_full_row) {
                nt_check = R;
                tile_base = fp32_view.data();
            } else {
                tiles_storage = extractTiles(fp32_view.data(), R, C,
                                              cfg.tile_rows, cfg.tile_cols, nt_check);
                tile_base = tiles_storage.data();
                std::vector<float>().swap(fp32_view);  // free early in 2D path
            }

            if (!use_reservoir) {
                std::memcpy(pooled.data() + static_cast<size_t>(pool_fill) * tile_dim,
                            tile_base,
                            static_cast<size_t>(nt_check) * tile_dim * sizeof(float));
                pool_fill += nt_check;
                seen_tiles += nt_check;
                pooled_seg_indices.push_back(idx);
                cached_fp32.push_back(std::move(fp32_view));
                continue;
            }
            // Reservoir sampling: each incoming tile either lands in the
            // reservoir directly (until full) or replaces a random existing
            // entry with probability (capacity / seen_so_far).
            for (uint32_t i = 0; i < nt_check; ++i) {
                const float* src = tile_base + static_cast<size_t>(i) * tile_dim;
                if (pool_fill < pool_capacity_tiles) {
                    std::memcpy(pooled.data() + static_cast<size_t>(pool_fill) * tile_dim,
                                src, tile_dim * sizeof(float));
                    ++pool_fill;
                } else {
                    std::uniform_int_distribution<uint64_t> dist(0, seen_tiles);
                    const uint64_t j = dist(res_rng);
                    if (j < pool_capacity_tiles) {
                        std::memcpy(pooled.data() + static_cast<size_t>(j) * tile_dim,
                                    src, tile_dim * sizeof(float));
                    }
                }
                ++seen_tiles;
            }
            pooled_seg_indices.push_back(idx);
            // In 1D mode fp32_view is still alive (we streamed tiles from it
            // directly without an extractTiles copy). Keep it for encode reuse.
            // 2D mode already freed it above; cache an empty placeholder so
            // indices stay aligned.
            cached_fp32.push_back(std::move(fp32_view));
        }
        // Trim the pool buffer to actual fill (rare in capped path; usual in
        // the uncapped path where pool_fill == pooled_count). Also make
        // pooled_count reflect the in-pool count from this point forward.
        if (pool_fill != pooled.size() / tile_dim) {
            pooled.resize(static_cast<size_t>(pool_fill) * tile_dim);
        }

        // Pool integrity check. If any non-finite values made it into the
        // pool, the K-SVD fit will produce NaN atoms regardless of how
        // robust the update is. Scan, count, and sanitize.
        {
            size_t nan_n = 0, inf_n = 0;
            const size_t pool_n = static_cast<size_t>(pool_fill) * tile_dim;
            for (size_t i = 0; i < pool_n; ++i) {
                const float v = pooled[i];
                if (std::isnan(v)) { pooled[i] = 0.0f; ++nan_n; }
                else if (std::isinf(v)) { pooled[i] = 0.0f; ++inf_n; }
            }
            if (nan_n + inf_n > 0) {
                std::cerr << "[hsdr-pool] sanitized " << nan_n << " NaN + "
                          << inf_n << " Inf values in pool (size " << pool_n
                          << " floats, " << pool_fill << " tiles)\n";
            }
        }

        if (pool_fill < cfg.n_atoms) {
            // Pool too small to fit the requested dictionary — skip the whole bucket
            // and mark every successfully-pooled member as skipped (segments that
            // already failed dequant were pushed to out_skipped_names above).
            for (size_t idx : pooled_seg_indices) {
                if (out_skipped_names) out_skipped_names->push_back(segments[idx].name);
            }
            continue;
        }

        // 2c. Fit shared dictionary on the reservoir-sampled pool.
        SharedDictionary shared = fitSharedDictionary(pooled, pool_fill, cfg);
        const size_t dict_index = archive.dictionaries.size();
        archive.dictionaries.push_back(std::move(shared));

        // 2d. Encode each successfully-pooled segment against the new shared dict.
        // Reuse the FP32 view cached during pool fill so we don't pay a second
        // dequant. For segments whose fp32_view was freed early (2D mode),
        // fall back to the dequant-internal path.
        const SharedDictionary& dict_ref = archive.dictionaries[dict_index];
        for (size_t ki = 0; ki < pooled_seg_indices.size(); ++ki) {
            const size_t idx = pooled_seg_indices[ki];
            const auto& s = segments[idx];
            SharedDictArchive::SegmentEntry entry;
            entry.name = s.name;
            entry.dict_index = dict_index;
            try {
                const auto& dims = s.tensor_metadata.value().dimensions;
                const auto rc = effective2DShape_impl(dims);
                const uint32_t R = static_cast<uint32_t>(rc.first);
                const uint32_t C = static_cast<uint32_t>(rc.second);
                if (!cached_fp32[ki].empty()) {
                    entry.codes_bytes = compressFP32WithExternalDictionary(
                        cached_fp32[ki].data(), R, C, dict_ref, s.type, s.name);
                    // Free the cached fp32 as soon as it's encoded — reduces
                    // peak memory across the encode loop.
                    std::vector<float>().swap(cached_fp32[ki]);
                } else {
                    entry.codes_bytes = compressWithExternalDictionary(s, dict_ref);
                }
            } catch (const std::bad_alloc&) {
                if (out_skipped_names) out_skipped_names->push_back(s.name);
                continue;
            } catch (const std::exception&) {
                if (out_skipped_names) out_skipped_names->push_back(s.name);
                continue;
            }
            // original_size is the FP32 reconstruction byte size, NOT the
            // on-disk source size. The decompressor reconstructs into FP32
            // and validates against this field. Source dtype is preserved
            // separately in original_type for ratio reporting. For N-D
            // tensors we use the product of all dimensions (the bytes are
            // the same; only the logical shape is flattened for tiling).
            const auto& dims = s.tensor_metadata.value().dimensions;
            size_t elem_count = 1;
            for (size_t d : dims) elem_count *= d;
            entry.original_size = elem_count * sizeof(float);
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
