/**
 * @file HierarchicalSDRStrategy.hpp
 * @brief Hierarchical-binary SDR compression for weight tensors (V4b).
 *
 * Implements the Cortex-SDR design validated in research/step0:
 *   - Tile each weight matrix into fixed-size blocks (default 128x128).
 *   - Fit a per-tensor dictionary D and multi-stage binary codes via
 *     hierarchical K-SVD: tile ≈ Σ_l γ_l · (Σ_i sign_l,i · D[idx_l,i]).
 *   - Storage = dictionary atoms (FP32) + per-tile (index, sign) entries only;
 *     no real-valued sparse coefficients leave the encoder.
 *
 * Per-role hyperparameters are auto-selected from segment metadata
 * (attention vs MLP), matching the B.2 adaptive allocation finding.
 *
 * Limitations of this first implementation:
 *   - Per-tensor dictionary (no cross-layer/role sharing yet — that requires
 *     a multi-pass compression pipeline above the strategy interface).
 *   - Edge tiles (when shape isn't divisible by tile size) raise an error.
 *     Production should pad with FP16 fallback; deferred for now.
 *   - Only FP32 weight tensors are accepted; other dtypes fall through.
 */
#ifndef HIERARCHICAL_SDR_STRATEGY_HPP
#define HIERARCHICAL_SDR_STRATEGY_HPP

#include "CompressionStrategy.hpp"
#include "../core/ModelSegment.hpp"
#include <cstdint>
#include <vector>

namespace CortexAICompression {

/**
 * @brief Hyperparameters for the hierarchical K-SVD fit + serialization.
 *
 * Defaults match the B.2 adaptive role-aware setting for attention layers.
 * MLP variants are picked up automatically in compress() based on segment role.
 */
struct HierarchicalSDRConfig {
    uint16_t tile_rows = 128;
    uint16_t tile_cols = 128;
    uint16_t n_atoms = 256;             ///< Dictionary size (K). MLP override uses 512.
    uint8_t  n_stages = 3;
    uint8_t  active_bits_per_stage = 5; ///< Per-stage budget. MLP override uses 8.
    uint8_t  ksvd_iters = 8;            ///< Iterations of greedy MP + atom update.
    uint8_t  reserved_pad = 0;          ///< Keeps the struct 32-bit aligned in the stream.
    float    stage_decay = 0.5f;        ///< γ_l = stage_decay^l.

    /** Auto-pick MLP-flavoured hyperparameters; matches research/step0 exp_b2. */
    static HierarchicalSDRConfig forMLP() {
        HierarchicalSDRConfig c;
        c.n_atoms = 512;
        c.active_bits_per_stage = 8;
        c.ksvd_iters = 12;
        return c;
    }

    /**
     * @brief 1D row-tile config: each row of W is its own tile.
     *
     * This is the layout that makes the precompute-once-per-token-block fused
     * inference math actually win against dense matmul. Each atom is a row
     * vector of length row_width (= input dim of the matmul). See
     * matmulRowMajor() and the 2D-vs-1D analysis in STEP0_REPORT.md.
     */
    static HierarchicalSDRConfig forRow1D(uint16_t row_width,
                                          uint16_t n_atoms = 512,
                                          uint8_t k_per_stage = 8) {
        HierarchicalSDRConfig c;
        c.tile_rows = 1;
        c.tile_cols = row_width;
        c.n_atoms = n_atoms;
        c.active_bits_per_stage = k_per_stage;
        c.n_stages = 3;
        c.ksvd_iters = 12;
        c.stage_decay = 0.5f;
        return c;
    }

    /** Total per-tile active bits = n_stages * active_bits_per_stage. */
    uint32_t totalActiveBits() const {
        return static_cast<uint32_t>(n_stages) * active_bits_per_stage;
    }
    /** Per-tile flattened element count. */
    uint32_t tileSize() const {
        return static_cast<uint32_t>(tile_rows) * tile_cols;
    }
};

/**
 * @brief V4b hierarchical binary SDR compression strategy.
 *
 * compress() throws CompressionError on:
 *   - non-FP32 weight tensors
 *   - tensors smaller than the tile size
 *   - tensors whose 2-D shape isn't divisible by (tile_rows, tile_cols)
 *
 * On success, the byte stream contains a header + stage scales + dictionary +
 * packed per-tile codes. decompress() inverts that.
 */
class HierarchicalSDRStrategy : public ICompressionStrategy {
public:
    explicit HierarchicalSDRStrategy(const HierarchicalSDRConfig& attn_default = {})
        : attn_default_(attn_default), mlp_default_(HierarchicalSDRConfig::forMLP()) {}

    HierarchicalSDRStrategy(const HierarchicalSDRConfig& attn_default,
                            const HierarchicalSDRConfig& mlp_default)
        : attn_default_(attn_default), mlp_default_(mlp_default) {}

    std::vector<std::byte> compress(const ModelSegment& segment) const override;

    std::vector<std::byte> decompress(const std::vector<std::byte>& compressedData,
                                      SegmentType originalType,
                                      size_t originalSize) const override;

    /**
     * @brief Fused matmul Y = W·x directly from the compressed bytes.
     *
     * Only meaningful for 1D row-tile encodings (tile_rows == 1); for 2D tiles
     * this throws CompressionError because the precompute cost exceeds dense
     * matmul (see implementation comments / STEP0_REPORT.md).
     *
     * Output Y has shape (original_rows, batch), row-major. The caller
     * provides x of shape (original_cols, batch).
     *
     * Conceptual cost (1D row tiles, binary codes):
     *   precompute Dx (K × batch):     2 · K · C · batch FLOPs
     *   per-row gather:                n_active · R · batch additions
     *
     * No FP32 weight matrix is materialised in this path.
     */
    std::vector<float> matmulRowMajor(const std::vector<std::byte>& compressedData,
                                      const float* x,
                                      size_t batch) const;

private:
    HierarchicalSDRConfig attn_default_;
    HierarchicalSDRConfig mlp_default_;

    /** Pick attn vs mlp config from segment role / type. */
    HierarchicalSDRConfig configFor(const ModelSegment& segment) const;
};

} // namespace CortexAICompression

#endif // HIERARCHICAL_SDR_STRATEGY_HPP
