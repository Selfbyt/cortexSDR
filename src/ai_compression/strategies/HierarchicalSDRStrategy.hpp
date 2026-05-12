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
#include <functional>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    /// Returns true if the segment should be PROTECTED (HSDR refuses to compress
    /// it, strategy chain falls through to the next-priority strategy such as
    /// Quant or Gzip — typically lossless FP16/FP32 storage). Used to implement
    /// the C.2 / D.1 finding that boundary-layer MLPs must stay lossless.
    using ProtectionPredicate = std::function<bool(const ModelSegment&)>;

    explicit HierarchicalSDRStrategy(const HierarchicalSDRConfig& attn_default = {})
        : attn_default_(attn_default), mlp_default_(HierarchicalSDRConfig::forMLP()) {}

    HierarchicalSDRStrategy(const HierarchicalSDRConfig& attn_default,
                            const HierarchicalSDRConfig& mlp_default)
        : attn_default_(attn_default), mlp_default_(mlp_default) {}

    /// Install a predicate that decides which segments to skip (force fallback).
    /// Pass an empty std::function to disable.
    void setProtectionPredicate(ProtectionPredicate pred) { protection_ = std::move(pred); }

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

    // ---------------------------------------------------------------------
    // Shared-dictionary mode (cross-layer / cross-role sharing).
    //
    // Building blocks for a multi-pass compression pipeline that fits one
    // dictionary on tiles drawn from many weight tensors, then encodes each
    // tensor's codes against that single shared dictionary. Amortises the
    // dictionary cost across all participating tensors — at Llama-scale this
    // is where the storage ratio actually wins.
    // ---------------------------------------------------------------------

    /// Shared dictionary trained across multiple sources.
    struct SharedDictionary {
        HierarchicalSDRConfig config; ///< (tile_rows, tile_cols, n_atoms, n_stages, k_per_stage)
        std::vector<float>    atoms;   ///< (n_atoms × tile_rows*tile_cols) row-major
        std::vector<float>    stage_scales; ///< n_stages entries
    };

    /**
     * @brief Fit a shared dictionary from tile data pooled across many tensors.
     *
     * Caller stacks tiles from all participating tensors into one big
     * (n_total_tiles × tile_dim) array and passes it in. The result is the
     * dictionary + stage scales matching `cfg`; encode() / matmul() against
     * this dictionary works for any tensor whose tile geometry matches `cfg`.
     */
    SharedDictionary fitSharedDictionary(const std::vector<float>& pooled_tiles,
                                         uint32_t n_total_tiles,
                                         const HierarchicalSDRConfig& cfg) const;

    /**
     * @brief Encode a single segment's tiles against a pre-fit shared dictionary.
     *
     * Output bytes contain only the per-tile (index, sign) codes plus a small
     * header — the dictionary is NOT embedded. Pair with a separately-stored
     * SharedDictionary to decode.
     */
    std::vector<std::byte> compressWithExternalDictionary(
        const ModelSegment& segment, const SharedDictionary& dict) const;

    /**
     * @brief Decompress a codes-only stream using an external shared dictionary.
     *
     * Output is the FP32 weight matrix bytes, identical in layout to what
     * decompress() returns for a per-tensor encoding.
     */
    std::vector<std::byte> decompressWithExternalDictionary(
        const std::vector<std::byte>& codes_bytes,
        const SharedDictionary& dict,
        size_t originalSize) const;

    /**
     * @brief Fused matmul using a codes-only stream + external shared dictionary.
     *
     * Same speedup characteristics as matmulRowMajor() — requires 1D row tiles.
     */
    std::vector<float> matmulWithExternalDictionary(
        const std::vector<std::byte>& codes_bytes,
        const SharedDictionary& dict,
        const float* x,
        size_t batch) const;

    // ---------------------------------------------------------------------
    // Multi-pass compression pipeline.
    //
    // Groups a batch of segments by (role-derived) compression config,
    // pools tiles per group, fits a shared dictionary per group, and
    // encodes each segment as codes-only bytes referencing its group's
    // dictionary. This is the orchestration that turns the shared-dict
    // building blocks into a usable cross-layer compressor.
    // ---------------------------------------------------------------------

    /// Result of compressGroupedSegments() — in-memory, not yet a file format.
    struct SharedDictArchive {
        std::vector<SharedDictionary> dictionaries; ///< One per group/role
        struct SegmentEntry {
            std::string name;
            size_t dict_index;                       ///< Index into dictionaries[]
            std::vector<std::byte> codes_bytes;
            size_t original_size;                    ///< Bytes for decompress validation
            SegmentType original_type;
        };
        std::vector<SegmentEntry> segments;

        /** Total bytes (dictionaries + all codes) — useful for storage comparisons. */
        size_t totalBytes() const;

        // -----------------------------------------------------------------
        // Serialisation. Self-contained format — not yet integrated with the
        // main .sdr archive (that requires touching AICompressor and
        // SDRModelLoader). Magic = "HSDA", little-endian, see implementation
        // for the exact layout.
        // -----------------------------------------------------------------
        /** Write to a binary stream. Throws on I/O failure. */
        void writeToStream(std::ostream& out) const;

        /** Read from a binary stream produced by writeToStream(). */
        static SharedDictArchive readFromStream(std::istream& in);

        /** Convenience: write to file at the given path. */
        void writeToFile(const std::string& path) const;

        /** Convenience: read from file at the given path. */
        static SharedDictArchive readFromFile(const std::string& path);
    };

    /**
     * @brief Compress a batch of segments using one shared dictionary per role.
     *
     * Segments with matching compression config (n_atoms, n_stages,
     * active_bits_per_stage, tile_rows, tile_cols, stage_decay) are pooled
     * and share a dictionary. Segments that fail validation (wrong dtype,
     * shape too small, protection predicate hits) are skipped — the caller
     * is expected to handle them via a parallel fallback path.
     *
     * Skipped segments are reported via `out_skipped_names` if provided.
     */
    SharedDictArchive compressGroupedSegments(
        const std::vector<ModelSegment>& segments,
        std::vector<std::string>* out_skipped_names = nullptr) const;

private:
    HierarchicalSDRConfig attn_default_;
    HierarchicalSDRConfig mlp_default_;
    ProtectionPredicate   protection_;

    /** Pick attn vs mlp config from segment role / type. */
    HierarchicalSDRConfig configFor(const ModelSegment& segment) const;
};

// --------------------------------------------------------------------------
// HSDAReader — name-indexed view over a SharedDictArchive.
//
// Wraps an in-memory SharedDictArchive (typically loaded from a .hsda file)
// with a name → segment-index map and convenience methods for decompression
// and fused matmul. This is the API the inference engine will use to fetch
// weights / multiply activations without ever materialising the full FP32
// weight matrices.
// --------------------------------------------------------------------------
class HSDAReader {
public:
    explicit HSDAReader(HierarchicalSDRStrategy::SharedDictArchive archive);

    /** Convenience: load straight from a .hsda file on disk. */
    static HSDAReader fromFile(const std::string& path);

    /** Number of distinct dictionaries (one per role/group). */
    size_t numDictionaries() const { return archive_.dictionaries.size(); }

    /** Number of compressed segments addressable by name. */
    size_t numSegments() const { return archive_.segments.size(); }

    /** Underlying archive (for advanced inspection). */
    const HierarchicalSDRStrategy::SharedDictArchive& archive() const { return archive_; }

    /** True iff `name` resolves to a compressed segment. */
    bool hasSegment(const std::string& name) const;

    /**
     * @brief Decompress a segment into FP32 bytes (row-major weight matrix).
     * @throws std::runtime_error if `name` isn't in the archive.
     */
    std::vector<std::byte> decompress(const std::string& name) const;

    /**
     * @brief Fused matmul: Y = W · x for the named segment.
     * @param x Input activation, row-major (cols × batch).
     * @param batch Number of input columns.
     * @return Y of shape (segment_rows, batch), row-major.
     *
     * @throws std::runtime_error If segment is missing or not 1D row-tile.
     *
     * No FP32 weight matrix is materialised; this is the speed-path.
     */
    std::vector<float> matmul(const std::string& name,
                              const float* x,
                              size_t batch) const;

private:
    HierarchicalSDRStrategy::SharedDictArchive archive_;
    HierarchicalSDRStrategy                    strat_;  // stateless decode/matmul
    std::unordered_map<std::string, size_t>    name_index_;
};

// --------------------------------------------------------------------------
// Helper factories for common protection policies.
// --------------------------------------------------------------------------
namespace ProtectionPolicies {

/**
 * @brief Protect MLP segments in the first and last `n_boundary` decoder layers.
 *
 * Encodes the C.2 Regime F finding: V4b'ing early-decoder MLPs is the main
 * source of super-linear perplexity compounding (Exp D.1). The C.2 sample
 * (5 sampled depths) showed best results at n_boundary in {2, 3} for TinyLlama.
 *
 * Reads `segment.layer_index` to identify decoder depth and `segment.type`
 * to detect MLP roles (FEED_FORWARD_WEIGHTS or names containing mlp/ffn).
 */
HierarchicalSDRStrategy::ProtectionPredicate boundaryMLPs(
    size_t n_boundary, size_t total_layers);

/**
 * @brief Protect any segment whose `name` appears in the supplied set.
 *        Useful for one-off overrides (e.g. embeddings, lm_head).
 */
HierarchicalSDRStrategy::ProtectionPredicate byName(
    std::unordered_set<std::string> names);

}  // namespace ProtectionPolicies

} // namespace CortexAICompression

#endif // HIERARCHICAL_SDR_STRATEGY_HPP
