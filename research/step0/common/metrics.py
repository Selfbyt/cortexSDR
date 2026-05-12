"""Reconstruction-quality metrics + FLOP counters.

Two layers of metric:
- structural: MSE / normalized MSE on the weight matrix itself (cheap, per-tile)
- behavioral: perplexity on real text (expensive, eval.py)

For Step 0 we use structural metrics for hyperparameter sweeps, then confirm
the locked design with behavioral metrics.
"""
from __future__ import annotations

import numpy as np


def reconstruction_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Plain mean-squared error between two equal-shape arrays."""
    diff = (original - reconstructed).astype(np.float64)
    return float(np.mean(diff * diff))


def normalized_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """MSE divided by original-variance. Comparable across layers of different scale.

    Returned value: 0 = perfect; 1 = no better than the mean; > 1 = worse than mean.
    """
    diff = (original - reconstructed).astype(np.float64)
    num = float(np.mean(diff * diff))
    denom = float(np.var(original.astype(np.float64)))
    return num / max(denom, 1e-12)


def dense_matmul_flops(m: int, k: int, n: int) -> int:
    """Standard 2*m*k*n for an m×k by k×n dense matmul (1 mul + 1 add per element)."""
    return 2 * m * k * n


def sparse_decode_flops(
    *,
    tile_rows: int,
    tile_cols: int,
    n_tiles: int,
    dict_atoms: int,
    active_bits: int,
    code_type: str,
    input_dim: int,
    output_dim: int,
    batch: int = 1,
) -> dict[str, int]:
    """FLOPs for the fused sparse-decode inference path of one matmul.

    Shapes:
      D : (dict_atoms K, tile_cols C)        — shared dictionary
      S : (n_tiles, K), sparse with `active_bits` non-zeros per row
      x : (input_dim, batch)
      Y : (output_dim, batch)

    For each column block c of the input (n_col_tiles = input_dim / tile_cols),
    compute Dx_c = D @ x_c   shape (K, batch).
    Then for each of n_row_tiles row tiles, gather `active_bits` rows of Dx_c
    weighted by the corresponding sparse code, accumulate into Y.

    Decomposed cost (units = FLOPs, counting 1 mul + 1 add as 2):
      - precompute_dx:   2 * K * input_dim * batch
                         (= n_col_tiles * 2*K*tile_cols*batch — same thing)
      - per_tile_gather: per_tile_ops * n_tiles * batch
                         where per_tile_ops = active_bits for binary (adds-only),
                         2*active_bits for real/hybrid (mul + add)
      - residual_mm:     only for hybrid; small dense residual matmul on top
    """
    # Precompute D @ x: across all column blocks, total = 2 * K * input_dim * batch
    precompute = 2 * dict_atoms * input_dim * batch

    # Per-tile work: for each tile, sum `active_bits` rows of Dx (length batch each)
    if code_type == "binary":
        per_tile_ops = active_bits  # adds-only across batch dim
    elif code_type in ("real", "hybrid"):
        per_tile_ops = 2 * active_bits  # mul + add
    else:
        raise ValueError(f"Unknown code_type: {code_type}")
    per_tile_total = per_tile_ops * n_tiles * batch

    residual_mm = 0
    if code_type == "hybrid":
        # A small dense residual matmul over the same x. Assume residual rank
        # = active_bits / 2 (heuristic — refine when we implement the residual path).
        residual_rank = max(1, active_bits // 2)
        residual_mm = dense_matmul_flops(residual_rank, input_dim, batch) + \
                      dense_matmul_flops(output_dim, residual_rank, batch)

    total = precompute + per_tile_total + residual_mm
    return {
        "precompute_dx": precompute,
        "per_tile_total": per_tile_total,
        "residual_mm": residual_mm,
        "total": total,
    }


def compression_ratio(
    *,
    original_numel: int,
    original_bytes_per_elem: int,
    dict_atoms: int,
    tile_cols: int,
    n_tiles: int,
    active_bits: int,
    code_type: str,
    dict_bytes_per_elem: int = 4,
) -> dict[str, float]:
    """Compute storage ratio + breakdown for the encoded model.

    Storage:
      - dictionary: dict_atoms × tile_cols × dict_bytes
      - per-tile SDR indices: ceil(log2(dict_atoms)) bits × active_bits
      - per-tile real coefficients (if applicable): active_bits × 4 bytes
        (hybrid uses half-precision real residuals; we use 2 bytes for that case)
    """
    bits_per_index = max(1, int(np.ceil(np.log2(max(2, dict_atoms)))))
    dict_bytes = dict_atoms * tile_cols * dict_bytes_per_elem

    sdr_bits = active_bits * bits_per_index
    if code_type == "binary":
        coef_bytes = 0
    elif code_type == "real":
        coef_bytes = active_bits * 4  # FP32 coefficients
    elif code_type == "hybrid":
        coef_bytes = (active_bits // 2) * 2  # half-precision residual coefs
    else:
        raise ValueError(f"Unknown code_type: {code_type}")
    per_tile_bytes = (sdr_bits + 7) // 8 + coef_bytes
    encoded_bytes = dict_bytes + per_tile_bytes * n_tiles

    original_bytes = original_numel * original_bytes_per_elem
    return {
        "ratio": original_bytes / max(1, encoded_bytes),
        "dict_bytes": dict_bytes,
        "per_tile_bytes": per_tile_bytes,
        "encoded_bytes": encoded_bytes,
        "original_bytes": original_bytes,
    }
