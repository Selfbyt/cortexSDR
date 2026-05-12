"""Three sparse-code variants over a shared dictionary-learning interface.

All three solve the same problem: given a tile X (R, C) and a dictionary D (K, C),
find a sparse code S (K,) such that X ≈ D.T @ S or X ≈ S @ D (with the right
convention). We use: tile (as row vector) ≈ s @ D, i.e. D is (K, tile_size).

Variants:
- real: standard Orthogonal Matching Pursuit (OMP), real-valued sparse coefficients
- binary: pick top-k columns of |s_omp|, set to ±1 (sign-preserved)
- hybrid: top-k indices binary {±1}, plus a small real residual code over
          uncovered atoms (rank `active_bits // 2`)

OMP via sklearn.linear_model.orthogonal_mp for batched solving.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import orthogonal_mp


CodeType = Literal["binary", "real", "hybrid"]


@dataclass
class SparseCode:
    """Encoded representation of a tile batch.

    For binary: signs is ±1 at active positions, 0 elsewhere; coefs unused.
    For real:   coefs holds the real-valued coefficients at active positions.
    For hybrid: signs holds ±1 binary; residual_coefs holds half-precision residual.
    """
    indices: np.ndarray       # (n_tiles, active_bits) int32 — atom indices
    signs: np.ndarray         # (n_tiles, active_bits) int8 ±1 (binary/hybrid) or unused
    coefs: np.ndarray         # (n_tiles, active_bits) float32 — for real
    residual_indices: np.ndarray  # (n_tiles, active_bits // 2) int32 — for hybrid
    residual_coefs: np.ndarray    # (n_tiles, active_bits // 2) float16 — for hybrid
    code_type: CodeType


def encode_real(
    tiles: np.ndarray, dictionary: np.ndarray, *, active_bits: int
) -> SparseCode:
    """OMP fit. tiles: (N, C). dictionary: (K, C). Returns SparseCode."""
    N = tiles.shape[0]
    K = dictionary.shape[0]
    # sklearn expects (n_features, n_samples) for D and (n_features, n_samples) for y
    # i.e. D as (C, K) and y as (C, N). Result: (K, N).
    D_T = dictionary.T.astype(np.float32, copy=False)  # (C, K)
    y = tiles.T.astype(np.float32, copy=False)         # (C, N)
    coefs = orthogonal_mp(D_T, y, n_nonzero_coefs=active_bits, precompute=True)
    coefs = coefs.T  # (N, K)

    indices = np.zeros((N, active_bits), dtype=np.int32)
    values = np.zeros((N, active_bits), dtype=np.float32)
    for i in range(N):
        nz = np.where(coefs[i] != 0)[0]
        # OMP may return fewer than active_bits non-zeros for easy tiles; pad with 0s.
        nz = nz[:active_bits]
        indices[i, : nz.size] = nz
        values[i, : nz.size] = coefs[i, nz]
    return SparseCode(
        indices=indices,
        signs=np.sign(values).astype(np.int8),
        coefs=values,
        residual_indices=np.zeros((N, 0), dtype=np.int32),
        residual_coefs=np.zeros((N, 0), dtype=np.float16),
        code_type="real",
    )


def encode_binary(
    tiles: np.ndarray, dictionary: np.ndarray, *, active_bits: int
) -> SparseCode:
    """Binary code: take real OMP, threshold to top-k by |coef|, set ±1.

    This is the cleanest Numenta-style SDR: indices + signs, no magnitudes.
    """
    real = encode_real(tiles, dictionary, active_bits=active_bits)
    # signs are already correct from encode_real; coefs zeroed (not used at decode)
    return SparseCode(
        indices=real.indices,
        signs=real.signs,
        coefs=np.zeros_like(real.coefs),
        residual_indices=np.zeros_like(real.residual_indices),
        residual_coefs=np.zeros_like(real.residual_coefs),
        code_type="binary",
    )


def encode_hybrid(
    tiles: np.ndarray, dictionary: np.ndarray, *, active_bits: int
) -> SparseCode:
    """Binary primary + small real residual.

    Step 1: binary code (indices + signs) for the dominant pattern.
    Step 2: residual r = tile - decode(binary). Fit a real code on r with
            active_bits // 2 atoms (half the bits, half-precision storage).
    """
    primary = encode_binary(tiles, dictionary, active_bits=active_bits)
    primary_reconstruction = _decode(primary, dictionary)
    residual = tiles - primary_reconstruction
    residual_active = max(1, active_bits // 2)
    real_residual = encode_real(residual, dictionary, active_bits=residual_active)
    return SparseCode(
        indices=primary.indices,
        signs=primary.signs,
        coefs=np.zeros_like(primary.coefs),
        residual_indices=real_residual.indices.astype(np.int32),
        residual_coefs=real_residual.coefs.astype(np.float16),
        code_type="hybrid",
    )


def decode(code: SparseCode, dictionary: np.ndarray) -> np.ndarray:
    """Reconstruct tile batch from a SparseCode. Dispatch on code_type."""
    return _decode(code, dictionary)


def _decode(code: SparseCode, dictionary: np.ndarray) -> np.ndarray:
    N = code.indices.shape[0]
    C = dictionary.shape[1]
    out = np.zeros((N, C), dtype=np.float32)

    if code.code_type == "binary":
        for i in range(N):
            for j in range(code.indices.shape[1]):
                idx = code.indices[i, j]
                if idx == 0 and code.signs[i, j] == 0:
                    continue  # padding slot
                out[i] += float(code.signs[i, j]) * dictionary[idx]
    elif code.code_type == "real":
        for i in range(N):
            for j in range(code.indices.shape[1]):
                v = code.coefs[i, j]
                if v == 0.0:
                    continue
                out[i] += v * dictionary[code.indices[i, j]]
    elif code.code_type == "hybrid":
        # Primary binary contribution
        for i in range(N):
            for j in range(code.indices.shape[1]):
                if code.signs[i, j] == 0:
                    continue
                out[i] += float(code.signs[i, j]) * dictionary[code.indices[i, j]]
        # Residual real contribution
        for i in range(N):
            for j in range(code.residual_indices.shape[1]):
                v = float(code.residual_coefs[i, j])
                if v == 0.0:
                    continue
                out[i] += v * dictionary[code.residual_indices[i, j]]
    else:
        raise ValueError(f"Unknown code_type: {code.code_type}")
    return out


ENCODERS: dict[CodeType, callable] = {
    "binary": encode_binary,
    "real": encode_real,
    "hybrid": encode_hybrid,
}
