"""Three binary-only sparse-code variants — non-codebook paths.

All three produce a "pure SDR" representation: only indices and signs of active
atoms (or for hierarchical, multiple binary layers of those). No real-valued
coefficients survive at storage time.

V1 — signs_only_magnitude:
    Fit a standard real-valued OMP dictionary. Then absorb each atom's mean
    |coefficient| into the atom itself (no longer unit-norm). Encode with
    binary signs only. Decoder: sum of ±atoms.

V2 — hierarchical_binary:
    M binary stages with progressively smaller atom scales. Stage 1 encodes
    binary on the tile at scale gamma_1=1. Stage 2 encodes binary on the
    residual at scale gamma_2=0.5. Stage 3 at gamma_3=0.25. The dictionary
    is shared across stages; only the active-atom scale changes per stage.
    Decoder: Σ_l gamma_l · Σ_{i in active_l} sign_l,i · D[i]

V3 — binary_ksvd:
    Joint dictionary + binary-code fitting. At each iteration: (a) greedy
    matching pursuit assigning ±1 codes to each tile, (b) atom update step
    computing each atom as the SVD-1 direction of its assigned signed targets.
    No constraint on atom norm — atoms carry their learned magnitudes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import orthogonal_mp


# ---------------------------------------------------------------------------
# Shared encoded-representation type
# ---------------------------------------------------------------------------
@dataclass
class BinaryCode:
    """Storage for any of the three variants.

    indices:      (n_stages, n_tiles, active_bits) int32
    signs:        (n_stages, n_tiles, active_bits) int8  ±1
    stage_scales: (n_stages,) float32 — per-stage atom-scale multiplier (1.0 for V1/V3)
    """
    indices: np.ndarray
    signs: np.ndarray
    stage_scales: np.ndarray
    variant: str

    @property
    def n_stages(self) -> int:
        return int(self.indices.shape[0])

    @property
    def n_tiles(self) -> int:
        return int(self.indices.shape[1])

    @property
    def active_bits_per_stage(self) -> int:
        return int(self.indices.shape[2])

    @property
    def total_active_bits(self) -> int:
        return self.n_stages * self.active_bits_per_stage


# ---------------------------------------------------------------------------
# V1: signs-only with per-atom learned magnitudes
# ---------------------------------------------------------------------------
def absorb_magnitudes_into_atoms(
    dictionary: np.ndarray, tiles: np.ndarray, *, active_bits: int
) -> np.ndarray:
    """Rescale each atom by the mean |coefficient| it gets across all tiles.

    Run OMP once on `tiles` with the input `dictionary`. For each atom j,
    compute mu_j = mean over tiles where j was selected of |coef_j|.
    Return a new dictionary where each atom is scaled by mu_j.
    """
    D_T = dictionary.T.astype(np.float32, copy=False)  # (C, K)
    y = tiles.T.astype(np.float32, copy=False)         # (C, N)
    coefs = orthogonal_mp(D_T, y, n_nonzero_coefs=active_bits, precompute=True).T  # (N, K)

    K = dictionary.shape[0]
    mu = np.zeros(K, dtype=np.float32)
    for j in range(K):
        mask = coefs[:, j] != 0
        if mask.any():
            mu[j] = float(np.mean(np.abs(coefs[mask, j])))
        else:
            mu[j] = 1.0  # unused atom — keep at unit scale (avoids zero atoms)

    D_scaled = dictionary * mu[:, None]
    return D_scaled


def encode_signs_only(
    tiles: np.ndarray, dictionary_scaled: np.ndarray, *, active_bits: int
) -> BinaryCode:
    """OMP fit with the magnitude-absorbed dictionary, keep signs only."""
    D_T = dictionary_scaled.T.astype(np.float32, copy=False)
    y = tiles.T.astype(np.float32, copy=False)
    coefs = orthogonal_mp(D_T, y, n_nonzero_coefs=active_bits, precompute=True).T

    N = tiles.shape[0]
    indices = np.zeros((1, N, active_bits), dtype=np.int32)
    signs = np.zeros((1, N, active_bits), dtype=np.int8)
    for i in range(N):
        nz = np.where(coefs[i] != 0)[0][:active_bits]
        indices[0, i, : nz.size] = nz
        signs[0, i, : nz.size] = np.sign(coefs[i, nz]).astype(np.int8)
    return BinaryCode(indices=indices, signs=signs,
                      stage_scales=np.array([1.0], dtype=np.float32),
                      variant="signs_only_magnitude")


# ---------------------------------------------------------------------------
# V2: hierarchical / multi-stage binary
# ---------------------------------------------------------------------------
def encode_hierarchical_binary(
    tiles: np.ndarray,
    dictionary: np.ndarray,
    *,
    active_bits_per_stage: int,
    n_stages: int = 3,
    stage_decay: float = 0.5,
) -> BinaryCode:
    """Encode tiles as a sum of M binary SDRs with geometrically decaying scales.

    Stage 0: pick k atoms with ±1 signs that best fit the tile, scaled by gamma_0=1.
    Stage l: pick k atoms with ±1 signs that best fit the residual, scaled by
             gamma_l = stage_decay**l.
    """
    N = tiles.shape[0]
    K, C = dictionary.shape
    indices = np.zeros((n_stages, N, active_bits_per_stage), dtype=np.int32)
    signs = np.zeros((n_stages, N, active_bits_per_stage), dtype=np.int8)
    scales = np.array([stage_decay ** l for l in range(n_stages)], dtype=np.float32)

    residuals = tiles.astype(np.float32, copy=True)
    for l in range(n_stages):
        gamma = scales[l]
        # Greedy matching pursuit on each tile's residual against gamma * D
        # Reduces to picking k atoms by max |<residual, D[j]>| / gamma magnitude
        # We compute correlation residuals @ D.T   (N, K)
        for i in range(N):
            r = residuals[i].copy()
            for step in range(active_bits_per_stage):
                proj = dictionary @ r  # (K,)
                # Prevent re-selecting the same atom within a stage
                if step > 0:
                    proj[indices[l, i, :step]] = 0
                j = int(np.argmax(np.abs(proj)))
                s = int(np.sign(proj[j])) if proj[j] != 0 else 1
                indices[l, i, step] = j
                signs[l, i, step] = s
                r = r - gamma * s * dictionary[j]
            residuals[i] = r

    return BinaryCode(indices=indices, signs=signs,
                      stage_scales=scales,
                      variant=f"hierarchical_binary_M{n_stages}")


# ---------------------------------------------------------------------------
# V3: binary K-SVD (joint fit)
# ---------------------------------------------------------------------------
def _greedy_binary_code(
    tile: np.ndarray, dictionary: np.ndarray, active_bits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matching pursuit with ±1 codes. Returns (indices, signs)."""
    r = tile.astype(np.float32, copy=True)
    K = dictionary.shape[0]
    indices = np.zeros(active_bits, dtype=np.int32)
    signs = np.zeros(active_bits, dtype=np.int8)
    used = np.zeros(K, dtype=bool)
    for step in range(active_bits):
        proj = dictionary @ r  # (K,)
        proj[used] = 0
        j = int(np.argmax(np.abs(proj)))
        if proj[j] == 0:
            break
        s = 1 if proj[j] > 0 else -1
        indices[step] = j
        signs[step] = s
        used[j] = True
        r = r - s * dictionary[j]
    return indices, signs


def fit_binary_ksvd(
    tiles: np.ndarray,
    *,
    n_atoms: int,
    active_bits: int,
    n_iter: int = 8,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, BinaryCode]:
    """Joint fit of dictionary D and binary codes (indices + ±1 signs).

    Returns (D, BinaryCode). D's atoms are NOT unit-norm — they carry magnitude.
    """
    N, C = tiles.shape
    rng = np.random.default_rng(seed)
    # Init: random subset of tiles (centred), normalised, then scaled by tile std
    init_ids = rng.choice(N, size=n_atoms, replace=(N < n_atoms))
    D = tiles[init_ids].astype(np.float32, copy=True)
    # Normalise per-atom to unit norm, then scale by mean tile-norm / sqrt(active_bits)
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    D = D / np.maximum(norms, 1e-9)
    tile_norm_mean = float(np.mean(np.linalg.norm(tiles, axis=1)))
    init_scale = tile_norm_mean / max(1.0, float(np.sqrt(active_bits)))
    D = D * init_scale

    indices = np.zeros((N, active_bits), dtype=np.int32)
    signs = np.zeros((N, active_bits), dtype=np.int8)
    history: list[float] = []

    for it in range(n_iter):
        # --- Code step: greedy binary MP for each tile
        for i in range(N):
            idx, sgn = _greedy_binary_code(tiles[i], D, active_bits)
            indices[i] = idx
            signs[i] = sgn

        # --- Atom update step: for each atom j, recompute as mean-of-signed-targets
        for j in range(n_atoms):
            # Which tiles use atom j? and with which sign?
            mask = (indices == j)  # (N, active_bits)
            users = np.where(mask.any(axis=1))[0]
            if users.size == 0:
                # Dead atom — replace with a random tile direction
                rid = int(rng.integers(N))
                D[j] = tiles[rid].astype(np.float32) - 0  # copy
                continue
            target_sum = np.zeros(C, dtype=np.float64)
            count = 0
            for i in users:
                slot = int(np.where(mask[i])[0][0])  # first hit
                s = float(signs[i, slot])
                # Reconstruction without atom j
                recon_minus_j = np.zeros(C, dtype=np.float64)
                for k in range(active_bits):
                    if k == slot:
                        continue
                    recon_minus_j += float(signs[i, k]) * D[int(indices[i, k])]
                target = tiles[i].astype(np.float64) - recon_minus_j
                target_sum += s * target
                count += 1
            new_atom = (target_sum / count).astype(np.float32)
            D[j] = new_atom

        # --- Diagnostic: reconstruction error this iter
        recon = decode_binary(BinaryCode(
            indices=indices[None, ...], signs=signs[None, ...],
            stage_scales=np.array([1.0], dtype=np.float32),
            variant="binary_ksvd",
        ), D)
        mse = float(np.mean((tiles - recon) ** 2))
        history.append(mse)
        if verbose:
            print(f"  ksvd iter {it+1}/{n_iter}  mse={mse:.4e}")

    code = BinaryCode(
        indices=indices[None, ...], signs=signs[None, ...],
        stage_scales=np.array([1.0], dtype=np.float32),
        variant="binary_ksvd",
    )
    return D, code


# ---------------------------------------------------------------------------
# Shared decoder
# ---------------------------------------------------------------------------
def decode_binary(code: BinaryCode, dictionary: np.ndarray) -> np.ndarray:
    """Decode BinaryCode → reconstructed tiles using shared dictionary D.

    For V1/V3 (n_stages=1, scale=1) this is plain ±atom sum.
    For V2 each stage contributes scale * (Σ ±atoms).
    """
    n_stages, N, k = code.indices.shape
    C = dictionary.shape[1]
    out = np.zeros((N, C), dtype=np.float32)
    for l in range(n_stages):
        gamma = float(code.stage_scales[l])
        for i in range(N):
            for j in range(k):
                idx = int(code.indices[l, i, j])
                s = int(code.signs[l, i, j])
                if s == 0:
                    continue
                out[i] += gamma * float(s) * dictionary[idx]
    return out
