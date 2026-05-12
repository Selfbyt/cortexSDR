"""V4 combination experiments — pairing binary K-SVD with hierarchical encoding.

V4a (simple combination):
    Use V3's `fit_binary_ksvd` to learn a binary-aware dictionary, then encode
    each tile via V2's hierarchical residual procedure. The dictionary stays
    fixed; only the encoding strategy changes.

V4b (hierarchical K-SVD, proper joint fit):
    Generalize K-SVD to multi-stage binary codes. The reconstruction is
        Σ_l γ_l · (Σ_i sign_l,i · D[a_l,i])
    and atoms are updated to minimize the *sum-of-stages* squared error.

V3+ (V3 with more iterations):
    The vanilla V3 from exp_a_v2 but with 12 iterations instead of 6, to see
    if the attention-layer oscillation resolves.
"""
from __future__ import annotations

import numpy as np

from exp_a_v2_binary.variants import (
    BinaryCode,
    _greedy_binary_code,
    decode_binary,
    encode_hierarchical_binary,
    fit_binary_ksvd,
)


# ---------------------------------------------------------------------------
# V4a — simple combination
# ---------------------------------------------------------------------------
def fit_and_encode_v4a(
    tiles: np.ndarray,
    *,
    n_atoms: int,
    active_bits_per_stage: int,
    ksvd_iters: int = 6,
    n_stages: int = 3,
    stage_decay: float = 0.5,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, BinaryCode]:
    """V4a: K-SVD-fitted dictionary used in hierarchical encoding mode.

    The K-SVD fit uses k = n_stages * active_bits_per_stage total active bits
    (matching V4's effective budget) for a fair comparison.
    """
    total_active = n_stages * active_bits_per_stage
    D, _ksvd_code = fit_binary_ksvd(
        tiles, n_atoms=n_atoms, active_bits=total_active,
        n_iter=ksvd_iters, seed=seed, verbose=verbose,
    )
    # Re-encode hierarchically using D
    code = encode_hierarchical_binary(
        tiles, D,
        active_bits_per_stage=active_bits_per_stage,
        n_stages=n_stages, stage_decay=stage_decay,
    )
    return D, code


# ---------------------------------------------------------------------------
# V4b — hierarchical K-SVD (joint optimization across stages)
# ---------------------------------------------------------------------------
def _hierarchical_binary_code(
    tile: np.ndarray,
    dictionary: np.ndarray,
    *,
    active_bits_per_stage: int,
    stage_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matching pursuit across multiple stages with given per-stage scales.

    Returns (indices, signs), each (n_stages, active_bits_per_stage).
    """
    n_stages = stage_scales.shape[0]
    K = dictionary.shape[0]
    indices = np.zeros((n_stages, active_bits_per_stage), dtype=np.int32)
    signs = np.zeros((n_stages, active_bits_per_stage), dtype=np.int8)
    r = tile.astype(np.float32, copy=True)
    for l in range(n_stages):
        gamma = float(stage_scales[l])
        used_this_stage = set()
        for step in range(active_bits_per_stage):
            proj = dictionary @ r
            for j in used_this_stage:
                proj[j] = 0
            j = int(np.argmax(np.abs(proj)))
            if proj[j] == 0:
                break
            s = 1 if proj[j] > 0 else -1
            indices[l, step] = j
            signs[l, step] = s
            used_this_stage.add(j)
            r = r - gamma * s * dictionary[j]
    return indices, signs


def fit_hierarchical_ksvd(
    tiles: np.ndarray,
    *,
    n_atoms: int,
    active_bits_per_stage: int,
    n_stages: int = 3,
    stage_decay: float = 0.5,
    n_iter: int = 6,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, BinaryCode]:
    """V4b: K-SVD generalized to multi-stage binary codes.

    Forward: hierarchical binary MP across all stages.
    Atom update: for each atom j, optimal direction is found by considering
    all (tile, stage) pairs that use atom j. Each such (i, l) contributes
    a target equal to `signs * scale * (tile - reconstruction_without_atom_j)`.
    The optimal atom is the (scaled-and-summed) average of these targets.

    Compared to V3:
      - V3: single stage, atoms summed with ±1
      - V4b: M stages, stage l contributes γ_l × (Σ ±atoms). Atoms used in
             different stages contribute at different effective magnitudes.
    """
    N, C = tiles.shape
    rng = np.random.default_rng(seed)
    stage_scales = np.array([stage_decay ** l for l in range(n_stages)], dtype=np.float32)

    # Init: random tiles, unit-normalised then scaled by effective magnitude
    init_ids = rng.choice(N, size=n_atoms, replace=(N < n_atoms))
    D = tiles[init_ids].astype(np.float32, copy=True)
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    D = D / np.maximum(norms, 1e-9)
    total_active = active_bits_per_stage * n_stages
    tile_norm_mean = float(np.mean(np.linalg.norm(tiles, axis=1)))
    init_scale = tile_norm_mean / max(1.0, float(np.sqrt(total_active)))
    D = D * init_scale

    indices = np.zeros((N, n_stages, active_bits_per_stage), dtype=np.int32)
    signs = np.zeros((N, n_stages, active_bits_per_stage), dtype=np.int8)

    for it in range(n_iter):
        # --- Code step: hierarchical greedy MP for each tile ---
        for i in range(N):
            idx, sgn = _hierarchical_binary_code(
                tiles[i], D,
                active_bits_per_stage=active_bits_per_stage,
                stage_scales=stage_scales,
            )
            indices[i] = idx
            signs[i] = sgn

        # --- Atom update step ---
        for j in range(n_atoms):
            # Find all (tile, stage, slot) where this atom is used
            # mask[i, l, k] = True if indices[i, l, k] == j
            mask = (indices == j)
            if not mask.any():
                rid = int(rng.integers(N))
                D[j] = tiles[rid].astype(np.float32, copy=True)
                continue

            # For each user (i, l, k), accumulate s_l,k * gamma_l * (tile - recon_minus_j)
            # weighted by the squared effective magnitude (gamma^2)
            target_sum = np.zeros(C, dtype=np.float64)
            weight_sum = 0.0

            # Pre-compute per-tile reconstructions (excluding atom j to subtract residual cleanly)
            # We'll iterate over the users and compute incremental.
            user_locs = np.argwhere(mask)  # (n_users, 3)
            for (i, l, k) in user_locs:
                gamma = float(stage_scales[l])
                s = float(signs[i, l, k])
                # Full reconstruction
                recon = np.zeros(C, dtype=np.float64)
                for lp in range(n_stages):
                    gp = float(stage_scales[lp])
                    for kp in range(active_bits_per_stage):
                        if lp == l and kp == k:
                            continue  # exclude atom j at this slot
                        jp = int(indices[i, lp, kp])
                        sp = float(signs[i, lp, kp])
                        if sp == 0:
                            continue
                        recon += gp * sp * D[jp]
                target = tiles[i].astype(np.float64) - recon
                # Optimal contribution at this slot is (s * gamma * d_j), so
                # d_j should minimize ||target - s*gamma*d_j||^2.
                # Solution: d_j = (s*gamma) * target / (s*gamma)^2 = target / (s*gamma)
                # When multiple users contribute, weighted least-squares averaging:
                # d_j ≈ (Σ_u s_u gamma_u target_u) / (Σ_u gamma_u^2)
                target_sum += s * gamma * target
                weight_sum += gamma * gamma

            if weight_sum > 1e-12:
                D[j] = (target_sum / weight_sum).astype(np.float32)

        if verbose:
            # Compute reconstruction error
            code = BinaryCode(
                indices=np.transpose(indices, (1, 0, 2)).copy(),
                signs=np.transpose(signs, (1, 0, 2)).copy(),
                stage_scales=stage_scales.copy(),
                variant="hierarchical_ksvd",
            )
            recon = decode_binary(code, D)
            mse = float(np.mean((tiles - recon) ** 2))
            print(f"  hksvd iter {it+1}/{n_iter}  mse={mse:.4e}")

    code = BinaryCode(
        indices=np.transpose(indices, (1, 0, 2)).copy(),
        signs=np.transpose(signs, (1, 0, 2)).copy(),
        stage_scales=stage_scales.copy(),
        variant="hierarchical_ksvd",
    )
    return D, code
