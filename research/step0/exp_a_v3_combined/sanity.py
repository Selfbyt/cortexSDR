"""Sanity check: optimized fit_hierarchical_ksvd should give ~same NMSE as before
on the attn-q layer 0 (where unoptimized V4b previously reached NMSE ~0.025)."""
from __future__ import annotations

import time

from common.model_io import collect_tile_matrix, load_model_layers
from common.metrics import normalized_mse

from exp_a_v2_binary.variants import decode_binary
from exp_a_v3_combined.combined import fit_hierarchical_ksvd


def main():
    print("Loading TinyLlama-1.1B (attn-q layer 0)...")
    layers, _, _ = load_model_layers(role_filter=["attn-q"], max_layers=1)
    L = layers[0]
    print(f"  {L.name}  shape={L.shape}")

    tiles = collect_tile_matrix(L.weight, 128, 128)
    # MAX_TILES_PER_LAYER from exp_a_v3
    if tiles.shape[0] > 700:
        import numpy as np
        rng = np.random.default_rng(hash(L.name) & 0xFFFFFFFF)
        idx = rng.choice(tiles.shape[0], size=700, replace=False)
        tiles = tiles[idx]
    print(f"  using {tiles.shape[0]} tiles")

    print("Running optimized V4b (K=256, k=5x3, 6 iters)...")
    t0 = time.perf_counter()
    D, code = fit_hierarchical_ksvd(
        tiles, n_atoms=256, active_bits_per_stage=5, n_stages=3,
        stage_decay=0.5, n_iter=6, verbose=True,
    )
    elapsed = time.perf_counter() - t0

    recon = decode_binary(code, D)
    nmse = float(normalized_mse(tiles, recon))

    print(f"\nOptimized V4b: nmse={nmse:.4f}  time={elapsed:.1f}s")
    print(f"Previous V4b (exp_a_v3):  nmse=0.025  time=~75s")
    # 256 atoms with 256 tiles is the edge case from exp_a_v2 (heavy memorization);
    # in exp_a_v3 we had MAX_TILES_PER_LAYER=700 + N_ATOMS=256 + ACTIVE_BITS=5×3 = exact match.
    print(f"PASS" if nmse < 0.05 else f"REGRESSION (nmse>{0.05})")


if __name__ == "__main__":
    main()
