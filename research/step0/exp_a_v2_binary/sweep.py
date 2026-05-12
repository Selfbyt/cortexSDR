"""Experiment A.v2 — three non-codebook binary variants vs real-OMP baseline.

Variants:
- V1: signs_only_magnitude (rescale atoms to absorb mean |coef|, then signs only)
- V2: hierarchical_binary (3-stage residual binary, scales 1, 0.5, 0.25)
- V3: binary_ksvd (joint dictionary + binary code fitting, 8 iterations)
- baseline_real: standard real-valued OMP (for ratio comparison)

Kill criterion (per variant): NMSE / real-OMP NMSE < 2.0 to reopen binary thesis.

Run:
    python -m exp_a_v2_binary.sweep
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from common.model_io import LayerInfo, collect_tile_matrix, load_model_layers
from common.metrics import normalized_mse

from exp_a_codes.dictionary import learn_dictionary
from exp_a_codes.codes import encode_real, decode as decode_real

from .variants import (
    BinaryCode,
    absorb_magnitudes_into_atoms,
    decode_binary,
    encode_hierarchical_binary,
    encode_signs_only,
    fit_binary_ksvd,
)


# --- Sweep configuration --------------------------------------------------
TILE_ROWS, TILE_COLS = 128, 128  # exp_a showed bigger tiles compress better
N_ATOMS = 256                     # constrained by smallest layer's tile count
ACTIVE_BITS = 16                  # single-stage budget; hierarchical splits this
HIERARCHICAL_STAGES = 3
HIERARCHICAL_PER_STAGE = max(1, ACTIVE_BITS // HIERARCHICAL_STAGES)
KSVD_ITERS = 6
SAMPLE_LAYER_ROLES = ["mlp-down", "attn-q"]
MAX_LAYERS = 2
MAX_TILES_PER_LAYER = 700         # cap to whatever the smaller layer naturally has
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def sample_tiles(layer: LayerInfo) -> np.ndarray:
    tiles = collect_tile_matrix(layer.weight, TILE_ROWS, TILE_COLS)
    if tiles.shape[0] > MAX_TILES_PER_LAYER:
        rng = np.random.default_rng(hash(layer.name) & 0xFFFFFFFF)
        idx = rng.choice(tiles.shape[0], size=MAX_TILES_PER_LAYER, replace=False)
        tiles = tiles[idx]
    return tiles


def run_layer(layer: LayerInfo) -> dict:
    tiles = sample_tiles(layer)
    if tiles.shape[0] < N_ATOMS:
        return {"layer_name": layer.name, "skipped": True,
                "reason": f"too few tiles ({tiles.shape[0]}) for {N_ATOMS} atoms"}

    # Shared starting dictionary (standard real OMP fit)
    print(f"\n[{layer.name}] fitting shared dictionary ({N_ATOMS} atoms)...")
    t0 = time.perf_counter()
    D = learn_dictionary(tiles, n_atoms=N_ATOMS, transform_n_nonzero_coefs=ACTIVE_BITS)
    t_dict = time.perf_counter() - t0
    print(f"  dict fit: {t_dict:.1f}s")

    out = {"layer_name": layer.name, "role": layer.role,
           "shape": list(layer.shape), "n_tiles": int(tiles.shape[0]),
           "tile_dim": TILE_ROWS * TILE_COLS, "t_dict_fit_s": t_dict,
           "variants": {}}

    # --- Baseline: real-valued OMP -----
    print("  [baseline] real-valued OMP...")
    t0 = time.perf_counter()
    code_real = encode_real(tiles, D, active_bits=ACTIVE_BITS)
    recon_real = decode_real(code_real, D)
    nmse_real = normalized_mse(tiles, recon_real)
    t_real = time.perf_counter() - t0
    out["variants"]["baseline_real"] = {
        "nmse": float(nmse_real),
        "t_encode_s": t_real,
        "active_bits": ACTIVE_BITS,
        "n_stages": 1,
        "total_active_bits": ACTIVE_BITS,
    }
    print(f"    nmse={nmse_real:.3f}")

    # --- V1: signs-only + per-atom magnitudes -----
    print("  [V1] signs-only with absorbed atom magnitudes...")
    t0 = time.perf_counter()
    D_scaled = absorb_magnitudes_into_atoms(D, tiles, active_bits=ACTIVE_BITS)
    code_v1 = encode_signs_only(tiles, D_scaled, active_bits=ACTIVE_BITS)
    recon_v1 = decode_binary(code_v1, D_scaled)
    nmse_v1 = normalized_mse(tiles, recon_v1)
    t_v1 = time.perf_counter() - t0
    out["variants"]["signs_only_magnitude"] = {
        "nmse": float(nmse_v1),
        "t_encode_s": t_v1,
        "active_bits": ACTIVE_BITS,
        "n_stages": 1,
        "total_active_bits": ACTIVE_BITS,
        "ratio_vs_real": float(nmse_v1 / max(nmse_real, 1e-12)),
    }
    print(f"    nmse={nmse_v1:.3f}  ratio_vs_real={nmse_v1/nmse_real:.2f}x")

    # --- V2: hierarchical/residual binary -----
    print(f"  [V2] hierarchical binary ({HIERARCHICAL_STAGES} stages × "
          f"{HIERARCHICAL_PER_STAGE} bits)...")
    t0 = time.perf_counter()
    code_v2 = encode_hierarchical_binary(
        tiles, D,
        active_bits_per_stage=HIERARCHICAL_PER_STAGE,
        n_stages=HIERARCHICAL_STAGES, stage_decay=0.5,
    )
    recon_v2 = decode_binary(code_v2, D)
    nmse_v2 = normalized_mse(tiles, recon_v2)
    t_v2 = time.perf_counter() - t0
    out["variants"]["hierarchical_binary"] = {
        "nmse": float(nmse_v2),
        "t_encode_s": t_v2,
        "active_bits": HIERARCHICAL_PER_STAGE,
        "n_stages": HIERARCHICAL_STAGES,
        "total_active_bits": HIERARCHICAL_PER_STAGE * HIERARCHICAL_STAGES,
        "ratio_vs_real": float(nmse_v2 / max(nmse_real, 1e-12)),
    }
    print(f"    nmse={nmse_v2:.3f}  ratio_vs_real={nmse_v2/nmse_real:.2f}x")

    # --- V3: binary K-SVD (joint fit) -----
    print(f"  [V3] binary K-SVD ({KSVD_ITERS} iters)...")
    t0 = time.perf_counter()
    D_ksvd, code_v3 = fit_binary_ksvd(
        tiles, n_atoms=N_ATOMS, active_bits=ACTIVE_BITS,
        n_iter=KSVD_ITERS, verbose=True,
    )
    recon_v3 = decode_binary(code_v3, D_ksvd)
    nmse_v3 = normalized_mse(tiles, recon_v3)
    t_v3 = time.perf_counter() - t0
    out["variants"]["binary_ksvd"] = {
        "nmse": float(nmse_v3),
        "t_encode_s": t_v3,
        "active_bits": ACTIVE_BITS,
        "n_stages": 1,
        "total_active_bits": ACTIVE_BITS,
        "ratio_vs_real": float(nmse_v3 / max(nmse_real, 1e-12)),
    }
    print(f"    nmse={nmse_v3:.3f}  ratio_vs_real={nmse_v3/nmse_real:.2f}x")

    return out


def main():
    print("Loading TinyLlama-1.1B...")
    layers, _, _ = load_model_layers(role_filter=SAMPLE_LAYER_ROLES, max_layers=MAX_LAYERS)
    print(f"Selected {len(layers)} layers:")
    for L in layers:
        print(f"  {L.role:<10} {L.name}  {L.shape}")

    t_total0 = time.perf_counter()
    results = []
    for L in layers:
        try:
            results.append(run_layer(L))
        except Exception as e:
            results.append({"layer_name": L.name, "skipped": True,
                            "reason": f"exception: {type(e).__name__}: {e}"})
    elapsed = time.perf_counter() - t_total0
    print(f"\nTotal time: {elapsed:.1f}s")

    out_path = RESULTS_DIR / "exp_a_v2_binary.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": {
            "tile_rows": TILE_ROWS, "tile_cols": TILE_COLS,
            "n_atoms": N_ATOMS,
            "active_bits": ACTIVE_BITS,
            "hierarchical_stages": HIERARCHICAL_STAGES,
            "hierarchical_per_stage": HIERARCHICAL_PER_STAGE,
            "ksvd_iters": KSVD_ITERS,
            "max_tiles_per_layer": MAX_TILES_PER_LAYER,
        }}, f, indent=2)
    print(f"Results: {out_path}")

    _summary(results)


def _summary(results: list[dict]) -> None:
    print("\n=== BINARY VARIANTS SUMMARY ===")
    print("Kill criterion: ratio_vs_real < 2.0 keeps the binary thesis alive.\n")
    header = f"{'layer':<48} {'real':>8} {'V1':>10} {'V2':>10} {'V3':>10}"
    print(header)
    print("-" * len(header))

    by_variant: dict[str, list[float]] = {"signs_only_magnitude": [],
                                           "hierarchical_binary": [],
                                           "binary_ksvd": []}
    for r in results:
        if r.get("skipped"):
            continue
        v = r["variants"]
        real = v["baseline_real"]["nmse"]
        v1 = v["signs_only_magnitude"]
        v2 = v["hierarchical_binary"]
        v3 = v["binary_ksvd"]
        print(f"{r['layer_name'][-48:]:<48} {real:>8.3f}"
              f"  {v1['nmse']:>5.3f}({v1['ratio_vs_real']:>3.1f}x)"
              f"  {v2['nmse']:>5.3f}({v2['ratio_vs_real']:>3.1f}x)"
              f"  {v3['nmse']:>5.3f}({v3['ratio_vs_real']:>3.1f}x)")
        by_variant["signs_only_magnitude"].append(v1["ratio_vs_real"])
        by_variant["hierarchical_binary"].append(v2["ratio_vs_real"])
        by_variant["binary_ksvd"].append(v3["ratio_vs_real"])

    print()
    for name, ratios in by_variant.items():
        if not ratios:
            continue
        med = float(np.median(ratios))
        mx = float(np.max(ratios))
        verdict = "VIABLE" if med < 2.0 else "FAILS"
        print(f"  {name:<26}  median {med:>4.2f}x  max {mx:>4.2f}x  -> {verdict}")


if __name__ == "__main__":
    main()
