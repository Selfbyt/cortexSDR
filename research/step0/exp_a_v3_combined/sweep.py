"""Experiment A.v3 — V2+V3 combinations.

Variants:
- baseline_real: real-valued OMP (control, same as exp_a_v2)
- V2: hierarchical_binary (3 stages × 5 bits, real-fit dict) [reference]
- V3: binary_ksvd (6 iters) [reference]
- V3+: binary_ksvd (12 iters) — does attention oscillation resolve?
- V4a: K-SVD-fit dict + hierarchical encoding (3 stages × 5 bits)
- V4b: hierarchical K-SVD (joint fit with stages)

Kill criterion: ratio_vs_real < 2.0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common.model_io import LayerInfo, collect_tile_matrix, load_model_layers
from common.metrics import normalized_mse

from exp_a_codes.codes import decode as decode_real
from exp_a_codes.codes import encode_real
from exp_a_codes.dictionary import learn_dictionary
from exp_a_v2_binary.variants import (
    decode_binary,
    encode_hierarchical_binary,
    fit_binary_ksvd,
)

from .combined import fit_and_encode_v4a, fit_hierarchical_ksvd


# --- Config ---------------------------------------------------------------
TILE_ROWS, TILE_COLS = 128, 128
N_ATOMS = 256
ACTIVE_BITS = 16
HIERARCHICAL_STAGES = 3
HIERARCHICAL_PER_STAGE = max(1, ACTIVE_BITS // HIERARCHICAL_STAGES)
KSVD_ITERS = 6
KSVD_PLUS_ITERS = 12
SAMPLE_LAYER_ROLES = ["mlp-down", "attn-q"]
MAX_LAYERS = 2
MAX_TILES_PER_LAYER = 700
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

    print(f"\n[{layer.name}] fitting shared real-fit dictionary ({N_ATOMS} atoms)...")
    t0 = time.perf_counter()
    D_real = learn_dictionary(tiles, n_atoms=N_ATOMS, transform_n_nonzero_coefs=ACTIVE_BITS)
    print(f"  real-fit dict: {time.perf_counter()-t0:.1f}s")

    out = {"layer_name": layer.name, "role": layer.role,
           "shape": list(layer.shape), "n_tiles": int(tiles.shape[0]),
           "variants": {}}

    # --- baseline ---
    print("  [baseline] real OMP...")
    code = encode_real(tiles, D_real, active_bits=ACTIVE_BITS)
    recon = decode_real(code, D_real)
    nmse_real = float(normalized_mse(tiles, recon))
    out["variants"]["baseline_real"] = {"nmse": nmse_real}
    print(f"    nmse={nmse_real:.3f}")

    # --- V2 (hierarchical on real-fit) ---
    print(f"  [V2] hierarchical ({HIERARCHICAL_STAGES} stages x {HIERARCHICAL_PER_STAGE} bits) "
          f"on real-fit dict...")
    code_v2 = encode_hierarchical_binary(
        tiles, D_real,
        active_bits_per_stage=HIERARCHICAL_PER_STAGE,
        n_stages=HIERARCHICAL_STAGES, stage_decay=0.5,
    )
    nmse_v2 = float(normalized_mse(tiles, decode_binary(code_v2, D_real)))
    out["variants"]["v2_hierarchical_real_dict"] = {
        "nmse": nmse_v2, "ratio_vs_real": nmse_v2 / max(nmse_real, 1e-12)
    }
    print(f"    nmse={nmse_v2:.3f}  ratio={nmse_v2/nmse_real:.2f}x")

    # --- V3 (binary K-SVD, 6 iters) ---
    print(f"  [V3] binary K-SVD ({KSVD_ITERS} iters)...")
    D_v3, code_v3 = fit_binary_ksvd(
        tiles, n_atoms=N_ATOMS, active_bits=ACTIVE_BITS,
        n_iter=KSVD_ITERS, verbose=True,
    )
    nmse_v3 = float(normalized_mse(tiles, decode_binary(code_v3, D_v3)))
    out["variants"]["v3_binary_ksvd"] = {
        "nmse": nmse_v3, "ratio_vs_real": nmse_v3 / max(nmse_real, 1e-12)
    }
    print(f"    nmse={nmse_v3:.3f}  ratio={nmse_v3/nmse_real:.2f}x")

    # --- V3+ (binary K-SVD, 12 iters) ---
    print(f"  [V3+] binary K-SVD ({KSVD_PLUS_ITERS} iters)...")
    D_v3plus, code_v3plus = fit_binary_ksvd(
        tiles, n_atoms=N_ATOMS, active_bits=ACTIVE_BITS,
        n_iter=KSVD_PLUS_ITERS, verbose=True, seed=1,
    )
    nmse_v3plus = float(normalized_mse(tiles, decode_binary(code_v3plus, D_v3plus)))
    out["variants"]["v3plus_binary_ksvd_12iter"] = {
        "nmse": nmse_v3plus, "ratio_vs_real": nmse_v3plus / max(nmse_real, 1e-12)
    }
    print(f"    nmse={nmse_v3plus:.3f}  ratio={nmse_v3plus/nmse_real:.2f}x")

    # --- V4a (K-SVD-fit dict + hierarchical encoding) ---
    print(f"  [V4a] K-SVD-fit dict + hierarchical "
          f"({HIERARCHICAL_STAGES} stages x {HIERARCHICAL_PER_STAGE} bits)...")
    D_v4a, code_v4a = fit_and_encode_v4a(
        tiles, n_atoms=N_ATOMS,
        active_bits_per_stage=HIERARCHICAL_PER_STAGE,
        ksvd_iters=KSVD_ITERS,
        n_stages=HIERARCHICAL_STAGES, stage_decay=0.5,
        verbose=True,
    )
    nmse_v4a = float(normalized_mse(tiles, decode_binary(code_v4a, D_v4a)))
    out["variants"]["v4a_ksvd_dict_hierarchical_enc"] = {
        "nmse": nmse_v4a, "ratio_vs_real": nmse_v4a / max(nmse_real, 1e-12)
    }
    print(f"    nmse={nmse_v4a:.3f}  ratio={nmse_v4a/nmse_real:.2f}x")

    # --- V4b (hierarchical K-SVD, joint fit) ---
    print(f"  [V4b] hierarchical K-SVD joint fit "
          f"({HIERARCHICAL_STAGES} stages x {HIERARCHICAL_PER_STAGE} bits, "
          f"{KSVD_ITERS} iters)...")
    D_v4b, code_v4b = fit_hierarchical_ksvd(
        tiles, n_atoms=N_ATOMS,
        active_bits_per_stage=HIERARCHICAL_PER_STAGE,
        n_stages=HIERARCHICAL_STAGES, stage_decay=0.5,
        n_iter=KSVD_ITERS, verbose=True,
    )
    nmse_v4b = float(normalized_mse(tiles, decode_binary(code_v4b, D_v4b)))
    out["variants"]["v4b_hierarchical_ksvd"] = {
        "nmse": nmse_v4b, "ratio_vs_real": nmse_v4b / max(nmse_real, 1e-12)
    }
    print(f"    nmse={nmse_v4b:.3f}  ratio={nmse_v4b/nmse_real:.2f}x")

    return out


def main():
    print("Loading TinyLlama-1.1B...")
    layers, _, _ = load_model_layers(role_filter=SAMPLE_LAYER_ROLES, max_layers=MAX_LAYERS)
    print(f"Selected {len(layers)} layers")

    t0 = time.perf_counter()
    results = [run_layer(L) for L in layers]
    print(f"\nTotal: {time.perf_counter()-t0:.1f}s")

    out_path = RESULTS_DIR / "exp_a_v3_combined.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": {
            "tile_rows": TILE_ROWS, "tile_cols": TILE_COLS,
            "n_atoms": N_ATOMS, "active_bits": ACTIVE_BITS,
            "hierarchical_stages": HIERARCHICAL_STAGES,
            "hierarchical_per_stage": HIERARCHICAL_PER_STAGE,
            "ksvd_iters": KSVD_ITERS, "ksvd_plus_iters": KSVD_PLUS_ITERS,
            "max_tiles_per_layer": MAX_TILES_PER_LAYER,
        }}, f, indent=2)
    print(f"Results: {out_path}")
    _summary(results)


def _summary(results: list[dict]) -> None:
    print("\n=== V2+V3 COMBINATION SUMMARY ===")
    print("Kill criterion: ratio_vs_real < 2.0 keeps the binary thesis alive.\n")
    var_order = [
        "v2_hierarchical_real_dict",
        "v3_binary_ksvd",
        "v3plus_binary_ksvd_12iter",
        "v4a_ksvd_dict_hierarchical_enc",
        "v4b_hierarchical_ksvd",
    ]
    header = f"{'layer':<48} {'real':>8}  "
    header += "  ".join(f"{name[:14]:>14}" for name in var_order)
    print(header)
    print("-" * len(header))

    by_var: dict[str, list[float]] = {v: [] for v in var_order}
    for r in results:
        if r.get("skipped"):
            continue
        real_nmse = r["variants"]["baseline_real"]["nmse"]
        cells = [f"{real_nmse:>8.3f}"]
        for v in var_order:
            d = r["variants"][v]
            cells.append(f"{d['nmse']:>5.3f}({d['ratio_vs_real']:>4.2f}x)")
            by_var[v].append(d["ratio_vs_real"])
        print(f"{r['layer_name'][-48:]:<48}  " + "  ".join(cells))

    print()
    print(f"{'variant':<32} {'median':>10} {'max':>10}  verdict")
    for name, ratios in by_var.items():
        if not ratios:
            continue
        med = float(np.median(ratios))
        mx = float(np.max(ratios))
        verdict = "VIABLE" if med < 2.0 else "FAILS"
        flag = " (BEATS REAL!)" if med < 1.0 else ""
        print(f"  {name:<30} {med:>8.2f}x {mx:>8.2f}x  {verdict}{flag}")


if __name__ == "__main__":
    main()
