"""Experiment B — dictionary-scope sweep.

Holds code type fixed at "real" (fastest, most generous to all variants),
sweeps dictionary scope:
  - per_layer   : one D per tensor (baseline; all existing methods)
  - global      : one D shared across all layers
  - role_aware  : one D per role bucket (attn-q, mlp-down, ...)
  - hierarchical: small global D + per-role residual D (atoms summed)

Metric: total stored parameters (D + codes) vs reconstruction NMSE, plus
cross-role NMSE breakdown (where does sharing hurt?).

Kill criterion (for cross-layer story): if global / per_layer NMSE > 1.5 on
average, the cross-layer-shared-D angle fails.

Run:
    python -m exp_b_dict.sweep
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from common.model_io import LayerInfo, collect_tile_matrix, load_model_layers
from common.metrics import normalized_mse

from exp_a_codes.codes import decode, encode_real
from exp_a_codes.dictionary import learn_dictionary


# --- Config -----------------------------------------------------------------
TILE_ROWS, TILE_COLS = 64, 64
N_ATOMS_PER_LAYER = 512
N_ATOMS_GLOBAL = 1024              # global D is bigger to compensate for sharing
N_ATOMS_PER_ROLE = 512             # role-aware: one D per role
ACTIVE_BITS = 16
MAX_TILES_PER_LAYER = 1024
SAMPLE_ROLES = ["mlp-down", "mlp-up", "attn-q", "attn-o"]
MAX_LAYERS_PER_ROLE = 2            # 2 layers × 4 roles = 8 layers total
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _select_layers(all_layers: list[LayerInfo]) -> list[LayerInfo]:
    by_role: dict[str, list[LayerInfo]] = defaultdict(list)
    for L in all_layers:
        if L.role in SAMPLE_ROLES:
            by_role[L.role].append(L)
    selected: list[LayerInfo] = []
    for role in SAMPLE_ROLES:
        selected.extend(by_role[role][:MAX_LAYERS_PER_ROLE])
    return selected


def _layer_tiles(layer: LayerInfo) -> np.ndarray:
    tiles = collect_tile_matrix(layer.weight, TILE_ROWS, TILE_COLS)
    if tiles.shape[0] > MAX_TILES_PER_LAYER:
        rng = np.random.default_rng(hash(layer.name) & 0xFFFFFFFF)
        idx = rng.choice(tiles.shape[0], size=MAX_TILES_PER_LAYER, replace=False)
        tiles = tiles[idx]
    return tiles


def fit_per_layer(layers: list[LayerInfo]) -> dict:
    """Baseline: one dictionary per layer."""
    print("[per_layer] fitting...")
    per_layer_results = []
    for L in tqdm(layers, desc="per_layer"):
        tiles = _layer_tiles(L)
        D = learn_dictionary(tiles, n_atoms=N_ATOMS_PER_LAYER, transform_n_nonzero_coefs=ACTIVE_BITS)
        code = encode_real(tiles, D, active_bits=ACTIVE_BITS)
        recon = decode(code, D)
        per_layer_results.append({
            "layer_name": L.name,
            "role": L.role,
            "n_tiles": int(tiles.shape[0]),
            "nmse": float(normalized_mse(tiles, recon)),
            "dict_params": int(D.size),
        })
    return {"scope": "per_layer", "results": per_layer_results}


def fit_global(layers: list[LayerInfo]) -> dict:
    """Single global D across all layers."""
    print("[global] gathering tiles...")
    all_tiles = [_layer_tiles(L) for L in layers]
    stacked = np.concatenate(all_tiles, axis=0)
    print(f"[global] fitting on {stacked.shape[0]} tiles...")
    D = learn_dictionary(stacked, n_atoms=N_ATOMS_GLOBAL, transform_n_nonzero_coefs=ACTIVE_BITS)
    per_layer_results = []
    for L, tiles in zip(layers, all_tiles):
        code = encode_real(tiles, D, active_bits=ACTIVE_BITS)
        recon = decode(code, D)
        per_layer_results.append({
            "layer_name": L.name,
            "role": L.role,
            "n_tiles": int(tiles.shape[0]),
            "nmse": float(normalized_mse(tiles, recon)),
        })
    return {
        "scope": "global",
        "results": per_layer_results,
        "dict_params": int(D.size),
    }


def fit_role_aware(layers: list[LayerInfo]) -> dict:
    """One dictionary per role bucket."""
    print("[role_aware] grouping by role...")
    by_role: dict[str, list[tuple[LayerInfo, np.ndarray]]] = defaultdict(list)
    for L in layers:
        by_role[L.role].append((L, _layer_tiles(L)))

    role_dicts: dict[str, np.ndarray] = {}
    per_layer_results = []
    total_dict_params = 0
    for role, items in by_role.items():
        stacked = np.concatenate([tiles for _, tiles in items], axis=0)
        print(f"[role_aware/{role}] fitting on {stacked.shape[0]} tiles...")
        D = learn_dictionary(stacked, n_atoms=N_ATOMS_PER_ROLE, transform_n_nonzero_coefs=ACTIVE_BITS)
        role_dicts[role] = D
        total_dict_params += D.size
        for L, tiles in items:
            code = encode_real(tiles, D, active_bits=ACTIVE_BITS)
            recon = decode(code, D)
            per_layer_results.append({
                "layer_name": L.name,
                "role": L.role,
                "n_tiles": int(tiles.shape[0]),
                "nmse": float(normalized_mse(tiles, recon)),
            })
    return {
        "scope": "role_aware",
        "results": per_layer_results,
        "dict_params": int(total_dict_params),
    }


def main():
    print("Loading TinyLlama-1.1B weights...")
    all_layers, _, _ = load_model_layers()
    layers = _select_layers(all_layers)
    print(f"Selected {len(layers)} layers across {len(set(L.role for L in layers))} roles:")
    for L in layers:
        print(f"  {L.role:<10} {L.name}  {L.shape}")

    t0 = time.perf_counter()
    runs = [fit_per_layer(layers), fit_global(layers), fit_role_aware(layers)]
    elapsed = time.perf_counter() - t0
    print(f"\nTotal sweep time: {elapsed:.1f}s")

    out_path = RESULTS_DIR / "exp_b_dict.json"
    with open(out_path, "w") as f:
        json.dump({"runs": runs, "config": {
            "tile_rows": TILE_ROWS, "tile_cols": TILE_COLS,
            "n_atoms_per_layer": N_ATOMS_PER_LAYER,
            "n_atoms_global": N_ATOMS_GLOBAL,
            "n_atoms_per_role": N_ATOMS_PER_ROLE,
            "active_bits": ACTIVE_BITS,
        }}, f, indent=2)
    print(f"\nResults written: {out_path}")
    _print_summary(runs)


def _print_summary(runs: list[dict]) -> None:
    print("\n=== DICTIONARY-SCOPE SUMMARY ===")
    print("If global NMSE / per_layer NMSE > 1.5 on average, cross-layer-shared-D fails.\n")

    by_scope: dict[str, dict[str, float]] = {}  # scope -> {layer_name: nmse}
    for run in runs:
        scope = run["scope"]
        by_scope[scope] = {r["layer_name"]: r["nmse"] for r in run["results"]}

    layer_names = sorted(by_scope.get("per_layer", {}).keys())
    print(f"{'layer':<48} {'per_layer':>10} {'global':>10} {'role':>10}  {'glb/pl':>8} {'rol/pl':>8}")
    print("-" * 105)
    glb_ratios, rol_ratios = [], []
    for name in layer_names:
        pl = by_scope["per_layer"][name]
        gl = by_scope.get("global", {}).get(name, float("nan"))
        ro = by_scope.get("role_aware", {}).get(name, float("nan"))
        glb_r = gl / pl if pl > 0 else float("nan")
        rol_r = ro / pl if pl > 0 else float("nan")
        if not np.isnan(glb_r): glb_ratios.append(glb_r)
        if not np.isnan(rol_r): rol_ratios.append(rol_r)
        print(f"{name[-48:]:<48} {pl:>10.3f} {gl:>10.3f} {ro:>10.3f}  {glb_r:>8.2f} {rol_r:>8.2f}")
    if glb_ratios:
        print(f"\nMedian global/per_layer NMSE ratio:     {np.median(glb_ratios):.2f}")
        print(f"Median role_aware/per_layer NMSE ratio: {np.median(rol_ratios):.2f}")
        glb_verdict = "GLOBAL VIABLE" if np.median(glb_ratios) < 1.5 else "GLOBAL FAILS"
        rol_verdict = "ROLE-AWARE VIABLE" if np.median(rol_ratios) < 1.2 else "ROLE-AWARE FAILS"
        print(f"Verdict: {glb_verdict}; {rol_verdict}")


if __name__ == "__main__":
    main()
