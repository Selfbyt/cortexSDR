"""Experiment A — code-type sweep.

For a small set of representative layers, fit a per-layer dictionary, then
encode each tile under three code types (binary / real / hybrid) across a
sweep of (n_atoms, active_bits, tile size). Report reconstruction NMSE and
storage ratio.

This is the kill-criterion experiment: if binary NMSE is > 2× real NMSE
across the whole reasonable hyperparameter region, the binary thesis is dead.

Run:
    python -m exp_a_codes.sweep

Reads:  TinyLlama weights via HF transformers
Writes: results/exp_a_codes.json  + plot
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from common.model_io import (
    LayerInfo,
    collect_tile_matrix,
    load_model_layers,
)
from common.metrics import compression_ratio, normalized_mse

from .codes import CodeType, ENCODERS, decode
from .dictionary import learn_dictionary


# --- Sweep configuration -----------------------------------------------------
# Kept small so a full sweep fits in ~10 min on a laptop CPU at TinyLlama scale.
TILE_SHAPES = [(64, 64), (128, 128)]       # (rows, cols) — square tiles for now
N_ATOMS_GRID = [512, 1024]                  # dictionary atom counts
ACTIVE_BITS_GRID = [16, 32]                # k = active SDR bits
CODE_TYPES: list[CodeType] = ["binary", "real", "hybrid"]
SAMPLE_LAYER_ROLES = ["mlp-down", "attn-q"]  # representative — extend after first run
MAX_LAYERS = 2                              # one of each role for speed
MAX_TILES_PER_LAYER = 2048                  # cap to keep encode time bounded
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _sample_tiles(layer: LayerInfo, tile_rows: int, tile_cols: int, max_n: int) -> np.ndarray:
    """Collect tiles from a layer, randomly subsample if more than max_n."""
    tiles = collect_tile_matrix(layer.weight, tile_rows, tile_cols)
    if tiles.shape[0] > max_n:
        rng = np.random.default_rng(0)
        idx = rng.choice(tiles.shape[0], size=max_n, replace=False)
        tiles = tiles[idx]
    return tiles


def run_single(
    layer: LayerInfo,
    tile_rows: int,
    tile_cols: int,
    n_atoms: int,
    active_bits: int,
    code_type: CodeType,
) -> dict:
    tile_dim = tile_rows * tile_cols
    tiles = _sample_tiles(layer, tile_rows, tile_cols, MAX_TILES_PER_LAYER)
    if tiles.shape[0] < n_atoms:
        return {
            "skipped": True,
            "reason": f"not enough tiles ({tiles.shape[0]}) for {n_atoms} atoms",
        }

    t0 = time.perf_counter()
    D = learn_dictionary(
        tiles, n_atoms=n_atoms, transform_n_nonzero_coefs=active_bits
    )
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    code = ENCODERS[code_type](tiles, D, active_bits=active_bits)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconstructed = decode(code, D)
    t_decode = time.perf_counter() - t0

    nmse = normalized_mse(tiles, reconstructed)
    ratio = compression_ratio(
        original_numel=layer.numel,
        original_bytes_per_elem=4,
        dict_atoms=n_atoms,
        tile_cols=tile_dim,
        n_tiles=layer.numel // tile_dim,
        active_bits=active_bits,
        code_type=code_type,
    )

    return {
        "skipped": False,
        "layer_name": layer.name,
        "role": layer.role,
        "shape": list(layer.shape),
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "n_atoms": n_atoms,
        "active_bits": active_bits,
        "code_type": code_type,
        "n_tiles_sampled": int(tiles.shape[0]),
        "nmse": float(nmse),
        "compression_ratio": ratio["ratio"],
        "encoded_bytes": ratio["encoded_bytes"],
        "t_fit_s": t_fit,
        "t_encode_s": t_encode,
        "t_decode_s": t_decode,
    }


def main():
    print("Loading TinyLlama-1.1B weights...")
    layers, _, _ = load_model_layers(role_filter=SAMPLE_LAYER_ROLES, max_layers=MAX_LAYERS)
    print(f"  selected {len(layers)} layers:")
    for L in layers:
        print(f"    {L.name}  role={L.role}  shape={L.shape}")

    results = []
    total_runs = (
        len(layers) * len(TILE_SHAPES) * len(N_ATOMS_GRID) * len(ACTIVE_BITS_GRID) * len(CODE_TYPES)
    )
    pbar = tqdm(total=total_runs, desc="exp_a_codes")
    for layer in layers:
        for tile_rows, tile_cols in TILE_SHAPES:
            for n_atoms in N_ATOMS_GRID:
                for active_bits in ACTIVE_BITS_GRID:
                    for code_type in CODE_TYPES:
                        try:
                            r = run_single(layer, tile_rows, tile_cols, n_atoms, active_bits, code_type)
                        except Exception as e:
                            r = {
                                "skipped": True,
                                "reason": f"exception: {type(e).__name__}: {e}",
                                "layer_name": layer.name,
                                "tile_rows": tile_rows,
                                "tile_cols": tile_cols,
                                "n_atoms": n_atoms,
                                "active_bits": active_bits,
                                "code_type": code_type,
                            }
                        results.append(r)
                        pbar.update(1)
                        if not r["skipped"]:
                            pbar.set_postfix(
                                code=code_type, k=active_bits, nmse=f"{r['nmse']:.3f}"
                            )
    pbar.close()

    out_path = RESULTS_DIR / "exp_a_codes.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": {
            "tile_shapes": TILE_SHAPES,
            "n_atoms_grid": N_ATOMS_GRID,
            "active_bits_grid": ACTIVE_BITS_GRID,
            "code_types": CODE_TYPES,
            "max_tiles_per_layer": MAX_TILES_PER_LAYER,
        }}, f, indent=2)
    print(f"\nResults written: {out_path}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Print a kill-criterion summary: at each (layer, tile, n_atoms, k),
    show NMSE for each code_type and the binary/real ratio."""
    print("\n=== KILL-CRITERION SUMMARY ===")
    print("If binary NMSE / real NMSE > 2.0 across most cells, binary thesis fails.\n")

    by_cell: dict[tuple, dict[str, float]] = {}
    for r in results:
        if r.get("skipped"):
            continue
        key = (r["layer_name"], r["tile_rows"], r["tile_cols"], r["n_atoms"], r["active_bits"])
        by_cell.setdefault(key, {})[r["code_type"]] = r["nmse"]

    header = f"{'layer':<32} {'tile':>9} {'K':>5} {'k':>4}  {'bin':>8} {'hyb':>8} {'real':>8}  {'bin/real':>9}"
    print(header)
    print("-" * len(header))
    ratios = []
    for key, vals in sorted(by_cell.items()):
        layer_name, tr, tc, K, k = key
        bin_v = vals.get("binary", float("nan"))
        hyb_v = vals.get("hybrid", float("nan"))
        real_v = vals.get("real", float("nan"))
        ratio = (bin_v / real_v) if real_v and real_v > 0 else float("nan")
        if not np.isnan(ratio):
            ratios.append(ratio)
        print(
            f"{layer_name[-32:]:<32} {tr:>3}x{tc:<5} {K:>5} {k:>4}  "
            f"{bin_v:>8.3f} {hyb_v:>8.3f} {real_v:>8.3f}  {ratio:>9.2f}"
        )
    if ratios:
        print(f"\nMedian binary/real NMSE ratio: {np.median(ratios):.2f}")
        print(f"Max binary/real NMSE ratio:    {np.max(ratios):.2f}")
        verdict = "BINARY VIABLE" if np.median(ratios) < 2.0 else "BINARY FAILS -- REPLAN"
        print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
