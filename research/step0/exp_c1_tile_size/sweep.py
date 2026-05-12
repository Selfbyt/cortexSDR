"""Experiment C.1 — smaller MLP tiles to reduce NMSE.

B.3 exposed that MLP NMSE ~0.27 (at 128x128 tiles, K=512) compounds super-
linearly across many simultaneous swaps. The ceiling is the binding constraint.
This experiment tests whether smaller tiles + bigger K reduces MLP NMSE enough
to make full-model perplexity acceptable.

Hypothesis: 64x64 tiles give 4x more tiles per layer, permitting K up to ~2048
on TinyLlama MLP layers. Combined with more active bits per stage, this should
drop MLP NMSE toward ~0.10-0.15.

Configs swept on mlp.down_proj.0 (the worst offender in B.3):

  C1a: 128x128, K=512,  k=8 x 3 stages, 12 KSVD iters  [B.2 baseline]
  C1b: 64x64,   K=1024, k=8 x 3 stages, 12 KSVD iters  [more atoms]
  C1c: 64x64,   K=2048, k=8 x 3 stages, 12 KSVD iters  [push K]
  C1d: 64x64,   K=1024, k=12 x 3 stages, 12 KSVD iters [more active bits]
  C1e: 64x64,   K=2048, k=12 x 3 stages, 12 KSVD iters [push both]

Also: storage cost per layer (bytes for indices + dictionary share).

Run:
    python -m exp_c1_tile_size.sweep
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common.model_io import collect_tile_matrix, load_model_layers
from common.metrics import normalized_mse

from exp_a_v2_binary.variants import decode_binary
from exp_a_v3_combined.combined import fit_hierarchical_ksvd


CONFIGS = [
    # name, tile_rows, tile_cols, n_atoms, k_per_stage, n_stages, ksvd_iters
    ("C1a baseline (128x128, K=512, k=24)",   128, 128,  512,  8, 3, 12),
    ("C1b (64x64,   K=1024, k=24)",            64,  64, 1024,  8, 3, 12),
    ("C1c (64x64,   K=2048, k=24)",            64,  64, 2048,  8, 3, 12),
    ("C1d (64x64,   K=1024, k=36)",            64,  64, 1024, 12, 3, 12),
    ("C1e (64x64,   K=2048, k=36)",            64,  64, 2048, 12, 3, 12),
]

TARGET_LAYER = "mlp.down_proj"
TARGET_LAYER_IDX = 0
STAGE_DECAY = 0.5

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def storage_bytes_per_layer(
    n_tiles: int, n_atoms: int, k_per_stage: int, n_stages: int,
    tile_size: int,
) -> dict[str, int]:
    """Compute storage breakdown for one layer's V4b representation.

    Dictionary is amortizable across many layers, but we report it standalone here.
    """
    bits_per_idx = max(1, int(np.ceil(np.log2(max(2, n_atoms)))))
    bits_per_active_slot = bits_per_idx + 1  # index + sign bit
    bits_per_tile = n_stages * k_per_stage * bits_per_active_slot
    bytes_per_tile = (bits_per_tile + 7) // 8
    storage_bytes = n_tiles * bytes_per_tile
    dict_bytes = n_atoms * tile_size * 4
    return {
        "indices_signs_bytes": storage_bytes,
        "dict_bytes": dict_bytes,
        "total_bytes_standalone": storage_bytes + dict_bytes,
    }


def main():
    print(f"Loading TinyLlama-1.1B  (target: model.layers.{TARGET_LAYER_IDX}.{TARGET_LAYER})...")
    _, model, _ = load_model_layers(role_filter=None, max_layers=1)
    name_to_param = dict(model.named_parameters())
    target_name = f"model.layers.{TARGET_LAYER_IDX}.{TARGET_LAYER}.weight"
    if target_name not in name_to_param:
        raise SystemExit(f"target not found: {target_name}")
    weight = name_to_param[target_name].detach().to(np.float32 if False else None).cpu().numpy()
    weight = weight.astype(np.float32)
    print(f"  weight shape={weight.shape}  numel={weight.size}  fp32_bytes={weight.nbytes/1024:.1f}KB")

    rows = []
    for name, tr, tc, n_atoms, k_per_stage, n_stages, n_iter in CONFIGS:
        print(f"\n=== {name} ===")
        tiles = collect_tile_matrix(weight, tr, tc)
        if tiles.shape[0] < n_atoms:
            print(f"  SKIP: {tiles.shape[0]} tiles < {n_atoms} atoms")
            rows.append({"config": name, "skipped": True,
                         "reason": f"{tiles.shape[0]} tiles < {n_atoms} atoms"})
            continue
        print(f"  tile_count={tiles.shape[0]}  tile_dim={tr*tc}  K={n_atoms}  "
              f"k={k_per_stage}x{n_stages}={k_per_stage*n_stages}  iters={n_iter}")
        t0 = time.perf_counter()
        D, code = fit_hierarchical_ksvd(
            tiles, n_atoms=n_atoms,
            active_bits_per_stage=k_per_stage,
            n_stages=n_stages, stage_decay=STAGE_DECAY,
            n_iter=n_iter, verbose=False,
        )
        fit_time = time.perf_counter() - t0
        recon = decode_binary(code, D)
        nmse = float(normalized_mse(tiles, recon))
        storage = storage_bytes_per_layer(
            n_tiles=tiles.shape[0], n_atoms=n_atoms,
            k_per_stage=k_per_stage, n_stages=n_stages,
            tile_size=tr*tc,
        )
        ratio_vs_fp32 = weight.nbytes / storage["total_bytes_standalone"]
        ratio_indices_only = weight.nbytes / storage["indices_signs_bytes"]
        row = {
            "config": name, "tile_rows": tr, "tile_cols": tc,
            "n_atoms": n_atoms, "k_per_stage": k_per_stage, "n_stages": n_stages,
            "ksvd_iters": n_iter, "n_tiles": int(tiles.shape[0]),
            "nmse": nmse, "fit_time_s": fit_time,
            "storage": storage,
            "ratio_vs_fp32_standalone": ratio_vs_fp32,
            "ratio_indices_only": ratio_indices_only,
        }
        print(f"  nmse={nmse:.4f}  fit={fit_time:.1f}s")
        print(f"  storage: indices={storage['indices_signs_bytes']/1024:.1f}KB  "
              f"dict={storage['dict_bytes']/1024:.1f}KB  total={storage['total_bytes_standalone']/1024:.1f}KB")
        print(f"  ratio vs FP32 (standalone): {ratio_vs_fp32:.1f}x   "
              f"(indices-only, dict amortized): {ratio_indices_only:.1f}x")
        rows.append(row)

    out_path = RESULTS_DIR / "exp_c1_tile_size.json"
    with open(out_path, "w") as f:
        json.dump({"target": target_name, "weight_shape": list(weight.shape),
                   "results": rows}, f, indent=2)
    print(f"\nResults: {out_path}")

    print("\n=== TILE-SIZE SUMMARY ===")
    print("Goal: drop MLP NMSE from B.3 baseline ~0.27 toward ~0.10-0.15.\n")
    header = f"{'config':<48}  {'nmse':>7}  {'storage_kb':>12}  {'ratio_amort':>12}  {'fit_s':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("skipped"):
            print(f"{r['config']:<48}  SKIPPED ({r['reason']})")
            continue
        print(f"{r['config']:<48}  {r['nmse']:>7.3f}  "
              f"{r['storage']['total_bytes_standalone']/1024:>10.1f}KB  "
              f"{r['ratio_indices_only']:>11.1f}x  {r['fit_time_s']:>7.1f}")


if __name__ == "__main__":
    main()
