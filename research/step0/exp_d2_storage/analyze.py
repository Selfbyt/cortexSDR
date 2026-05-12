"""Experiment D.2 — Storage / compression analysis.

For each TinyLlama weight tensor under the locked V4b configuration (B.2
adaptive hyperparameters), compute:

  - FP32 baseline bytes (just for reference; we ship vs FP16 in practice)
  - FP16 baseline bytes (production-typical baseline)
  - V4b indices+signs bytes
  - V4b dictionary bytes (one role-shared dictionary across all layers of a role)
  - Hybrid storage = V4b for V4b'd layers + FP16 for protected layers

Outputs the full-model storage breakdown and compression ratios.

Run:
    python -m exp_d2_storage.analyze
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common.model_io import classify_layer_role, load_model_layers


TILE_ROWS, TILE_COLS = 128, 128

# Per-role V4b configuration (matches B.2 / exp_b3 / exp_d1)
ROLE_CONFIG = {
    "attn-q":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3),
    "attn-k":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3),
    "attn-v":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3),
    "attn-o":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3),
    "mlp-gate": dict(n_atoms=512, active_bits_per_stage=8, n_stages=3),
    "mlp-up":   dict(n_atoms=512, active_bits_per_stage=8, n_stages=3),
    "mlp-down": dict(n_atoms=512, active_bits_per_stage=8, n_stages=3),
}

# Protection regimes inspired by C.2 Regime F findings
# Layers in PROTECTED_DEPTHS get FP16 for the listed roles, V4b otherwise.
PROTECTION_REGIMES = {
    "no_protection": {"depths": set(), "roles": set()},
    "boundary_2": {  # protect depths 0,1, 20,21 for MLPs
        "depths": {0, 1, 20, 21},
        "roles": {"mlp-gate", "mlp-up", "mlp-down"},
    },
    "boundary_3": {  # protect depths 0,1,2, 19,20,21 for MLPs
        "depths": {0, 1, 2, 19, 20, 21},
        "roles": {"mlp-gate", "mlp-up", "mlp-down"},
    },
    "boundary_plus_5_18": {  # boundary_3 + add depth 5, 18 (Regime F extrapolated)
        "depths": {0, 1, 2, 5, 18, 19, 20, 21},
        "roles": {"mlp-gate", "mlp-up", "mlp-down"},
    },
}


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _bits_per_index(n_atoms: int) -> int:
    return max(1, int(np.ceil(np.log2(max(2, n_atoms)))))


def v4b_bytes_for_layer(shape: tuple[int, int], cfg: dict) -> dict[str, int]:
    """Per-tile indices+signs only (dictionary is amortized separately)."""
    R, C = shape
    n_full_rows = R // TILE_ROWS
    n_full_cols = C // TILE_COLS
    n_tiles = n_full_rows * n_full_cols
    if n_tiles == 0:
        # Tensor too small for full tiles — must fall back to FP16
        return {"n_tiles": 0, "indices_signs_bytes": 0, "padding_fp16_bytes": R * C * 2}

    K = cfg["n_atoms"]
    bits_per_active = _bits_per_index(K) + 1  # index + sign bit
    bits_per_tile = cfg["n_stages"] * cfg["active_bits_per_stage"] * bits_per_active
    bytes_per_tile = (bits_per_tile + 7) // 8
    indices_bytes = n_tiles * bytes_per_tile

    # Edge regions outside the full-tile grid stay as FP16 (preserved by reassemble_from_tiles)
    edge_elements = (R * C) - (n_full_rows * TILE_ROWS) * (n_full_cols * TILE_COLS)
    padding_bytes = edge_elements * 2  # FP16

    return {
        "n_tiles": n_tiles,
        "indices_signs_bytes": indices_bytes,
        "padding_fp16_bytes": padding_bytes,
    }


def role_dict_bytes(role: str) -> int:
    """Shared dictionary for this role — single instance across all layers."""
    cfg = ROLE_CONFIG[role]
    return cfg["n_atoms"] * (TILE_ROWS * TILE_COLS) * 4  # FP32 atoms


def main():
    print("Loading TinyLlama-1.1B (param shapes only)...")
    _, model, _ = load_model_layers(role_filter=None, max_layers=1)

    # Group parameters by role
    by_role: dict[str, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    total_params_fp32 = 0
    total_params_fp16 = 0
    layer_meta = {}  # name -> (role, shape, depth)
    for name, p in model.named_parameters():
        if p.dim() < 2:
            # 1-D params (biases, norms) — small, stored as FP32 in our pipeline for now
            shape = tuple(p.shape)
            elems = int(np.prod(shape))
            total_params_fp32 += elems * 4
            total_params_fp16 += elems * 2
            continue
        role = classify_layer_role(name)
        shape = tuple(p.shape)
        by_role[role].append((name, shape))
        total_params_fp32 += int(np.prod(shape)) * 4
        total_params_fp16 += int(np.prod(shape)) * 2
        # Try to parse depth from "model.layers.<N>." prefix
        depth = None
        if name.startswith("model.layers."):
            try:
                depth = int(name.split(".")[2])
            except (ValueError, IndexError):
                pass
        layer_meta[name] = {"role": role, "shape": shape, "depth": depth}

    print(f"\nModel storage at baselines:")
    print(f"  FP32:  {total_params_fp32 / 2**20:>9.1f} MB")
    print(f"  FP16:  {total_params_fp16 / 2**20:>9.1f} MB")

    print("\nWeight-tensor counts by role:")
    for role, items in sorted(by_role.items()):
        print(f"  {role:<12}  count={len(items)}  example_shape={items[0][1] if items else '-'}")

    # --- Storage for various regimes ---
    print("\n=== STORAGE BREAKDOWN PER REGIME ===")
    regime_results = []
    for regime_name, regime in PROTECTION_REGIMES.items():
        prot_depths = regime["depths"]
        prot_roles = regime["roles"]

        v4b_indices_bytes_total = 0
        v4b_padding_bytes_total = 0
        fp16_protected_bytes = 0
        v4b_layer_count = 0
        protected_layer_count = 0
        roles_used = set()

        # Per-layer accounting
        for name, meta in layer_meta.items():
            role = meta["role"]
            shape = meta["shape"]
            depth = meta["depth"]
            # Decide: V4b or FP16-protected?
            protected = (depth is not None and depth in prot_depths and role in prot_roles)
            if role not in ROLE_CONFIG:
                # Unknown role (e.g. embed, lm_head) → keep at FP16
                fp16_protected_bytes += int(np.prod(shape)) * 2
                protected_layer_count += 1
                continue
            if protected:
                fp16_protected_bytes += int(np.prod(shape)) * 2
                protected_layer_count += 1
            else:
                cfg = ROLE_CONFIG[role]
                bd = v4b_bytes_for_layer(shape, cfg)
                v4b_indices_bytes_total += bd["indices_signs_bytes"] + bd["padding_fp16_bytes"]
                # padding bytes already include FP16 of edge elements
                v4b_layer_count += 1
                roles_used.add(role)

        # Dictionary cost: one shared dict per role that's actually used
        dict_bytes = sum(role_dict_bytes(r) for r in roles_used)

        total_compressed = v4b_indices_bytes_total + fp16_protected_bytes + dict_bytes
        ratio_vs_fp32 = total_params_fp32 / total_compressed
        ratio_vs_fp16 = total_params_fp16 / total_compressed

        result = {
            "regime": regime_name,
            "protected_depths": sorted(prot_depths),
            "protected_roles": sorted(prot_roles),
            "v4b_layer_count": v4b_layer_count,
            "protected_layer_count": protected_layer_count,
            "v4b_indices_bytes": v4b_indices_bytes_total,
            "fp16_protected_bytes": fp16_protected_bytes,
            "dict_bytes": dict_bytes,
            "total_compressed_bytes": total_compressed,
            "ratio_vs_fp32": ratio_vs_fp32,
            "ratio_vs_fp16": ratio_vs_fp16,
        }
        regime_results.append(result)

        print(f"\nRegime: {regime_name}")
        print(f"  protected_depths={sorted(prot_depths) or 'none'}  protected_roles={sorted(prot_roles) or 'none'}")
        print(f"  V4b layers: {v4b_layer_count}    FP16-protected layers: {protected_layer_count}")
        print(f"  V4b indices+padding: {v4b_indices_bytes_total/2**20:>8.2f} MB")
        print(f"  Dictionary (role-shared): {dict_bytes/2**20:>8.2f} MB")
        print(f"  FP16 protected: {fp16_protected_bytes/2**20:>8.2f} MB")
        print(f"  TOTAL compressed: {total_compressed/2**20:>8.2f} MB")
        print(f"  Compression vs FP32: {ratio_vs_fp32:.2f}x")
        print(f"  Compression vs FP16: {ratio_vs_fp16:.2f}x")

    out_path = RESULTS_DIR / "exp_d2_storage.json"
    with open(out_path, "w") as f:
        json.dump({
            "total_params_fp32_bytes": total_params_fp32,
            "total_params_fp16_bytes": total_params_fp16,
            "regimes": regime_results,
        }, f, indent=2)
    print(f"\nResults: {out_path}")

    print("\n=== STORAGE SUMMARY ===")
    header = f"{'regime':<24}  {'V4b layers':>10}  {'total MB':>9}  {'vs FP32':>9}  {'vs FP16':>9}"
    print(header)
    print("-" * len(header))
    for r in regime_results:
        print(f"{r['regime']:<24}  {r['v4b_layer_count']:>10}  "
              f"{r['total_compressed_bytes']/2**20:>7.2f}MB  "
              f"{r['ratio_vs_fp32']:>7.2f}x  {r['ratio_vs_fp16']:>7.2f}x")


if __name__ == "__main__":
    main()
