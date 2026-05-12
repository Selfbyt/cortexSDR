"""Experiment D.1 — Extended scaling test with Regime F-style protection.

Extends B.3's 5 sampled depths to 9 by adding depths {2, 7, 12, 17}. Reuses all
B.3 checkpoints and adds new fits with the same V4b adaptive hyperparameters.
After all fits, runs the multi-swap with Regime F-style protection:

    PROTECTED (kept at FP16):  MLPs at depths 0, 7, 20 (boundary + early)
    V4b APPLIED:               all attention at all 9 depths
                                MLPs at depths 2, 5, 10, 12, 15, 17

This is a partial scaling test (9 of 22 depths = 41% of model). True full-
model would take ~7 hours; the goal here is to confirm the C.2 protection
pattern holds at 9 depths and to estimate the full-model number.

Run:
    python -m exp_d1_full_scaling.swap
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common.eval import evaluate_perplexity
from common.model_io import (
    classify_layer_role,
    collect_tile_matrix,
    load_model_layers,
    reassemble_from_tiles,
)
from common.metrics import normalized_mse

from exp_a_v3_combined.combined import fit_hierarchical_ksvd
from exp_a_v2_binary.variants import decode_binary


TILE_ROWS, TILE_COLS = 128, 128

ROLE_CONFIG = {
    "attn-q":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8,  stage_decay=0.5),
    "attn-k":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8,  stage_decay=0.5),
    "attn-v":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8,  stage_decay=0.5),
    "attn-o":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8,  stage_decay=0.5),
    "mlp-gate": dict(n_atoms=512, active_bits_per_stage=8, n_stages=3, ksvd_iters=12, stage_decay=0.5),
    "mlp-up":   dict(n_atoms=512, active_bits_per_stage=8, n_stages=3, ksvd_iters=12, stage_decay=0.5),
    "mlp-down": dict(n_atoms=512, active_bits_per_stage=8, n_stages=3, ksvd_iters=12, stage_decay=0.5),
}

# 9 sampled depths total
ALL_DEPTHS = [0, 2, 5, 7, 10, 12, 15, 17, 20]
# Regime F-style protection: skip MLPs at boundary + early depths
PROTECTED_DEPTHS_FOR_MLP = {0, 7, 20}  # MLPs at these depths stay FP16

WEIGHT_PATTERNS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

PERPLEXITY_SAMPLES = 8
PERPLEXITY_SEQ_LEN = 512

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
# Reuse B.3 checkpoints + write new ones to the same directory
CHECKPOINT_DIR = RESULTS_DIR / "exp_b3_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def _ckpt_path(name: str) -> Path:
    safe = name.replace("/", "_").replace(".", "_")
    return CHECKPOINT_DIR / f"{safe}.npz"


def _load_ckpt(name: str) -> tuple[np.ndarray, float] | None:
    p = _ckpt_path(name)
    if not p.exists():
        return None
    try:
        data = np.load(p)
        return data["recon"], float(data["nmse"])
    except Exception:
        return None


def _save_ckpt(name: str, recon: np.ndarray, nmse: float) -> None:
    np.savez_compressed(_ckpt_path(name), recon=recon, nmse=nmse)


def reconstruct_layer_via_v4b(weight: np.ndarray, cfg: dict) -> tuple[np.ndarray, float, float]:
    tiles = collect_tile_matrix(weight, TILE_ROWS, TILE_COLS)
    n_atoms = int(cfg["n_atoms"])
    if tiles.shape[0] < n_atoms:
        n_atoms = max(64, tiles.shape[0])
        cfg = {**cfg, "n_atoms": n_atoms}
    t0 = time.perf_counter()
    D, code = fit_hierarchical_ksvd(
        tiles,
        n_atoms=n_atoms,
        active_bits_per_stage=int(cfg["active_bits_per_stage"]),
        n_stages=int(cfg["n_stages"]),
        stage_decay=float(cfg["stage_decay"]),
        n_iter=int(cfg["ksvd_iters"]),
        verbose=False,
    )
    fit_time = time.perf_counter() - t0
    recon_tiles = decode_binary(code, D)
    nmse = float(normalized_mse(tiles, recon_tiles))

    R, C = weight.shape
    tr, tc = TILE_ROWS, TILE_COLS
    n_full_rows = (R // tr) * tr
    n_full_cols = (C // tc) * tc
    recon_full = weight.copy()
    if n_full_rows > 0 and n_full_cols > 0:
        recon_full[:n_full_rows, :n_full_cols] = reassemble_from_tiles(
            recon_tiles, (n_full_rows, n_full_cols), tr, tc
        )
    return recon_full, nmse, fit_time


@torch.no_grad()
def swap_and_measure(model, tokenizer, swaps: list[tuple[str, np.ndarray]]) -> float:
    name_to_param = dict(model.named_parameters())
    originals = []
    for name, new_w in swaps:
        p = name_to_param[name]
        originals.append((name, p.data.clone()))
        if tuple(p.shape) != new_w.shape:
            raise ValueError(f"shape mismatch on {name}")
        p.data.copy_(torch.from_numpy(new_w).to(p.dtype).to(p.device))
    try:
        ppl = evaluate_perplexity(
            model, tokenizer,
            n_samples=PERPLEXITY_SAMPLES, seq_len=PERPLEXITY_SEQ_LEN, device="cpu",
        )
    finally:
        for name, orig in originals:
            name_to_param[name].data.copy_(orig)
    return float(ppl)


def main():
    print("Loading TinyLlama-1.1B...")
    _, model, tokenizer = load_model_layers(role_filter=None, max_layers=1)
    name_to_param = dict(model.named_parameters())

    print("\nBaseline perplexity...")
    t0 = time.perf_counter()
    baseline_ppl = evaluate_perplexity(
        model, tokenizer,
        n_samples=PERPLEXITY_SAMPLES, seq_len=PERPLEXITY_SEQ_LEN, device="cpu",
    )
    print(f"  baseline = {baseline_ppl:.3f}  ({time.perf_counter()-t0:.1f}s)")

    # Build target list across all 9 depths × 7 patterns = 63 tensors
    targets = []
    for depth in ALL_DEPTHS:
        for pat in WEIGHT_PATTERNS:
            name = f"model.layers.{depth}.{pat}.weight"
            if name in name_to_param:
                targets.append((depth, name, name_to_param[name]))
    print(f"\n{len(targets)} candidate target tensors (9 depths × 7 modules)")

    # Fit any missing checkpoints
    print("\n=== V4b FITTING (cached + new) ===")
    t_fit_start = time.perf_counter()
    n_fitted = 0
    n_cached = 0
    for k, (depth, name, param) in enumerate(targets):
        role = classify_layer_role(name)
        cfg = dict(ROLE_CONFIG.get(role, ROLE_CONFIG["attn-q"]))

        cached = _load_ckpt(name)
        if cached is not None:
            n_cached += 1
            print(f"  [{k+1}/{len(targets)}] depth={depth} {role:<10} CACHED   "
                  f"nmse={cached[1]:.3f}", flush=True)
            continue

        weight = param.detach().to(torch.float32).cpu().numpy()
        try:
            recon, nmse, fit_time = reconstruct_layer_via_v4b(weight, cfg)
        except Exception as e:
            print(f"  [{k+1}/{len(targets)}] depth={depth} {name}  FAIL: {e}", flush=True)
            continue
        _save_ckpt(name, recon, nmse)
        n_fitted += 1
        elapsed = (time.perf_counter() - t_fit_start) / 60
        print(f"  [{k+1}/{len(targets)}] depth={depth} {role:<10} FIT      "
              f"shape={tuple(weight.shape)}  nmse={nmse:.3f}  fit={fit_time:.0f}s  "
              f"total={elapsed:.1f}min", flush=True)
    print(f"\nFitting done: {n_cached} cached + {n_fitted} new in "
          f"{(time.perf_counter()-t_fit_start)/60:.1f} min")

    # Build swap list: V4b for everything EXCEPT MLPs at protected depths
    print("\n=== APPLYING REGIME-F PROTECTION (MLPs at depths "
          f"{sorted(PROTECTED_DEPTHS_FOR_MLP)} stay FP16) ===")
    swaps = []
    skipped = []
    for depth, name, _ in targets:
        role = classify_layer_role(name)
        if role.startswith("mlp") and depth in PROTECTED_DEPTHS_FOR_MLP:
            skipped.append(name)
            continue
        ckpt = _load_ckpt(name)
        if ckpt is None:
            continue
        swaps.append((name, ckpt[0]))
    print(f"  V4b-swapped: {len(swaps)}    FP16-protected: {len(skipped)}    "
          f"total candidate: {len(targets)}")

    print("\n=== MULTI-LAYER SIMULTANEOUS SWAP ===")
    t0 = time.perf_counter()
    ppl = swap_and_measure(model, tokenizer, swaps)
    delta = ppl - baseline_ppl
    elapsed = time.perf_counter() - t0
    print(f"  baseline={baseline_ppl:.3f}  multi-swap={ppl:.3f}  delta=+{delta:.3f}  "
          f"({elapsed:.1f}s)")

    summary = {
        "config": {
            "tile_rows": TILE_ROWS, "tile_cols": TILE_COLS,
            "all_depths": ALL_DEPTHS,
            "protected_depths_for_mlp": sorted(PROTECTED_DEPTHS_FOR_MLP),
            "role_config": {k: v for k, v in ROLE_CONFIG.items()},
        },
        "baseline_ppl": baseline_ppl,
        "n_swapped": len(swaps),
        "n_protected": len(skipped),
        "multi_swap_ppl": ppl,
        "multi_swap_delta": delta,
    }
    out_path = RESULTS_DIR / "exp_d1_full_scaling.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults: {out_path}")

    print(f"\n=== EXTENDED SCALING VERDICT ===")
    print(f"baseline:                10.901")
    print(f"9-depth multi-swap:      {ppl:.3f}  (+{delta:.3f})")
    print(f"  with {len(swaps)} V4b + {len(skipped)} FP16-protected tensors")
    # Compare with B.3 (5 depths, Regime F-style was 26 swaps): 11.97 / +1.07
    # Linear extrapolation to all 22 depths
    if len(ALL_DEPTHS) > 0:
        scale_factor = 22.0 / len(ALL_DEPTHS)
        projected = delta * scale_factor
        print(f"linear projection (22-depth full model):  {baseline_ppl + projected:.2f}  "
              f"(+{projected:.2f})")
    if delta < 1.5:
        print("verdict: SHIP-READY (delta < 1.5 ppl at 9-depth swap)")
    elif delta < 3.0:
        print("verdict: GOOD (delta < 3.0 — may need full-model fit to confirm)")
    else:
        print("verdict: HYBRID HELPS BUT MORE PROTECTION NEEDED")


if __name__ == "__main__":
    main()
