"""Experiment B.2 — adaptive role-aware hyperparameters for V4b.

Exp B.1 exposed the asymmetry: attention layers reconstruct nearly perfectly at
K=256, k=15 (NMSE ~0.003) while MLP layers stall at NMSE ~0.63. This experiment
gives MLP more code budget while keeping attention compact, and re-runs the
perplexity eval to see if the multi-swap delta tightens.

Allocations:
- attention: K=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8 (same as B.1)
- mlp:       K=512, active_bits_per_stage=8, n_stages=3, ksvd_iters=12

Storage cost per tile:
- attention: 15 active bits across 3 stages, log2(256) bits/index = ~13 bytes
- mlp:       24 active bits across 3 stages, log2(512) bits/index = ~24 bytes

Plus dictionaries (one per role for the multi-swap layers).

Run:
    python -m exp_b2_adaptive.swap
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


# --- Config ---------------------------------------------------------------
TILE_ROWS, TILE_COLS = 128, 128

# Role-keyed hyperparameter map. Roles come from classify_layer_role().
ROLE_CONFIG: dict[str, dict[str, int | float]] = {
    "attn-q":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8,  stage_decay=0.5),
    "attn-k":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8,  stage_decay=0.5),
    "attn-v":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8,  stage_decay=0.5),
    "attn-o":   dict(n_atoms=256, active_bits_per_stage=5, n_stages=3, ksvd_iters=8,  stage_decay=0.5),
    "mlp-gate": dict(n_atoms=512, active_bits_per_stage=8, n_stages=3, ksvd_iters=12, stage_decay=0.5),
    "mlp-up":   dict(n_atoms=512, active_bits_per_stage=8, n_stages=3, ksvd_iters=12, stage_decay=0.5),
    "mlp-down": dict(n_atoms=512, active_bits_per_stage=8, n_stages=3, ksvd_iters=12, stage_decay=0.5),
}
DEFAULT_CONFIG = dict(n_atoms=256, active_bits_per_stage=5, n_stages=3,
                      ksvd_iters=8, stage_decay=0.5)

PERPLEXITY_SAMPLES = 8
PERPLEXITY_SEQ_LEN = 512

# Same target layers as B.1 for a clean A/B comparison
TARGET_LAYER_PATTERNS: list[tuple[str, int]] = [
    ("self_attn.q_proj", 0),
    ("self_attn.q_proj", 10),
    ("mlp.down_proj", 0),
    ("mlp.down_proj", 10),
    ("mlp.up_proj", 5),
    ("self_attn.o_proj", 5),
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _config_for(role: str) -> dict:
    cfg = ROLE_CONFIG.get(role, DEFAULT_CONFIG)
    return dict(cfg)


def _select_target_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    name_to_param = dict(model.named_parameters())
    chosen: list[tuple[str, torch.nn.Parameter]] = []
    for pattern, layer_idx in TARGET_LAYER_PATTERNS:
        candidate = f"model.layers.{layer_idx}.{pattern}.weight"
        if candidate in name_to_param:
            chosen.append((candidate, name_to_param[candidate]))
        else:
            print(f"WARN: param not found: {candidate}")
    return chosen


def reconstruct_layer_via_v4b(weight: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    tiles = collect_tile_matrix(weight, TILE_ROWS, TILE_COLS)
    n_atoms = int(cfg["n_atoms"])
    if tiles.shape[0] < n_atoms:
        raise ValueError(
            f"Layer has only {tiles.shape[0]} tiles; need >= {n_atoms} atoms. "
            f"Adjust ROLE_CONFIG."
        )
    D, code = fit_hierarchical_ksvd(
        tiles,
        n_atoms=n_atoms,
        active_bits_per_stage=int(cfg["active_bits_per_stage"]),
        n_stages=int(cfg["n_stages"]),
        stage_decay=float(cfg["stage_decay"]),
        n_iter=int(cfg["ksvd_iters"]),
        verbose=False,
    )
    recon_tiles = decode_binary(code, D)
    nmse = float(normalized_mse(tiles, recon_tiles))

    R, C = weight.shape
    tr, tc = TILE_ROWS, TILE_COLS
    n_full_rows = (R // tr) * tr
    n_full_cols = (C // tc) * tc
    recon_full = weight.copy()
    recon_full[:n_full_rows, :n_full_cols] = reassemble_from_tiles(
        recon_tiles, (n_full_rows, n_full_cols), tr, tc
    )
    return recon_full, nmse


@torch.no_grad()
def swap_and_measure(model, tokenizer, swaps: list[tuple[str, np.ndarray]]) -> float:
    name_to_param = dict(model.named_parameters())
    originals = []
    for name, new_w in swaps:
        p = name_to_param[name]
        originals.append((name, p.data.clone()))
        if tuple(p.shape) != new_w.shape:
            raise ValueError(f"shape mismatch on {name}: {tuple(p.shape)} vs {new_w.shape}")
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

    print("\nMeasuring baseline perplexity (no swaps)...")
    t0 = time.perf_counter()
    baseline_ppl = evaluate_perplexity(
        model, tokenizer,
        n_samples=PERPLEXITY_SAMPLES, seq_len=PERPLEXITY_SEQ_LEN, device="cpu",
    )
    print(f"  baseline ppl = {baseline_ppl:.3f}  ({time.perf_counter()-t0:.1f}s)")

    print("\nFinding target params + role-aware config...")
    target_params = _select_target_params(model)
    for name, p in target_params:
        role = classify_layer_role(name)
        cfg = _config_for(role)
        print(f"  {name}  role={role}  K={cfg['n_atoms']} k_per_stage={cfg['active_bits_per_stage']}")

    print("\n=== PHASE 1: per-layer V4b fit + single-layer swap ===")
    per_layer_results = []
    reconstructions: list[tuple[str, np.ndarray]] = []
    for name, param in target_params:
        weight = param.detach().to(torch.float32).cpu().numpy()
        role = classify_layer_role(name)
        cfg = _config_for(role)
        print(f"\n[{name}]  role={role}  shape={weight.shape}")
        t0 = time.perf_counter()
        recon, nmse = reconstruct_layer_via_v4b(weight, cfg)
        fit_time = time.perf_counter() - t0
        print(f"  V4b fit (K={cfg['n_atoms']}, k={cfg['active_bits_per_stage']}, "
              f"iters={cfg['ksvd_iters']}): {fit_time:.1f}s   nmse={nmse:.4f}")

        t0 = time.perf_counter()
        ppl_swap = swap_and_measure(model, tokenizer, [(name, recon)])
        eval_time = time.perf_counter() - t0
        delta = ppl_swap - baseline_ppl
        print(f"  ppl: baseline={baseline_ppl:.3f}  swapped={ppl_swap:.3f}  "
              f"delta=+{delta:.3f}  ({eval_time:.1f}s)")
        per_layer_results.append({
            "name": name, "role": role, "shape": list(weight.shape),
            "config": cfg, "nmse": nmse, "fit_time_s": fit_time,
            "ppl_swap": ppl_swap, "delta_ppl": delta,
        })
        reconstructions.append((name, recon))

    print("\n=== PHASE 2: multi-layer (all-at-once) swap ===")
    t0 = time.perf_counter()
    ppl_multi = swap_and_measure(model, tokenizer, reconstructions)
    multi_delta = ppl_multi - baseline_ppl
    print(f"  swapped {len(reconstructions)} layers simultaneously")
    print(f"  ppl: baseline={baseline_ppl:.3f}  multi-swap={ppl_multi:.3f}  "
          f"delta=+{multi_delta:.3f}  ({time.perf_counter()-t0:.1f}s)")

    summary = {
        "baseline_ppl": baseline_ppl,
        "role_config": ROLE_CONFIG,
        "per_layer": per_layer_results,
        "multi_swap": {"n_layers": len(reconstructions),
                       "ppl": ppl_multi, "delta_ppl": multi_delta},
    }
    out_path = RESULTS_DIR / "exp_b2_adaptive.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults: {out_path}")

    print("\n=== ADAPTIVE PERPLEXITY VERDICT ===")
    print(f"baseline (FP32):              {baseline_ppl:.3f}")
    print(f"multi-swap ({len(reconstructions)} layers):     {ppl_multi:.3f}  (+{multi_delta:.3f})")
    print(f"per-layer delta sum:         +{sum(r['delta_ppl'] for r in per_layer_results):.3f}")
    print(f"per-layer max delta:         +{max(r['delta_ppl'] for r in per_layer_results):.3f}")
    print()
    # Cross-reference with B.1
    print("Compared to B.1 uniform allocation (baseline-multi delta +1.347):")
    if multi_delta < 1.0:
        improvement = (1.347 - multi_delta) / 1.347 * 100
        print(f"  Multi-swap delta IMPROVED by {improvement:.0f}% "
              f"(+1.347 -> +{multi_delta:.3f})")
    elif multi_delta < 1.347:
        improvement = (1.347 - multi_delta) / 1.347 * 100
        print(f"  Multi-swap delta improved {improvement:.0f}% "
              f"(+1.347 -> +{multi_delta:.3f})")
    else:
        print(f"  Multi-swap delta WORSE (+1.347 -> +{multi_delta:.3f}) -- "
              f"hyperparameter increase didn't help")


if __name__ == "__main__":
    main()
