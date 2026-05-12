"""Experiment B.1 — behavioral perplexity check on V4b reconstructions.

Validates that the structural NMSE wins from exp_a_v3 translate to actual
task performance. For each chosen layer:

  1. Extract ALL tiles (no sampling — we need the full layer back)
  2. Fit hierarchical K-SVD (V4b) jointly on those tiles
  3. Decode all tiles into a reconstructed weight matrix
  4. Swap that reconstruction into the model
  5. Measure wikitext-2 perplexity vs FP32 baseline
  6. Restore original weight

Plus a multi-layer swap: all chosen layers swapped simultaneously to check
whether errors compound or stay bounded.

Run:
    python -m exp_b1_perplexity.swap
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
N_ATOMS = 256
ACTIVE_BITS_PER_STAGE = 5
N_STAGES = 3
STAGE_DECAY = 0.5
KSVD_ITERS = 8                    # a bit more than the 6 in exp_a_v3 to be safe
PERPLEXITY_SAMPLES = 8
PERPLEXITY_SEQ_LEN = 512

# Pick a representative mix of layers across roles + depth
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


def _select_target_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    """Find the actual parameter names matching our patterns."""
    name_to_param = dict(model.named_parameters())
    chosen: list[tuple[str, torch.nn.Parameter]] = []
    for pattern, layer_idx in TARGET_LAYER_PATTERNS:
        candidate = f"model.layers.{layer_idx}.{pattern}.weight"
        if candidate in name_to_param:
            chosen.append((candidate, name_to_param[candidate]))
        else:
            print(f"WARN: param not found: {candidate}")
    return chosen


def reconstruct_layer_via_v4b(weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Run V4b end-to-end on a layer's weight matrix. Return (reconstructed, nmse)."""
    tiles = collect_tile_matrix(weight, TILE_ROWS, TILE_COLS)
    if tiles.shape[0] < N_ATOMS:
        raise ValueError(
            f"Layer has only {tiles.shape[0]} tiles; need >= {N_ATOMS} atoms. "
            f"Reduce N_ATOMS or use smaller tiles."
        )
    D, code = fit_hierarchical_ksvd(
        tiles,
        n_atoms=N_ATOMS,
        active_bits_per_stage=ACTIVE_BITS_PER_STAGE,
        n_stages=N_STAGES,
        stage_decay=STAGE_DECAY,
        n_iter=KSVD_ITERS,
        verbose=False,
    )
    recon_tiles = decode_binary(code, D)
    nmse = float(normalized_mse(tiles, recon_tiles))

    # Reassemble into full weight matrix; edge tiles (not full size) stay zero,
    # so copy them from the original to avoid spurious damage from tiling edges.
    R, C = weight.shape
    tr, tc = TILE_ROWS, TILE_COLS
    n_full_rows = (R // tr) * tr
    n_full_cols = (C // tc) * tc
    recon_full = weight.copy()  # start from original to preserve edges
    recon_full[:n_full_rows, :n_full_cols] = reassemble_from_tiles(
        recon_tiles, (n_full_rows, n_full_cols), tr, tc
    )
    return recon_full, nmse


@torch.no_grad()
def swap_and_measure(
    model, tokenizer, swaps: list[tuple[str, np.ndarray]],
) -> float:
    """Replace specified params with the given reconstructions, measure
    perplexity, restore originals. Returns the swapped perplexity."""
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
            n_samples=PERPLEXITY_SAMPLES,
            seq_len=PERPLEXITY_SEQ_LEN,
            device="cpu",
        )
    finally:
        for name, orig in originals:
            name_to_param[name].data.copy_(orig)
    return float(ppl)


def main():
    print("Loading TinyLlama-1.1B...")
    _, model, tokenizer = load_model_layers(role_filter=None, max_layers=1)
    # Note: load_model_layers with max_layers=1 returns only one LayerInfo, but
    # the full model is loaded — we just discard the layer list.

    print("\nMeasuring baseline perplexity (no swaps)...")
    t0 = time.perf_counter()
    baseline_ppl = evaluate_perplexity(
        model, tokenizer,
        n_samples=PERPLEXITY_SAMPLES, seq_len=PERPLEXITY_SEQ_LEN, device="cpu",
    )
    print(f"  baseline ppl = {baseline_ppl:.3f}  ({time.perf_counter()-t0:.1f}s)")

    print(f"\nFinding target params...")
    target_params = _select_target_params(model)
    print(f"  matched {len(target_params)} params:")
    for name, p in target_params:
        print(f"    {name}  shape={tuple(p.shape)}")

    # Phase 1: per-layer reconstruction + single-layer swap
    print("\n=== PHASE 1: single-layer swaps ===")
    per_layer_results = []
    reconstructions: list[tuple[str, np.ndarray]] = []
    for name, param in target_params:
        weight = param.detach().to(torch.float32).cpu().numpy()
        print(f"\n[{name}] shape={weight.shape}")
        t0 = time.perf_counter()
        recon, nmse = reconstruct_layer_via_v4b(weight)
        fit_time = time.perf_counter() - t0
        print(f"  V4b fit: {fit_time:.1f}s   nmse={nmse:.4f}")

        t0 = time.perf_counter()
        ppl_swap = swap_and_measure(model, tokenizer, [(name, recon)])
        eval_time = time.perf_counter() - t0
        delta = ppl_swap - baseline_ppl
        print(f"  ppl: baseline={baseline_ppl:.3f}  swapped={ppl_swap:.3f}  "
              f"delta=+{delta:.3f}  ({eval_time:.1f}s)")
        per_layer_results.append({
            "name": name,
            "role": classify_layer_role(name),
            "shape": list(weight.shape),
            "nmse": nmse,
            "fit_time_s": fit_time,
            "ppl_swap": ppl_swap,
            "delta_ppl": delta,
        })
        reconstructions.append((name, recon))

    # Phase 2: all-at-once swap
    print("\n=== PHASE 2: multi-layer (all-at-once) swap ===")
    t0 = time.perf_counter()
    ppl_multi = swap_and_measure(model, tokenizer, reconstructions)
    eval_time = time.perf_counter() - t0
    multi_delta = ppl_multi - baseline_ppl
    print(f"  swapped {len(reconstructions)} layers simultaneously")
    print(f"  ppl: baseline={baseline_ppl:.3f}  multi-swap={ppl_multi:.3f}  "
          f"delta=+{multi_delta:.3f}  ({eval_time:.1f}s)")

    # Persist
    summary = {
        "config": {
            "tile_rows": TILE_ROWS, "tile_cols": TILE_COLS,
            "n_atoms": N_ATOMS,
            "active_bits_per_stage": ACTIVE_BITS_PER_STAGE,
            "n_stages": N_STAGES, "stage_decay": STAGE_DECAY,
            "ksvd_iters": KSVD_ITERS,
            "perplexity_samples": PERPLEXITY_SAMPLES,
            "perplexity_seq_len": PERPLEXITY_SEQ_LEN,
        },
        "baseline_ppl": baseline_ppl,
        "per_layer": per_layer_results,
        "multi_swap": {
            "n_layers": len(reconstructions),
            "ppl": ppl_multi,
            "delta_ppl": multi_delta,
        },
    }
    out_path = RESULTS_DIR / "exp_b1_perplexity.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {out_path}")

    print("\n=== PERPLEXITY VERDICT ===")
    print(f"baseline (FP32):          {baseline_ppl:.3f}")
    print(f"multi-swap ({len(reconstructions)} layers):  {ppl_multi:.3f}  (+{multi_delta:.3f})")
    print(f"per-layer delta sum:     +{sum(r['delta_ppl'] for r in per_layer_results):.3f}")
    print(f"per-layer max delta:     +{max(r['delta_ppl'] for r in per_layer_results):.3f}")
    print()
    if multi_delta < 2.0:
        print("Verdict: V4b reconstructions PRESERVE task quality (delta < 2.0 ppl)")
    elif multi_delta < 5.0:
        print("Verdict: V4b reconstructions COST ~modest quality (delta 2-5 ppl)")
    else:
        print(f"Verdict: V4b reconstructions DAMAGE quality (delta {multi_delta:.1f} ppl)")


if __name__ == "__main__":
    main()
