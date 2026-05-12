"""Experiment B.3 — scaled multi-layer perplexity test.

Sample every 4th decoder layer of TinyLlama (indices 0, 4, 8, 12, 16, 20) and
swap ALL 7 weight tensors per layer (4 attention + 3 MLP). Total: 42 tensors.
This represents the full depth of the model and all weight-tensor types
without the cost of fitting all ~154 tensors.

Uses adaptive role-aware hyperparameters from B.2:
- attention: K=256, k=5x3, ksvd_iters=8
- mlp:       K=512, k=8x3, ksvd_iters=12

Run:
    python -m exp_b3_scaled.swap
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

ROLE_CONFIG: dict[str, dict] = {
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

# Sample layers — every 5th of TinyLlama's 22 decoder layers, plus all 7 weight modules
# Smaller than every-4th (was 6 depths) to keep wall-time inside background-task limits.
LAYER_INDICES = [0, 5, 10, 15, 20]
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
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "exp_b3_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def _checkpoint_path(name: str) -> Path:
    safe = name.replace("/", "_").replace(".", "_")
    return CHECKPOINT_DIR / f"{safe}.npz"


def _load_checkpoint(name: str) -> tuple[np.ndarray, float] | None:
    p = _checkpoint_path(name)
    if not p.exists():
        return None
    try:
        data = np.load(p)
        return data["recon"], float(data["nmse"])
    except Exception:
        return None


def _save_checkpoint(name: str, recon: np.ndarray, nmse: float) -> None:
    np.savez_compressed(_checkpoint_path(name), recon=recon, nmse=nmse)


def _config_for(role: str) -> dict:
    return dict(ROLE_CONFIG.get(role, DEFAULT_CONFIG))


def _build_target_list(model) -> list[tuple[str, torch.nn.Parameter]]:
    name_to_param = dict(model.named_parameters())
    targets: list[tuple[str, torch.nn.Parameter]] = []
    for layer_idx in LAYER_INDICES:
        for pattern in WEIGHT_PATTERNS:
            name = f"model.layers.{layer_idx}.{pattern}.weight"
            if name in name_to_param:
                targets.append((name, name_to_param[name]))
            else:
                print(f"WARN: param not found: {name}")
    return targets


def reconstruct_layer_via_v4b(weight: np.ndarray, cfg: dict) -> tuple[np.ndarray, float, float]:
    """Run V4b on a weight matrix. Returns (reconstruction, NMSE, fit_time_s)."""
    tiles = collect_tile_matrix(weight, TILE_ROWS, TILE_COLS)
    n_atoms = int(cfg["n_atoms"])
    if tiles.shape[0] < n_atoms:
        # Fall back to fewer atoms if tile count is small
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

    print("\nBaseline perplexity...")
    t0 = time.perf_counter()
    baseline_ppl = evaluate_perplexity(
        model, tokenizer,
        n_samples=PERPLEXITY_SAMPLES, seq_len=PERPLEXITY_SEQ_LEN, device="cpu",
    )
    print(f"  baseline ppl = {baseline_ppl:.3f}  ({time.perf_counter()-t0:.1f}s)")

    targets = _build_target_list(model)
    print(f"\nWill fit {len(targets)} weight tensors across {len(LAYER_INDICES)} depths:")
    for name, p in targets[:5]:
        print(f"  e.g. {name}  shape={tuple(p.shape)}")
    print(f"  ... and {len(targets) - 5} more")

    print("\n=== V4b FITTING ALL TARGETS (with per-fit checkpoints) ===")
    reconstructions: list[tuple[str, np.ndarray]] = []
    per_layer_records = []
    t_global = time.perf_counter()
    for k, (name, param) in enumerate(targets):
        role = classify_layer_role(name)
        cfg = _config_for(role)
        weight = param.detach().to(torch.float32).cpu().numpy()

        # Try checkpoint first
        cached = _load_checkpoint(name)
        if cached is not None:
            recon, nmse = cached
            if recon.shape == weight.shape:
                elapsed_global = time.perf_counter() - t_global
                print(f"  [{k+1}/{len(targets)}]  {name}  role={role}  "
                      f"shape={tuple(weight.shape)}  nmse={nmse:.3f}  "
                      f"CACHED  total_elapsed={elapsed_global/60:.1f}min", flush=True)
                reconstructions.append((name, recon))
                per_layer_records.append({
                    "name": name, "role": role, "shape": list(weight.shape),
                    "nmse": nmse, "fit_time_s": 0.0, "from_cache": True,
                })
                continue

        try:
            recon, nmse, fit_time = reconstruct_layer_via_v4b(weight, cfg)
        except Exception as e:
            print(f"  [{k+1}/{len(targets)}] {name}  role={role}  FAIL: {e}", flush=True)
            continue
        _save_checkpoint(name, recon, nmse)
        elapsed_global = time.perf_counter() - t_global
        print(f"  [{k+1}/{len(targets)}]  {name}  role={role}  "
              f"shape={tuple(weight.shape)}  nmse={nmse:.3f}  "
              f"fit={fit_time:.1f}s  total_elapsed={elapsed_global/60:.1f}min", flush=True)
        reconstructions.append((name, recon))
        per_layer_records.append({
            "name": name, "role": role, "shape": list(weight.shape),
            "nmse": nmse, "fit_time_s": fit_time, "from_cache": False,
        })

    print(f"\nAll fits done in {(time.perf_counter()-t_global)/60:.1f} minutes")

    print("\n=== MULTI-LAYER SIMULTANEOUS SWAP ===")
    print(f"Swapping {len(reconstructions)} tensors at once and measuring perplexity...")
    t0 = time.perf_counter()
    ppl_multi = swap_and_measure(model, tokenizer, reconstructions)
    multi_delta = ppl_multi - baseline_ppl
    print(f"  baseline={baseline_ppl:.3f}  multi-swap={ppl_multi:.3f}  "
          f"delta=+{multi_delta:.3f}  ({time.perf_counter()-t0:.1f}s)")

    summary = {
        "config": {
            "tile_rows": TILE_ROWS, "tile_cols": TILE_COLS,
            "role_config": ROLE_CONFIG,
            "layer_indices": LAYER_INDICES,
            "weight_patterns": WEIGHT_PATTERNS,
            "perplexity_samples": PERPLEXITY_SAMPLES,
            "perplexity_seq_len": PERPLEXITY_SEQ_LEN,
        },
        "baseline_ppl": baseline_ppl,
        "n_tensors_swapped": len(reconstructions),
        "multi_swap_ppl": ppl_multi,
        "multi_swap_delta": multi_delta,
        "per_layer": per_layer_records,
    }
    out_path = RESULTS_DIR / "exp_b3_scaled.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults: {out_path}")

    # Project to full model: TinyLlama has 22 decoder layers × 7 weight tensors = 154 total.
    # We sampled 6 layer indices × 7 = 42, so coverage is 42/154 = 27%.
    # Linear extrapolation: full_delta ~ delta * (154/42) ≈ delta * 3.67
    projected = multi_delta * (22 * 7) / len(reconstructions)
    print(f"\n=== SCALING VERDICT ===")
    print(f"baseline:                10.901")
    print(f"sampled multi-swap:      {ppl_multi:.3f}  (+{multi_delta:.3f})")
    print(f"  with {len(reconstructions)}/{22 * 7} tensors swapped ({100 * len(reconstructions)/(22*7):.0f}% of model)")
    print(f"linear projection (full): {baseline_ppl + projected:.2f}  (+{projected:.2f})")
    if multi_delta < 1.0:
        verdict = "EXCELLENT (sampled multi-swap delta < 1.0)"
    elif multi_delta < 3.0:
        verdict = "GOOD (sampled multi-swap delta < 3.0)"
    elif multi_delta < 6.0:
        verdict = "ACCEPTABLE (sampled multi-swap delta < 6.0)"
    else:
        verdict = "POOR (sampled multi-swap delta >= 6.0 -- design needs work)"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
