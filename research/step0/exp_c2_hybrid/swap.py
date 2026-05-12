"""Experiment C.2 — hybrid FP16 protection of sensitive layers.

Re-uses the B.3 checkpoints (35 V4b reconstructions on disk). Tests several
"protection regimes" — which layers stay at FP16 baseline vs which get V4b.
Goal: identify the smallest set of FP16-protected layers that brings the
35-tensor multi-swap perplexity delta back into acceptable range (<1.5 ppl).

Regimes:
- A (baseline): swap all 35 V4b reconstructions [B.3 result for reference]
- B: skip depth 0 entirely (28 swaps, 4 sensitive layers protected)
- C: skip depth 20 entirely (28 swaps)
- D: skip both depth 0 AND depth 20 (21 swaps)
- E: skip only MLPs at depth 0 and 20 (29 swaps — attn kept everywhere)
- F: skip ALL MLPs at depth 0/5/20 (only depths 10/15 MLP swapped + all attn)

Run:
    python -m exp_c2_hybrid.swap
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common.eval import evaluate_perplexity
from common.model_io import classify_layer_role, load_model_layers


PERPLEXITY_SAMPLES = 8
PERPLEXITY_SEQ_LEN = 512

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "results" / "exp_b3_checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


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


def _checkpoint_path(name: str) -> Path:
    safe = name.replace("/", "_").replace(".", "_")
    return CHECKPOINT_DIR / f"{safe}.npz"


def _load_checkpoint(name: str) -> np.ndarray | None:
    p = _checkpoint_path(name)
    if not p.exists():
        return None
    try:
        return np.load(p)["recon"]
    except Exception:
        return None


def build_swap_list(
    *,
    skip_depths: set[int] = set(),
    skip_roles_at_depths: dict[int, set[str]] | None = None,
    skip_roles_global: set[str] = set(),
) -> list[tuple[str, np.ndarray]]:
    """Build swap list from B.3 checkpoints, with various skipping rules."""
    swaps: list[tuple[str, np.ndarray]] = []
    skip_at = skip_roles_at_depths or {}
    for d in LAYER_INDICES:
        for pat in WEIGHT_PATTERNS:
            if d in skip_depths:
                continue
            role_label = classify_layer_role(pat)
            if role_label in skip_roles_global:
                continue
            if d in skip_at and role_label in skip_at[d]:
                continue
            name = f"model.layers.{d}.{pat}.weight"
            recon = _load_checkpoint(name)
            if recon is None:
                print(f"  (no checkpoint, skipping) {name}")
                continue
            swaps.append((name, recon))
    return swaps


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


REGIMES = [
    {"id": "A", "desc": "all 35 swaps (B.3 baseline)",        "kwargs": {}},
    {"id": "B", "desc": "skip depth 0 entirely",              "kwargs": {"skip_depths": {0}}},
    {"id": "C", "desc": "skip depth 20 entirely",             "kwargs": {"skip_depths": {20}}},
    {"id": "D", "desc": "skip both depth 0 AND depth 20",     "kwargs": {"skip_depths": {0, 20}}},
    {"id": "E", "desc": "skip MLPs at depth 0 and 20",
        "kwargs": {"skip_roles_at_depths": {0: {"mlp-gate","mlp-up","mlp-down"},
                                             20: {"mlp-gate","mlp-up","mlp-down"}}}},
    {"id": "F", "desc": "skip MLPs at depth 0/5/20 (keep only middle MLPs + all attn)",
        "kwargs": {"skip_roles_at_depths": {0: {"mlp-gate","mlp-up","mlp-down"},
                                             5: {"mlp-gate","mlp-up","mlp-down"},
                                             20: {"mlp-gate","mlp-up","mlp-down"}}}},
]


def main():
    print("Loading TinyLlama-1.1B...")
    _, model, tokenizer = load_model_layers(role_filter=None, max_layers=1)

    print("\nBaseline perplexity (no swaps)...")
    t0 = time.perf_counter()
    baseline_ppl = evaluate_perplexity(
        model, tokenizer,
        n_samples=PERPLEXITY_SAMPLES, seq_len=PERPLEXITY_SEQ_LEN, device="cpu",
    )
    print(f"  baseline = {baseline_ppl:.3f}  ({time.perf_counter()-t0:.1f}s)")

    results = []
    for regime in REGIMES:
        print(f"\n=== Regime {regime['id']}: {regime['desc']} ===")
        swaps = build_swap_list(**regime["kwargs"])
        print(f"  swapping {len(swaps)} tensors")
        if not swaps:
            print("  (empty — skipping)")
            continue
        t0 = time.perf_counter()
        ppl = swap_and_measure(model, tokenizer, swaps)
        elapsed = time.perf_counter() - t0
        delta = ppl - baseline_ppl
        print(f"  baseline={baseline_ppl:.3f}  swapped={ppl:.3f}  delta=+{delta:.3f}  "
              f"({elapsed:.1f}s)")
        results.append({
            "regime": regime["id"], "desc": regime["desc"],
            "n_swapped": len(swaps), "ppl": ppl, "delta": delta, "elapsed_s": elapsed,
        })

    summary = {"baseline_ppl": baseline_ppl, "regimes": results}
    out_path = RESULTS_DIR / "exp_c2_hybrid.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {out_path}")

    print("\n=== HYBRID-PROTECTION SUMMARY ===")
    print("Goal: smallest 'FP16 protection set' that brings delta < 1.5 ppl.\n")
    header = f"{'regime':>3}  {'n':>4}  {'desc':<55}  {'ppl':>7}  {'delta':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        flag = " VIABLE" if r["delta"] < 1.5 else ""
        print(f"{r['regime']:>3}   {r['n_swapped']:>3}  {r['desc']:<55}  "
              f"{r['ppl']:>7.3f}  +{r['delta']:>6.3f}{flag}")


if __name__ == "__main__":
    main()
