# Step 0 — Cortex-Hybrid Feasibility Experiments

Purpose: validate the three novelty legs of the proposed compression scheme
**before** any C++ engine work. If the binary-SDR thesis fails reconstruction
quality at viable hyperparameters, we replan; the C++ kernel work is wasted
otherwise.

## Three parallel experiments

| Dir | Question being answered | Kill criterion |
|---|---|---|
| [exp_a_codes/](exp_a_codes/) | Binary vs hybrid vs real-valued sparse codes — does binary-SDR alone work? | Reconstruction MSE > 2× real-valued AQLM-class baseline at any reasonable (k, K) |
| [exp_b_dict/](exp_b_dict/) | Per-layer vs cross-layer-shared vs role-aware dictionaries — does sharing generalize? | Cross-layer reconstruction worse than per-layer by > 1.5× |
| [exp_c_kernel/](exp_c_kernel/) | Fused `D·(S·x)` inference — does the FLOP-reduction story hold? | < 20% wall-time reduction vs dense baseline at matching arithmetic |

## Iteration scale

- Step 0 design lock: **TinyLlama-1.1B** (~10 min calibration per experiment)
- Step 1 (after design lock): **Llama-2-7B** for paper-headline numbers

## Setup

```powershell
cd research/step0
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

Each experiment self-contained:

```powershell
python -m exp_a_codes.sweep
python -m exp_b_dict.sweep
python -m exp_c_kernel.benchmark
```

Results land in `results/` as JSON + plots.

## What this is NOT

- Not the production code path. C++ engine in `src/` is independent.
- Not a benchmark of the final system. Step 0 is design-validation only.
- Not paper-quality numbers. We re-run at 7B in Step 1 once the design is locked.
