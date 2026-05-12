"""Experiment C — fused inference kernel proof on synthetic data.

The math: W ≈ S @ D where S is (n_tiles, K) sparse and D is (K, tile_cols).
A standard matmul Y = W @ x costs 2 * n_tiles * tile_cols * input_cols FLOPs.
The fused form Y = S @ (D @ x) costs:
    precompute_dx:   2 * K * tile_cols * input_cols   (once per token block)
    per_tile_gather: active_bits * n_tiles * tile_rows  (adds-only for binary)

This experiment runs four implementations on synthetic data shaped like a real
LLM MLP block, and measures wall-time + FLOP count for each:

    1. dense_baseline    — torch float32 matmul
    2. reconstruct       — decode W from sparse code, then dense matmul (no win)
    3. fused_real        — S @ (D @ x), real-valued S
    4. fused_binary      — S @ (D @ x), binary S, adds-only inner loop

Reports wall-time and FLOP-cost-per-output-token. Synthetic so the "right"
W reconstruction quality is not assessed here — exp_a does that.

Run:
    python -m exp_c_kernel.benchmark
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common.metrics import dense_matmul_flops, sparse_decode_flops

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- Shapes representative of a Llama-7B MLP `down_proj` --------------------
# W: (output_dim, input_dim) = (4096, 11008) for Llama-2-7B.
# We benchmark at smaller sizes for fast iteration; scale-up extrapolation is linear.
SHAPES = [
    # (output_dim, input_dim, batch_tokens)
    (1024, 2752, 1),     # TinyLlama-class slice, single token
    (1024, 2752, 8),     # batch of 8 tokens (typical decode batch)
    (4096, 11008, 1),    # Llama-7B slice, single token
    (4096, 11008, 8),
]
TILE_ROWS, TILE_COLS = 64, 64
N_ATOMS = 1024
ACTIVE_BITS = 16
N_REPEAT = 5


def make_synthetic(output_dim: int, input_dim: int, batch_tokens: int):
    """Build a (D, sparse code S as dense indices, dense W reconstruction, x)."""
    rng = np.random.default_rng(0)
    D = rng.standard_normal((N_ATOMS, TILE_COLS)).astype(np.float32)
    n_row_tiles = output_dim // TILE_ROWS
    n_col_tiles = input_dim // TILE_COLS
    n_tiles = n_row_tiles * n_col_tiles

    # Real-valued sparse code: (n_tiles, K)
    S_real = np.zeros((n_tiles, N_ATOMS), dtype=np.float32)
    for t in range(n_tiles):
        idx = rng.choice(N_ATOMS, size=ACTIVE_BITS, replace=False)
        S_real[t, idx] = rng.standard_normal(ACTIVE_BITS).astype(np.float32)
    # Binary code: same indices, ±1 signs
    S_bin = np.sign(S_real).astype(np.float32)

    # Reconstruct W from the real code (this is the "true" W for benchmarking)
    # Each tile (TILE_ROWS rows wide) shares ONE atom set in this synthetic — that's
    # an over-simplification but fine for kernel benchmarking.
    # For benchmark fairness across all 4 paths, reconstruct W for the dense baseline:
    W = (S_real @ D).reshape(n_row_tiles, n_col_tiles, TILE_COLS)
    # arrange into (output_dim, input_dim)
    W_dense = np.zeros((output_dim, input_dim), dtype=np.float32)
    for r in range(n_row_tiles):
        for c in range(n_col_tiles):
            W_dense[r * TILE_ROWS : (r + 1) * TILE_ROWS, c * TILE_COLS : (c + 1) * TILE_COLS] = (
                W[r, c]
            )  # broadcast tile_col vector to TILE_ROWS rows (degenerate but kernel-correct)

    x = rng.standard_normal((input_dim, batch_tokens)).astype(np.float32)
    return D, S_real, S_bin, W_dense, x, n_row_tiles, n_col_tiles


# --- Implementations --------------------------------------------------------
def impl_dense(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    return W @ x


def impl_reconstruct(D, S, n_row_tiles, n_col_tiles, x):
    """Worst case: rebuild W from (S, D) then dense matmul. No win expected."""
    # Reconstruct (n_tiles, TILE_COLS) = S @ D
    W_tiles = S @ D
    # tile -> output_dim × input_dim (degenerate broadcast as in make_synthetic)
    output_dim = n_row_tiles * TILE_ROWS
    input_dim = n_col_tiles * TILE_COLS
    W = np.zeros((output_dim, input_dim), dtype=np.float32)
    for r in range(n_row_tiles):
        for c in range(n_col_tiles):
            W[r * TILE_ROWS : (r + 1) * TILE_ROWS, c * TILE_COLS : (c + 1) * TILE_COLS] = (
                W_tiles[r * n_col_tiles + c]
            )
    return W @ x


def impl_fused_real(D, S, n_row_tiles, n_col_tiles, x):
    """Y = S @ (D @ x_col_block) per col block, broadcast over row tiles.

    For each column block c, compute Dx_c = D @ x_col_block (K, batch).
    Then for each row block r, each tile (r, c) contributes its row of
    S @ Dx_c into output rows [r*TILE_ROWS : (r+1)*TILE_ROWS].

    This is the "real-valued shared-D" inference path.
    """
    output_dim = n_row_tiles * TILE_ROWS
    batch = x.shape[1]
    Y = np.zeros((output_dim, batch), dtype=np.float32)
    for c in range(n_col_tiles):
        x_block = x[c * TILE_COLS : (c + 1) * TILE_COLS]  # (TILE_COLS, batch)
        Dx = D @ x_block                                   # (K, batch) — precomputed once
        S_block = S[c * n_row_tiles : (c + 1) * n_row_tiles]  # (n_row_tiles, K) — careful indexing
        # NOTE: in our synthetic layout, tile index = r*n_col_tiles + c, so we need a strided gather.
        # Build the correct per-block tile set:
        tile_idxs = [r * n_col_tiles + c for r in range(n_row_tiles)]
        S_for_block = S[tile_idxs]                        # (n_row_tiles, K)
        contribution = S_for_block @ Dx                   # (n_row_tiles, batch)
        # broadcast contribution[r] to TILE_ROWS rows of Y (synthetic-only degeneracy)
        for r in range(n_row_tiles):
            Y[r * TILE_ROWS : (r + 1) * TILE_ROWS] += contribution[r][None, :]
    return Y


def impl_fused_binary(D, S_bin, n_row_tiles, n_col_tiles, x):
    """Same as fused_real but exploiting that S has only ±1 values.

    Per col block: precompute Dx; then for each row tile, find the active
    atom indices in the binary code and sum (D · x)_k contributions with
    only sign flips — no multiplications inside the inner loop.
    """
    output_dim = n_row_tiles * TILE_ROWS
    batch = x.shape[1]
    Y = np.zeros((output_dim, batch), dtype=np.float32)
    for c in range(n_col_tiles):
        x_block = x[c * TILE_COLS : (c + 1) * TILE_COLS]
        Dx = D @ x_block                                  # (K, batch)
        for r in range(n_row_tiles):
            tile_idx = r * n_col_tiles + c
            active = np.nonzero(S_bin[tile_idx])[0]       # k indices
            signs = S_bin[tile_idx, active]               # k signs ±1
            # Sum signed rows of Dx; adds + negations only
            contribution = (signs[:, None] * Dx[active]).sum(axis=0)  # (batch,)
            Y[r * TILE_ROWS : (r + 1) * TILE_ROWS] += contribution[None, :]
    return Y


def bench(fn, repeats: int = N_REPEAT) -> float:
    """Return median wall-time in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_shape(output_dim, input_dim, batch_tokens):
    D, S_real, S_bin, W, x, n_rt, n_ct = make_synthetic(output_dim, input_dim, batch_tokens)
    n_tiles = n_rt * n_ct

    dense_t = bench(lambda: impl_dense(W, x))
    recon_t = bench(lambda: impl_reconstruct(D, S_real, n_rt, n_ct, x))
    real_t = bench(lambda: impl_fused_real(D, S_real, n_rt, n_ct, x))
    bin_t = bench(lambda: impl_fused_binary(D, S_bin, n_rt, n_ct, x))

    # Sanity: fused_real and reconstruct must produce matching outputs (within fp noise)
    y_dense = impl_dense(W, x)
    y_real = impl_fused_real(D, S_real, n_rt, n_ct, x)
    correctness = float(np.linalg.norm(y_real - y_dense) / max(np.linalg.norm(y_dense), 1e-9))

    flops_dense = dense_matmul_flops(output_dim, input_dim, batch_tokens)
    flops_fused_real = sparse_decode_flops(
        tile_rows=TILE_ROWS, tile_cols=TILE_COLS,
        n_tiles=n_tiles, dict_atoms=N_ATOMS, active_bits=ACTIVE_BITS,
        code_type="real", input_dim=input_dim, output_dim=output_dim, batch=batch_tokens,
    )
    flops_fused_bin = sparse_decode_flops(
        tile_rows=TILE_ROWS, tile_cols=TILE_COLS,
        n_tiles=n_tiles, dict_atoms=N_ATOMS, active_bits=ACTIVE_BITS,
        code_type="binary", input_dim=input_dim, output_dim=output_dim, batch=batch_tokens,
    )

    return {
        "shape": [output_dim, input_dim, batch_tokens],
        "n_tiles": n_tiles,
        "wall_time_s": {
            "dense": dense_t,
            "reconstruct": recon_t,
            "fused_real": real_t,
            "fused_binary": bin_t,
        },
        "speedup_vs_dense": {
            "reconstruct": dense_t / recon_t,
            "fused_real": dense_t / real_t,
            "fused_binary": dense_t / bin_t,
        },
        "theoretical_flops": {
            "dense": flops_dense,
            "fused_real_total": flops_fused_real["total"],
            "fused_binary_total": flops_fused_bin["total"],
            "fused_real_reduction": 1.0 - flops_fused_real["total"] / flops_dense,
            "fused_binary_reduction": 1.0 - flops_fused_bin["total"] / flops_dense,
        },
        "correctness_relative_err_fused_real_vs_dense": correctness,
    }


def main():
    print(f"Synthetic kernel benchmark — {N_REPEAT} repeats each\n")
    results = []
    for shape in SHAPES:
        print(f"shape (out, in, batch) = {shape}")
        r = run_shape(*shape)
        results.append(r)
        wt = r["wall_time_s"]
        sp = r["speedup_vs_dense"]
        th = r["theoretical_flops"]
        print(f"  dense          {wt['dense']*1000:>7.2f} ms (1.00×)")
        print(f"  reconstruct    {wt['reconstruct']*1000:>7.2f} ms ({sp['reconstruct']:>4.2f}×)")
        print(
            f"  fused_real     {wt['fused_real']*1000:>7.2f} ms ({sp['fused_real']:>4.2f}×)  "
            f"FLOP reduction (theoretical): {th['fused_real_reduction']*100:.1f}%"
        )
        print(
            f"  fused_binary   {wt['fused_binary']*1000:>7.2f} ms ({sp['fused_binary']:>4.2f}×)  "
            f"FLOP reduction (theoretical): {th['fused_binary_reduction']*100:.1f}%"
        )
        print(f"  correctness (fused_real vs dense relative err): {r['correctness_relative_err_fused_real_vs_dense']:.2e}\n")

    out_path = RESULTS_DIR / "exp_c_kernel.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "config": {
                "tile_rows": TILE_ROWS,
                "tile_cols": TILE_COLS,
                "n_atoms": N_ATOMS,
                "active_bits": ACTIVE_BITS,
                "n_repeat": N_REPEAT,
            }
        }, f, indent=2)
    print(f"Results written: {out_path}")

    _summary(results)


def _summary(results: list[dict]) -> None:
    print("\n=== KERNEL FLOP-REDUCTION SUMMARY ===")
    print("Wall-time speedups in NumPy are NOT representative — NumPy is unoptimized for our pattern.")
    print("Theoretical FLOP reductions are the relevant signal until we write the C++ kernel.\n")

    bin_red = [r["theoretical_flops"]["fused_binary_reduction"] for r in results]
    real_red = [r["theoretical_flops"]["fused_real_reduction"] for r in results]
    print(f"Mean theoretical FLOP reduction (fused_real):   {100 * np.mean(real_red):.1f}%")
    print(f"Mean theoretical FLOP reduction (fused_binary): {100 * np.mean(bin_red):.1f}%")
    if np.mean(bin_red) >= 0.30:
        print("Verdict: >=30% FLOP reduction achievable -- speed story confirmed at theory level.")
    else:
        print("Verdict: <30% FLOP reduction at these hyperparameters -- tune (K, active_bits, tile size).")


if __name__ == "__main__":
    main()
