"""Model loading + weight tile extraction.

Tiles are the unit of sparse coding: each tile becomes a sparse-code +
dictionary-atoms reconstruction. Tile shape choice matters for storage
and for kernel layout downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

ROLE_PATTERNS: list[tuple[str, str]] = [
    ("attn-q", "q_proj"),
    ("attn-k", "k_proj"),
    ("attn-v", "v_proj"),
    ("attn-o", "o_proj"),
    ("mlp-gate", "gate_proj"),
    ("mlp-up", "up_proj"),
    ("mlp-down", "down_proj"),
    ("norm", "norm"),
    ("embed", "embed_tokens"),
    ("lm_head", "lm_head"),
]


def classify_layer_role(name: str) -> str:
    """Map a parameter name to a coarse role label used for role-aware dictionaries."""
    lower = name.lower()
    for role, pattern in ROLE_PATTERNS:
        if pattern in lower:
            return role
    return "other"


@dataclass
class LayerInfo:
    name: str
    role: str
    shape: tuple[int, ...]
    weight: np.ndarray  # FP32, on CPU

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))


def load_model_layers(
    model_name: str = DEFAULT_MODEL,
    *,
    role_filter: list[str] | None = None,
    max_layers: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[list[LayerInfo], AutoModelForCausalLM, AutoTokenizer]:
    """Load a HF causal-LM and return its weight tensors as numpy FP32 arrays.

    role_filter: keep only these roles (e.g. ["mlp-down", "attn-q"]).
    max_layers: cap on number of LayerInfo returned (useful for fast iteration).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Note: avoid `device_map=` to skip the accelerate dependency.
    # For Step 0 we're CPU-only on a small model — direct .to() is fine.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if device != "cpu":
        model = model.to(device)
    model.eval()

    layers: list[LayerInfo] = []
    for name, param in model.named_parameters():
        role = classify_layer_role(name)
        if role_filter is not None and role not in role_filter:
            continue
        if param.dim() < 2:  # skip 1-D biases / norms for now
            continue
        weight = param.detach().to(torch.float32).cpu().numpy()
        layers.append(LayerInfo(name=name, role=role, shape=tuple(weight.shape), weight=weight))
        if max_layers is not None and len(layers) >= max_layers:
            break
    return layers, model, tokenizer


def extract_tiles(
    weight: np.ndarray, tile_rows: int, tile_cols: int
) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
    """Yield ((row_block, col_block), tile_view) for each non-overlapping tile.

    Edge tiles smaller than (tile_rows, tile_cols) are skipped — handled by the caller
    via padding or a residual store. Step 0 uses only full tiles.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D weight, got shape {weight.shape}")
    R, C = weight.shape
    for i, r0 in enumerate(range(0, R - tile_rows + 1, tile_rows)):
        for j, c0 in enumerate(range(0, C - tile_cols + 1, tile_cols)):
            yield (i, j), weight[r0 : r0 + tile_rows, c0 : c0 + tile_cols]


def collect_tile_matrix(
    weight: np.ndarray, tile_rows: int, tile_cols: int
) -> np.ndarray:
    """Stack all tiles of a layer into an (n_tiles, tile_rows*tile_cols) matrix.

    This is the matrix passed to dictionary learning. Each row = one flattened tile.
    """
    flats = [tile.reshape(-1) for _, tile in extract_tiles(weight, tile_rows, tile_cols)]
    if not flats:
        return np.zeros((0, tile_rows * tile_cols), dtype=np.float32)
    return np.stack(flats, axis=0).astype(np.float32, copy=False)


def reassemble_from_tiles(
    tiles: np.ndarray, weight_shape: tuple[int, int], tile_rows: int, tile_cols: int
) -> np.ndarray:
    """Inverse of collect_tile_matrix. Edge regions left zero (Step 0 ignores them)."""
    R, C = weight_shape
    out = np.zeros((R, C), dtype=np.float32)
    n_per_row = (C // tile_cols)
    for k in range(tiles.shape[0]):
        i, j = divmod(k, n_per_row)
        out[i * tile_rows : (i + 1) * tile_rows, j * tile_cols : (j + 1) * tile_cols] = (
            tiles[k].reshape(tile_rows, tile_cols)
        )
    return out
