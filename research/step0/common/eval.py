"""Behavioral evaluation: swap a single layer's weights in a real model, measure
the perplexity hit on a small calibration set. Used to validate structural MSE
sweeps don't lie.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _wikitext_calibration_inputs(
    tokenizer: AutoTokenizer, *, n_samples: int = 8, seq_len: int = 512
) -> torch.Tensor:
    """Small wikitext-2 sample for behavioral eval. Cached by datasets lib."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(min(n_samples, ids.numel() // seq_len)):
        chunks.append(ids[i * seq_len : (i + 1) * seq_len])
    return torch.stack(chunks, dim=0)


@torch.no_grad()
def evaluate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    n_samples: int = 8,
    seq_len: int = 512,
    device: str = "cpu",
) -> float:
    """Plain perplexity over a small wikitext-2 slice. Uses HF's loss output."""
    model.eval()
    inputs = _wikitext_calibration_inputs(tokenizer, n_samples=n_samples, seq_len=seq_len)
    inputs = inputs.to(device)
    total_nll = 0.0
    total_tokens = 0
    for i in tqdm(range(inputs.shape[0]), desc="perplexity", leave=False):
        ids = inputs[i : i + 1]
        out = model(input_ids=ids, labels=ids)
        # HF returns mean NLL over predicted tokens (seq_len - 1)
        nll = out.loss.float().item() * (seq_len - 1)
        total_nll += nll
        total_tokens += seq_len - 1
    return math.exp(total_nll / total_tokens)


@torch.no_grad()
def per_layer_swap_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    target_param_name: str,
    reconstructed_weight: np.ndarray,
    n_samples: int = 8,
    seq_len: int = 512,
    device: str = "cpu",
) -> dict[str, float]:
    """Replace one parameter with its reconstruction, measure perplexity delta.

    Returns: {baseline_ppl, swapped_ppl, delta_ppl}. Original weight is restored
    after measurement.
    """
    # Find the parameter
    param = dict(model.named_parameters()).get(target_param_name)
    if param is None:
        raise KeyError(f"Param not found: {target_param_name}")
    if tuple(param.shape) != reconstructed_weight.shape:
        raise ValueError(
            f"Shape mismatch: param {tuple(param.shape)} vs reconstructed {reconstructed_weight.shape}"
        )

    baseline_ppl = evaluate_perplexity(
        model, tokenizer, n_samples=n_samples, seq_len=seq_len, device=device
    )

    original_data = param.data.clone()
    try:
        param.data.copy_(torch.from_numpy(reconstructed_weight).to(param.dtype).to(param.device))
        swapped_ppl = evaluate_perplexity(
            model, tokenizer, n_samples=n_samples, seq_len=seq_len, device=device
        )
    finally:
        param.data.copy_(original_data)

    return {
        "baseline_ppl": baseline_ppl,
        "swapped_ppl": swapped_ppl,
        "delta_ppl": swapped_ppl - baseline_ppl,
    }
