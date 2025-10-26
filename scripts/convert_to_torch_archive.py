#!/usr/bin/env python3
"""
Convert a Hugging Face-style checkpoint (e.g., LLaMA) into a Torch state dict archive.

This avoids parsing raw .pkl/.pth pickle/zip formats in C++ by producing a
Torch archive that LibTorch can load via InputArchive.

Usage:
  python3 scripts/convert_to_torch_archive.py \
    --model-dir /home/mbishu/.llama/checkpoints/Llama3.1-8B-Instruct \
    --out llama8b_state.pt \
    --dtype fp16

Requirements:
  pip install torch transformers
"""

import argparse
import sys
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to Torch state archive")
    parser.add_argument("--model-dir", required=True, help="Path to HF model directory (checkpoint)")
    parser.add_argument("--out", required=True, help="Output .pt file path (Torch state dict archive)")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16",
                        help="Dtype for weights when loading (default: fp16)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Enable trust_remote_code for custom models")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", default=True,
                        help="Use Transformers low_cpu_mem_usage path (default: True)")
    return parser.parse_args()


def dtype_from_flag(flag: str):
    if flag == "fp32":
        return torch.float32
    if flag == "fp16":
        return torch.float16
    if flag == "bf16":
        return torch.bfloat16
    return torch.float16


def main() -> int:
    args = parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except Exception as e:
        print("[ERROR] transformers not available:", e, file=sys.stderr)
        return 2

    torch_dtype = dtype_from_flag(args.dtype)

    try:
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            device_map=None,
        )
        model.eval()
    except Exception as e:
        print("[ERROR] Failed to load model from directory:", args.model_dir, file=sys.stderr)
        print(e, file=sys.stderr)
        return 3

    try:
        state = model.state_dict()
        torch.save(state, args.out)
        print(f"[OK] Saved Torch state dict archive to: {args.out}")
    except Exception as e:
        print("[ERROR] Failed to save Torch state dict archive:", e, file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


