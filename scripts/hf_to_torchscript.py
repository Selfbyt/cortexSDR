#!/usr/bin/env python3
"""
Complete workflow: Clone model from Hugging Face and convert to TorchScript.

This script handles:
1. Downloading/cloning models from Hugging Face Hub
2. Loading models from .pth, .safetensors, or standard HF format
3. Converting to TorchScript for deployment

Usage:
  # Clone and convert a model
  python3 scripts/hf_to_torchscript.py \
    --model-name microsoft/DialoGPT-medium \
    --output-dir ./converted_models \
    --format torchscript

  # Convert existing local model
  python3 scripts/hf_to_torchscript.py \
    --model-dir /path/to/local/model \
    --output-dir ./converted_models \
    --format torchscript

Requirements:
  pip install torch transformers safetensors huggingface_hub
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone HF model and convert to TorchScript")
    
    # Model source options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-name", help="Hugging Face model name (e.g., microsoft/DialoGPT-medium)")
    model_group.add_argument("--model-dir", help="Path to local model directory")
    
    # Output options
    parser.add_argument("--output-dir", required=True, help="Output directory for converted model")
    parser.add_argument("--format", choices=["torchscript", "state_dict", "both"], default="torchscript",
                        help="Output format (default: torchscript)")
    
    # Model loading options
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16",
                        help="Data type for model weights (default: fp16)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Enable trust_remote_code for custom models")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", default=True,
                        help="Use low CPU memory usage (default: True)")
    parser.add_argument("--device", default="cpu", help="Device to load model on (default: cpu)")
    
    # TorchScript specific options
    parser.add_argument("--example-inputs", nargs="+", type=int, default=[1, 512],
                        help="Example input shape for TorchScript tracing (default: [1, 512])")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Use strict mode for TorchScript compilation")
    
    # Download options
    parser.add_argument("--cache-dir", help="Cache directory for downloaded models")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download even if model exists in cache")
    
    return parser.parse_args()


def dtype_from_flag(flag: str) -> torch.dtype:
    """Convert string flag to torch dtype."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(flag, torch.float16)


def download_model_from_hf(model_name: str, cache_dir: Optional[str] = None, 
                          force_download: bool = False) -> str:
    """Download model from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not available. Install with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Downloading model: {model_name}")
    
    try:
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=False
        )
        print(f"[OK] Model downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"[ERROR] Failed to download model {model_name}: {e}", file=sys.stderr)
        sys.exit(1)


def load_model_from_directory(model_dir: str, torch_dtype: torch.dtype, 
                            trust_remote_code: bool, low_cpu_mem_usage: bool,
                            device: str) -> torch.nn.Module:
    """Load model from directory, handling different formats."""
    try:
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    except ImportError:
        print("[ERROR] transformers not available. Install with: pip install transformers", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loading model from: {model_dir}")
    
    try:
        # Load config
        config = AutoConfig.from_pretrained(
            model_dir, 
            trust_remote_code=trust_remote_code
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
            device_map=None,  # We'll move to device manually
        )
        
        # Move to specified device
        model = model.to(device)
        model.eval()
        
        print(f"[OK] Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_dir}: {e}", file=sys.stderr)
        sys.exit(1)


def create_example_inputs(example_shape: list, device: str) -> torch.Tensor:
    """Create example inputs for TorchScript tracing."""
    # Create dummy input tensor
    input_ids = torch.randint(0, 1000, example_shape, device=device, dtype=torch.long)
    return input_ids


def convert_to_torchscript(model: torch.nn.Module, example_inputs: torch.Tensor, 
                          strict: bool = True) -> torch.jit.ScriptModule:
    """Convert model to TorchScript using tracing."""
    print("[INFO] Converting model to TorchScript...")
    
    try:
        # Use torch.jit.trace for conversion
        traced_model = torch.jit.trace(model, example_inputs, strict=strict)
        
        # Optimize the traced model
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        print("[OK] Model converted to TorchScript successfully")
        return traced_model
        
    except Exception as e:
        print(f"[ERROR] Failed to convert to TorchScript: {e}", file=sys.stderr)
        sys.exit(1)


def save_model(model: Union[torch.nn.Module, torch.jit.ScriptModule], 
               output_path: str, format_type: str):
    """Save model in specified format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        if format_type == "torchscript":
            model.save(output_path)
            print(f"[OK] TorchScript model saved to: {output_path}")
        elif format_type == "state_dict":
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), output_path)
            else:
                # For TorchScript models, save the entire model
                torch.save(model, output_path)
            print(f"[OK] State dict saved to: {output_path}")
        else:
            raise ValueError(f"Unknown format: {format_type}")
            
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    args = parse_args()
    
    # Determine model directory
    if args.model_name:
        model_dir = download_model_from_hf(
            args.model_name, 
            cache_dir=args.cache_dir,
            force_download=args.force_download
        )
    else:
        model_dir = args.model_dir
        if not os.path.exists(model_dir):
            print(f"[ERROR] Model directory does not exist: {model_dir}", file=sys.stderr)
            return 1
    
    # Convert dtype
    torch_dtype = dtype_from_flag(args.dtype)
    
    # Load model
    model = load_model_from_directory(
        model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        device=args.device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model name for output files
    if args.model_name:
        model_name = args.model_name.replace("/", "_")
    else:
        model_name = os.path.basename(model_dir)
    
    # Convert and save based on format
    if args.format in ["torchscript", "both"]:
        # Create example inputs for tracing
        example_inputs = create_example_inputs(args.example_inputs, args.device)
        
        # Convert to TorchScript
        traced_model = convert_to_torchscript(model, example_inputs, args.strict)
        
        # Save TorchScript model
        torchscript_path = os.path.join(args.output_dir, f"{model_name}_torchscript.pt")
        save_model(traced_model, torchscript_path, "torchscript")
    
    if args.format in ["state_dict", "both"]:
        # Save state dict
        state_dict_path = os.path.join(args.output_dir, f"{model_name}_state_dict.pt")
        save_model(model, state_dict_path, "state_dict")
    
    print(f"[OK] Conversion completed. Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
