#!/usr/bin/env python3
"""
Clone Mistral-7B-Instruct-v0.3 from Hugging Face and convert to TorchScript.

This script specifically handles the Mistral-7B-Instruct-v0.3 model which:
- Uses SafeTensors format (consolidated.safetensors)
- Has 7.25B parameters
- Supports function calling
- Uses BF16 precision by default

Usage:
  python3 scripts/clone_mistral.py --output-dir ./mistral_converted
"""

import argparse
import os
import sys
import torch
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone Mistral-7B-Instruct-v0.3 and convert to TorchScript")
    parser.add_argument("--output-dir", required=True, help="Output directory for converted model")
    parser.add_argument("--format", choices=["torchscript", "state_dict", "both"], default="torchscript",
                        help="Output format (default: torchscript)")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16",
                        help="Data type for model weights (default: bf16)")
    parser.add_argument("--device", default="cpu", help="Device to load model on (default: cpu)")
    parser.add_argument("--cache-dir", help="Cache directory for downloaded models")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download even if model exists in cache")
    parser.add_argument("--example-inputs", nargs="+", type=int, default=[1, 512],
                        help="Example input shape for TorchScript tracing (default: [1, 512])")
    return parser.parse_args()


def dtype_from_flag(flag: str) -> torch.dtype:
    """Convert string flag to torch dtype."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(flag, torch.bfloat16)


def download_mistral_model(cache_dir: str = None, force_download: bool = False) -> str:
    """Download Mistral-7B-Instruct-v0.3 from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not available. Install with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"[INFO] Downloading Mistral-7B-Instruct-v0.3 model...")
    print(f"[INFO] Model size: 7.25B parameters")
    print(f"[INFO] Format: SafeTensors")
    
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


def load_mistral_model(model_dir: str, torch_dtype: torch.dtype, device: str):
    """Load Mistral model from directory."""
    try:
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    except ImportError:
        print("[ERROR] transformers not available. Install with: pip install transformers", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loading Mistral model from: {model_dir}")
    
    try:
        # Load config
        config = AutoConfig.from_pretrained(model_dir)
        print(f"[INFO] Model config loaded: {config.model_type}")
        
        # Load model with SafeTensors support
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,  # We'll move to device manually
        )
        
        # Move to specified device
        model = model.to(device)
        model.eval()
        
        print(f"[OK] Mistral model loaded successfully on {device}")
        print(f"[INFO] Model dtype: {model.dtype}")
        print(f"[INFO] Model device: {next(model.parameters()).device}")
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to load Mistral model from {model_dir}: {e}", file=sys.stderr)
        sys.exit(1)


def create_example_inputs(example_shape: list, device: str) -> torch.Tensor:
    """Create example inputs for TorchScript tracing."""
    # Create dummy input tensor with appropriate vocabulary size for Mistral
    input_ids = torch.randint(0, 32000, example_shape, device=device, dtype=torch.long)
    return input_ids


def convert_to_torchscript(model: torch.nn.Module, example_inputs: torch.Tensor) -> torch.jit.ScriptModule:
    """Convert Mistral model to TorchScript using tracing."""
    print("[INFO] Converting Mistral model to TorchScript...")
    
    try:
        # Wrap model to return logits tensor only and avoid HF ModelOutput
        class _CausalLMWrapper(torch.nn.Module):
            def __init__(self, inner_model: torch.nn.Module):
                super().__init__()
                self.inner_model = inner_model

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                # Disable cache and dict outputs for TorchScript friendliness
                outputs = self.inner_model(input_ids=input_ids, use_cache=False, return_dict=False)
                # HF returns tuple (logits, ...)
                if isinstance(outputs, (tuple, list)):
                    return outputs[0]
                # Fallback if an object with .logits
                return outputs.logits

        wrapper = _CausalLMWrapper(model)
        wrapper.eval()

        # Use torch.jit.trace for conversion on the wrapper
        traced_model = torch.jit.trace(wrapper, example_inputs, strict=True)
        
        # Optimize the traced model
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        print("[OK] Mistral model converted to TorchScript successfully")
        return traced_model
        
    except Exception as e:
        print(f"[ERROR] Failed to convert Mistral to TorchScript: {e}", file=sys.stderr)
        print("[INFO] This might be due to dynamic behavior in the model. Try with different input shapes.", file=sys.stderr)
        sys.exit(1)


def save_model(model, output_path: str, format_type: str):
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
                torch.save(model, output_path)
            print(f"[OK] State dict saved to: {output_path}")
        else:
            raise ValueError(f"Unknown format: {format_type}")
            
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    args = parse_args()
    
    # Convert dtype
    torch_dtype = dtype_from_flag(args.dtype)
    
    # Download Mistral model
    model_dir = download_mistral_model(
        cache_dir=args.cache_dir,
        force_download=args.force_download
    )
    
    # Load model
    model = load_mistral_model(model_dir, torch_dtype, args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert and save based on format
    if args.format in ["torchscript", "both"]:
        # Create example inputs for tracing
        example_inputs = create_example_inputs(args.example_inputs, args.device)
        
        # Convert to TorchScript
        traced_model = convert_to_torchscript(model, example_inputs)
        
        # Save TorchScript model
        torchscript_path = os.path.join(args.output_dir, "mistral_7b_instruct_torchscript.pt")
        save_model(traced_model, torchscript_path, "torchscript")
    
    if args.format in ["state_dict", "both"]:
        # Save state dict
        state_dict_path = os.path.join(args.output_dir, "mistral_7b_instruct_state_dict.pt")
        save_model(model, state_dict_path, "state_dict")
    
    print(f"[OK] Mistral-7B-Instruct-v0.3 conversion completed!")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Model supports function calling and instruction following")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
