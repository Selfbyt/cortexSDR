#!/usr/bin/env python3
"""
Download pre-trained ONNX models from Hugging Face Hub.

Many models on Hugging Face are already available in ONNX format.
This script helps you find and download them directly.

Usage:
  # Search for ONNX models
  python3 scripts/download_pretrained_onnx.py --search "mistral" --format onnx

  # Download specific ONNX model
  python3 scripts/download_pretrained_onnx.py \
    --model-name microsoft/DialoGPT-medium-onnx \
    --output-dir ./onnx_models

Requirements:
  pip install huggingface_hub transformers
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pre-trained ONNX models from Hugging Face")
    
    # Search or download options
    parser.add_argument("--search", help="Search for models with this term")
    parser.add_argument("--format", choices=["onnx", "safetensors", "pytorch"], default="onnx",
                        help="Model format to search for (default: onnx)")
    parser.add_argument("--model-name", help="Specific model name to download")
    parser.add_argument("--output-dir", help="Output directory for downloaded model")
    
    # Download options
    parser.add_argument("--cache-dir", help="Cache directory for downloaded models")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download even if model exists in cache")
    parser.add_argument("--max-results", type=int, default=10,
                        help="Maximum number of search results to show (default: 10)")
    
    return parser.parse_args()


def search_onnx_models(search_term: str, format_type: str, max_results: int):
    """Search for ONNX models on Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub not available. Install with: pip install huggingface_hub", file=sys.stderr)
        return False
    
    print(f"[INFO] Searching for {format_type} models with term: '{search_term}'")
    
    try:
        api = HfApi()
        
        # Search for models
        models = api.list_models(
            search=search_term,
            limit=max_results,
            sort="downloads",
            direction=-1
        )
        
        print(f"[INFO] Found {len(models)} models:")
        print()
        
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model.modelId}")
            print(f"    Downloads: {model.downloads:,}")
            print(f"    Tags: {', '.join(model.tags[:5])}")  # Show first 5 tags
            print()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Search failed: {e}", file=sys.stderr)
        return False


def download_onnx_model(model_name: str, output_dir: str, cache_dir: str = None, 
                       force_download: bool = False):
    """Download ONNX model from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not available. Install with: pip install huggingface_hub", file=sys.stderr)
        return False
    
    print(f"[INFO] Downloading ONNX model: {model_name}")
    
    try:
        # Download model files
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=False
        )
        
        # If output_dir is specified, copy files there
        if output_dir and output_dir != local_dir:
            import shutil
            os.makedirs(output_dir, exist_ok=True)
            shutil.copytree(local_dir, output_dir, dirs_exist_ok=True)
            print(f"[OK] Model copied to: {output_dir}")
        else:
            print(f"[OK] Model downloaded to: {local_dir}")
        
        # List downloaded files
        model_path = Path(output_dir) if output_dir else Path(local_dir)
        print(f"[INFO] Downloaded files:")
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download model {model_name}: {e}", file=sys.stderr)
        return False


def create_onnx_usage_example(model_name: str, output_dir: str):
    """Create usage example for the downloaded ONNX model."""
    example_script = f'''#!/usr/bin/env python3
"""
Example usage for ONNX model: {model_name}
"""

import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

def main():
    # Load tokenizer and ONNX model
    tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
    model = ORTModelForCausalLM.from_pretrained("{output_dir}")
    
    # Example input
    text = "Hello, how are you today?"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=100, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {{text}}")
    print(f"Output: {{response}}")

if __name__ == "__main__":
    main()
'''
    
    example_path = os.path.join(output_dir, "example_usage.py")
    with open(example_path, "w") as f:
        f.write(example_script)
    
    print(f"[INFO] Created usage example: {example_path}")


def main() -> int:
    args = parse_args()
    
    if args.search:
        # Search for models
        return 0 if search_onnx_models(args.search, args.format, args.max_results) else 1
    
    elif args.model_name and args.output_dir:
        # Download specific model
        success = download_onnx_model(
            model_name=args.model_name,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            force_download=args.force_download
        )
        
        if success:
            create_onnx_usage_example(args.model_name, args.output_dir)
            print(f"[OK] Download completed successfully!")
            print(f"[INFO] To test the model:")
            print(f"  python3 {args.output_dir}/example_usage.py")
        
        return 0 if success else 1
    
    else:
        print("[ERROR] Either --search or both --model-name and --output-dir must be specified")
        print("[INFO] Examples:")
        print("  # Search for ONNX models:")
        print("  python3 scripts/download_pretrained_onnx.py --search 'mistral' --format onnx")
        print()
        print("  # Download specific ONNX model:")
        print("  python3 scripts/download_pretrained_onnx.py \\")
        print("    --model-name microsoft/DialoGPT-medium-onnx \\")
        print("    --output-dir ./onnx_models")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



