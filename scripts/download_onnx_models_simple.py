#!/usr/bin/env python3
"""
Simple script to download pre-converted ONNX models from Hugging Face.

This is much easier for low-end devices since the heavy conversion work
is already done on Hugging Face's servers.

Usage:
  # Search for ONNX models
  python3 scripts/download_onnx_models_simple.py --search "mistral"

  # Download specific ONNX model
  python3 scripts/download_onnx_models_simple.py \
    --model-name microsoft/DialoGPT-medium-onnx \
    --output-dir ./onnx_models

Requirements:
  pip install huggingface_hub
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pre-converted ONNX models")
    parser.add_argument("--search", help="Search for models with this term")
    parser.add_argument("--model-name", help="Specific model name to download")
    parser.add_argument("--output-dir", help="Output directory for downloaded model")
    parser.add_argument("--max-results", type=int, default=10,
                        help="Maximum number of search results (default: 10)")
    return parser.parse_args()


def search_onnx_models(search_term: str, max_results: int):
    """Search for ONNX models on Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub not available. Install with: pip install huggingface_hub")
        return False
    
    print(f"[INFO] Searching for ONNX models with term: '{search_term}'")
    
    try:
        api = HfApi()
        models = list(api.list_models(
            search=search_term,
            limit=max_results,
            sort="downloads",
            direction=-1
        ))
        
        print(f"[INFO] Found {len(models)} models:")
        print()
        
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model.modelId}")
            print(f"    Downloads: {model.downloads:,}")
            print(f"    Tags: {', '.join(model.tags[:5])}")
            print()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return False


def download_onnx_model(model_name: str, output_dir: str):
    """Download ONNX model from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub not available. Install with: pip install huggingface_hub")
        return False
    
    print(f"[INFO] Downloading ONNX model: {model_name}")
    
    try:
        # Download model files
        local_dir = snapshot_download(
            repo_id=model_name,
            local_files_only=False
        )
        
        # Copy to output directory
        if output_dir != local_dir:
            import shutil
            os.makedirs(output_dir, exist_ok=True)
            shutil.copytree(local_dir, output_dir, dirs_exist_ok=True)
            print(f"[OK] Model copied to: {output_dir}")
        else:
            print(f"[OK] Model downloaded to: {local_dir}")
        
        # List downloaded files
        model_path = Path(output_dir)
        print(f"[INFO] Downloaded files:")
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download model {model_name}: {e}")
        return False


def main() -> int:
    args = parse_args()
    
    if args.search:
        # Search for models
        return 0 if search_onnx_models(args.search, args.max_results) else 1
    
    elif args.model_name and args.output_dir:
        # Download specific model
        success = download_onnx_model(args.model_name, args.output_dir)
        return 0 if success else 1
    
    else:
        print("[ERROR] Either --search or both --model-name and --output-dir must be specified")
        print("[INFO] Examples:")
        print("  # Search for ONNX models:")
        print("  python3 scripts/download_onnx_models_simple.py --search 'mistral'")
        print()
        print("  # Download specific ONNX model:")
        print("  python3 scripts/download_onnx_models_simple.py \\")
        print("    --model-name microsoft/DialoGPT-medium-onnx \\")
        print("    --output-dir ./onnx_models")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
