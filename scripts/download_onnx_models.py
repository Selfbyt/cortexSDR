#!/usr/bin/env python3
"""
Download and convert Hugging Face models to ONNX format.

This script uses the Optimum library to export models to ONNX format,
which provides better optimization and deployment options.

Usage:
  # Convert Mistral to ONNX
  python3 scripts/download_onnx_models.py \
    --model-name mistralai/Mistral-7B-Instruct-v0.3 \
    --output-dir ./onnx_models \
    --task text-generation

  # Convert local model to ONNX
  python3 scripts/download_onnx_models.py \
    --model-dir /path/to/local/model \
    --output-dir ./onnx_models \
    --task text-generation

Requirements:
  pip install optimum[exporters] transformers torch
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert models to ONNX")
    
    # Model source options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-name", help="Hugging Face model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)")
    model_group.add_argument("--model-dir", help="Path to local model directory")
    
    # Output options
    parser.add_argument("--output-dir", required=True, help="Output directory for ONNX model")
    parser.add_argument("--task", default="text-generation", 
                        help="Task type (text-generation, question-answering, etc.)")
    
    # ONNX export options
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--device", default="cpu", help="Device for export (default: cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization")
    
    # Download options
    parser.add_argument("--cache-dir", help="Cache directory for downloaded models")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download even if model exists in cache")
    
    return parser.parse_args()


def check_optimum_installation():
    """Check if Optimum is installed with exporters."""
    try:
        import optimum
        from optimum.exporters.onnx import main as onnx_export
        print("[OK] Optimum with exporters is installed")
        return True
    except ImportError:
        print("[ERROR] Optimum with exporters not found.")
        print("[INFO] Install with: pip install optimum[exporters]")
        return False


def download_model_with_optimum_cli(model_name: str, output_dir: str, task: str, 
                                  opset: int, device: str, fp16: bool, int8: bool,
                                  cache_dir: str = None, force_download: bool = False):
    """Download and convert model using Optimum CLI."""
    
    # Build optimum-cli command
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", model_name,
        "--task", task,
        "--opset", str(opset),
        "--device", device,
        output_dir
    ]
    
    if fp16:
        cmd.append("--fp16")
    
    if int8:
        cmd.append("--int8")
    
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])
    
    if force_download:
        cmd.append("--force-download")
    
    print(f"[INFO] Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("[OK] ONNX export completed successfully")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ONNX export failed: {e}")
        print(f"[ERROR] stdout: {e.stdout}")
        print(f"[ERROR] stderr: {e.stderr}")
        return False


def convert_local_model_to_onnx(model_dir: str, output_dir: str, task: str,
                               opset: int, device: str, fp16: bool, int8: bool):
    """Convert local model to ONNX using programmatic approach."""
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer, AutoConfig
    except ImportError:
        print("[ERROR] Required packages not found. Install with: pip install optimum[exporters] transformers")
        return False
    
    print(f"[INFO] Converting local model from: {model_dir}")
    
    try:
        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        
        # Create ONNX model
        print("[INFO] Creating ONNX model...")
        ort_model = ORTModelForCausalLM.from_pretrained(
            model_dir,
            export=True,
            opset=opset,
            device=device
        )
        
        # Save ONNX model
        print(f"[INFO] Saving ONNX model to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        ort_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("[OK] Local model converted to ONNX successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to convert local model to ONNX: {e}")
        return False


def create_onnx_inference_example(output_dir: str, model_name: str = None):
    """Create an example script for ONNX inference."""
    example_script = f'''#!/usr/bin/env python3
"""
Example script for running ONNX model inference.
Generated for: {model_name or "local model"}
"""

import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

def main():
    # Load tokenizer and ONNX model
    tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
    model = ORTModelForCausalLM.from_pretrained("{output_dir}")
    
    # Example input
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, do_sample=True)
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {{text}}")
    print(f"Output: {{response}}")

if __name__ == "__main__":
    main()
'''
    
    example_path = os.path.join(output_dir, "inference_example.py")
    with open(example_path, "w") as f:
        f.write(example_script)
    
    print(f"[INFO] Created inference example: {example_path}")


def main() -> int:
    args = parse_args()
    
    # Check if Optimum is installed
    if not check_optimum_installation():
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = False
    
    if args.model_name:
        # Download and convert from Hugging Face
        print(f"[INFO] Converting {args.model_name} to ONNX...")
        success = download_model_with_optimum_cli(
            model_name=args.model_name,
            output_dir=args.output_dir,
            task=args.task,
            opset=args.opset,
            device=args.device,
            fp16=args.fp16,
            int8=args.int8,
            cache_dir=args.cache_dir,
            force_download=args.force_download
        )
        
        if success:
            create_onnx_inference_example(args.output_dir, args.model_name)
    
    elif args.model_dir:
        # Convert local model
        if not os.path.exists(args.model_dir):
            print(f"[ERROR] Model directory does not exist: {args.model_dir}")
            return 1
        
        print(f"[INFO] Converting local model from {args.model_dir} to ONNX...")
        success = convert_local_model_to_onnx(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            task=args.task,
            opset=args.opset,
            device=args.device,
            fp16=args.fp16,
            int8=args.int8
        )
        
        if success:
            create_onnx_inference_example(args.output_dir)
    
    if success:
        print(f"[OK] ONNX conversion completed successfully!")
        print(f"[INFO] Output directory: {args.output_dir}")
        print(f"[INFO] Files created:")
        for file in os.listdir(args.output_dir):
            print(f"  - {file}")
        
        print(f"[INFO] To run inference:")
        print(f"  python3 {args.output_dir}/inference_example.py")
        
        return 0
    else:
        print("[ERROR] ONNX conversion failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



