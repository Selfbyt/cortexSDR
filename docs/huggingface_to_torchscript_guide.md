# Hugging Face to TorchScript Conversion Guide

This guide explains how to clone models from Hugging Face and convert them to TorchScript format for deployment.

## Overview

The conversion process involves:
1. **Cloning/Downloading** models from Hugging Face Hub
2. **Loading** models from various formats (.pth, .safetensors, standard HF format)
3. **Converting** to TorchScript for optimized inference
4. **Saving** in the desired format

## Prerequisites

Install required dependencies:

```bash
pip install torch transformers safetensors huggingface_hub
```

## Quick Start

### 1. Clone and Convert a Model

```bash
python3 scripts/hf_to_torchscript.py \
  --model-name microsoft/DialoGPT-medium \
  --output-dir ./converted_models \
  --format torchscript
```

### 2. Convert Existing Local Model

```bash
python3 scripts/hf_to_torchscript.py \
  --model-dir /path/to/local/model \
  --output-dir ./converted_models \
  --format torchscript
```

## Detailed Usage

### Command Line Options

#### Model Source
- `--model-name`: Hugging Face model identifier (e.g., `microsoft/DialoGPT-medium`)
- `--model-dir`: Path to local model directory

#### Output Options
- `--output-dir`: Directory to save converted models
- `--format`: Output format (`torchscript`, `state_dict`, or `both`)

#### Model Loading
- `--dtype`: Data type (`fp32`, `fp16`, `bf16`) - default: `fp16`
- `--device`: Device to load model on (`cpu`, `cuda`) - default: `cpu`
- `--trust-remote-code`: Enable for custom models
- `--low-cpu-mem-usage`: Use memory-efficient loading

#### TorchScript Options
- `--example-inputs`: Input shape for tracing (default: `[1, 512]`)
- `--strict`: Use strict mode for compilation

#### Download Options
- `--cache-dir`: Custom cache directory
- `--force-download`: Force re-download

## File Format Support

### Input Formats
- **Standard Hugging Face**: `config.json`, `pytorch_model.bin`
- **SafeTensors**: `model.safetensors` (faster, safer loading)
- **PyTorch Checkpoints**: `.pth` files
- **Sharded Models**: Multiple `.bin` or `.safetensors` files

### Output Formats
- **TorchScript**: `.pt` files optimized for inference
- **State Dict**: Raw PyTorch state dictionaries

## Examples

### Example 1: Convert GPT-2 Model

```bash
python3 scripts/hf_to_torchscript.py \
  --model-name gpt2 \
  --output-dir ./gpt2_converted \
  --format both \
  --dtype fp16 \
  --example-inputs 1 1024
```

### Example 2: Convert Local LLaMA Model

```bash
python3 scripts/hf_to_torchscript.py \
  --model-dir /home/user/llama-7b \
  --output-dir ./llama_converted \
  --format torchscript \
  --dtype bf16 \
  --trust-remote-code
```

### Example 3: Convert with Custom Cache

```bash
python3 scripts/hf_to_torchscript.py \
  --model-name microsoft/DialoGPT-large \
  --output-dir ./dialogpt_converted \
  --cache-dir /custom/cache \
  --force-download
```

## Understanding the Conversion Process

### 1. Model Download
- Uses `huggingface_hub.snapshot_download()` for efficient downloading
- Supports caching to avoid re-downloading
- Handles large models with automatic sharding

### 2. Model Loading
- Automatically detects model format (safetensors vs pytorch)
- Loads with specified precision (fp16/bf16 for memory efficiency)
- Supports custom models with `trust_remote_code`

### 3. TorchScript Conversion
- Uses `torch.jit.trace()` for model tracing
- Requires example inputs to trace execution path
- Applies optimizations with `optimize_for_inference()`

### 4. Output Generation
- TorchScript: Optimized for deployment and inference
- State Dict: Raw weights for custom loading

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Use lower precision and CPU
--dtype fp16 --device cpu --low-cpu-mem-usage
```

#### 2. Custom Model Errors
```bash
# Enable trust_remote_code for custom architectures
--trust-remote-code
```

#### 3. TorchScript Tracing Errors
```bash
# Adjust input shape to match model expectations
--example-inputs 1 512  # [batch_size, sequence_length]
```

#### 4. SafeTensors Loading
The script automatically handles SafeTensors format. If you encounter issues:
- Ensure `safetensors` package is installed
- Check if model files are corrupted

### Performance Tips

1. **Use FP16/BF16**: Reduces memory usage and improves speed
2. **CPU Loading**: Use `--device cpu` for large models
3. **Low Memory Mode**: Always use `--low-cpu-mem-usage`
4. **Appropriate Input Shape**: Match your actual inference input size

## Integration with Existing Code

### Loading TorchScript Models

```python
import torch

# Load TorchScript model
model = torch.jit.load("path/to/model_torchscript.pt")
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

### Loading State Dict

```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Load config and create model
config = AutoConfig.from_pretrained("model_dir")
model = AutoModelForCausalLM.from_config(config)

# Load state dict
state_dict = torch.load("path/to/model_state_dict.pt")
model.load_state_dict(state_dict)
model.eval()
```

## Advanced Usage

### Custom Input Shapes
For models with specific input requirements:

```bash
# For vision models
--example-inputs 1 3 224 224

# For sequence models
--example-inputs 1 1024

# For multi-input models (requires code modification)
```

### Batch Processing
Convert multiple models:

```bash
#!/bin/bash
models=("gpt2" "microsoft/DialoGPT-medium" "distilbert-base-uncased")
for model in "${models[@]}"; do
    python3 scripts/hf_to_torchscript.py \
        --model-name "$model" \
        --output-dir "./converted_models/$model" \
        --format torchscript
done
```

## Best Practices

1. **Test Conversion**: Always test the converted model with sample inputs
2. **Version Control**: Keep track of model versions and conversion parameters
3. **Documentation**: Document any custom modifications or special requirements
4. **Validation**: Compare outputs between original and converted models
5. **Performance Testing**: Benchmark inference speed and memory usage

## File Structure

After conversion, your output directory will contain:

```
converted_models/
├── model_name_torchscript.pt    # TorchScript model
├── model_name_state_dict.pt     # State dictionary (if requested)
└── conversion_log.txt           # Conversion details (if logging enabled)
```

This guide provides a comprehensive approach to converting Hugging Face models to TorchScript format, supporting various input formats and providing flexibility for different use cases.
