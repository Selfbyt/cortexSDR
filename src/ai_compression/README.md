# AI Compression Module

This module provides AI model compression capabilities with support for multiple model formats and compression strategies.

## Directory Structure

```
ai_compression/
├── core/           # Core compression functionality
│   ├── AICompressor.cpp/hpp
│   ├── AIDecompressor.cpp/hpp
│   ├── ModelSegment.hpp
│   └── AIModelParser.hpp
├── parsers/        # Model format parsers
│   ├── GGUFModelParser.cpp/hpp
│   ├── ONNXModelParser.cpp/hpp
│   ├── PyTorchModelParser.cpp/hpp
│   └── TensorFlowModelParser.cpp/hpp
├── strategies/     # Compression strategies
│   ├── CompressionStrategy.hpp
│   ├── GzipStrategy.cpp/hpp
│   ├── NumericalRLE.cpp/hpp
│   └── SDRIndexStorage.cpp/hpp
├── streaming/      # Streaming compression
│   ├── StreamingCompressor.cpp/hpp
│   └── ICompressionHandler.hpp
└── api/           # C API interface
    ├── c_api.cpp
    └── c_api.hpp
```

## Components

### Core
- `AICompressor`: Main compression engine
- `AIDecompressor`: Decompression engine
- `ModelSegment`: Data structure for model segments
- `AIModelParser`: Base interface for model parsers

### Parsers
- `GGUFModelParser`: GGUF format parser
- `ONNXModelParser`: ONNX format parser
- `PyTorchModelParser`: PyTorch format parser
- `TensorFlowModelParser`: TensorFlow format parser

### Strategies
- `CompressionStrategy`: Base interface for compression strategies
- `GzipStrategy`: Gzip-based compression
- `NumericalRLE`: Run-length encoding for numerical data
- `SDRIndexStorage`: Sparse distributed representation storage

### Streaming
- `StreamingCompressor`: Streaming compression support
- `ICompressionHandler`: Interface for streaming handlers

### API
- `c_api`: C interface for the compression module

## Features

- Support for multiple model formats (GGUF, ONNX, PyTorch, TensorFlow)
- Multiple compression strategies (Gzip, RLE, SDR)
- Streaming compression support
- Parallel processing capabilities
- C API for integration with other languages
- Model-aware chunking for optimal compression
- Metadata preservation
- Error handling and validation

## Dependencies

This module requires several external libraries that are not included in the repository due to their size. You need to download and extract them before building the project:

### ONNX Runtime

1. Download ONNX Runtime from the official website: https://github.com/microsoft/onnxruntime/releases
2. Extract the downloaded file to the `onnxruntime/` directory in the project root

```bash
# Example for Linux x64 (CPU version)
mkdir -p onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.1/onnxruntime-linux-x64-1.21.1.tgz
tar -xzf onnxruntime-linux-x64-1.21.1.tgz -C onnxruntime/
```

### TensorFlow

1. Download TensorFlow C API from the official website: https://www.tensorflow.org/install/lang_c
2. Extract the downloaded file to the `libtensorflow/` directory in the project root

```bash
# Example for Linux x64
mkdir -p libtensorflow
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.18.0.tar.gz
tar -xzf libtensorflow-cpu-linux-x86_64-2.18.0.tar.gz -C libtensorflow/
```

### PyTorch

1. Download LibTorch from the official website: https://pytorch.org/get-started/locally/
2. Extract the downloaded file to the `libtorch/` directory in the project root

```bash
# Example for Linux x64 (CPU version)
mkdir -p libtorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cpu.zip -d libtorch/
```

## Model Conversion

The module includes a model converter tool that can convert models from various formats (TensorFlow, PyTorch) to ONNX format. After building the project, you can use the `cortexsdr_model_converter` tool:

```bash
./build/cortexsdr_model_converter <input_model> <output_model.onnx> <input_format>
```

Supported formats:
- `pytorch`: PyTorch models (.pt, .pth)
- `tensorflow`: TensorFlow models (SavedModel directory)
- `onnx`: ONNX models (already in ONNX format, will be copied)

## Compression Tuning

You can control the fraction of active bits in the SDR encoding using the `--sparsity` or `-s` parameter with the CLI tool (default is 2%). This helps tune compression and achieve higher compression ratios.

```bash
./build/cortexsdr_cli --sparsity 0.01 --input model.onnx --output model.compressed
```

## Usage

See the main project documentation for more usage examples and API details.