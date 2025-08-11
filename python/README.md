# cortexsdr-sdk

Python bindings for the CortexSDR SDK (compression, decompression, sparse inference).

## Install (from source)

Prereqs: Python 3.8+, CMake>=3.20, a C++17 compiler, pybind11, scikit-build-core, numpy, and the CortexSDR SDK shared library built/installed.

```bash
python -m pip install -U pip build wheel
python -m pip install scikit-build-core pybind11 numpy cmake ninja
# Option A: point to system install
# export CORTEXSDR_LIB_DIR=/usr/local/lib
# Option B: point to your built SDK lib dir
# export CORTEXSDR_LIB_DIR=/path/to/build/lib

# Build wheel
python -m build .
# Install wheel
python -m pip install dist/cortexsdr_sdk-*.whl
```

## Usage

```python
import numpy as np
from cortexsdr_sdk import CompressionOptions, Compressor, Decompressor, InferenceEngine

# Compression
opts = CompressionOptions()
opts.num_threads = 4
opts.sparsity = 0.02
opts.use_quantization = 1
opts.quantization_bits = 8

cmp = Compressor("/path/model.onnx", "onnx", opts)
cmp.compress("/path/model.sdr")
print(cmp.stats())

# Inference (directly from the .sdr)
engine = InferenceEngine("/path/model.sdr")
out = engine.run(np.array([0.1, 0.2, 0.3], dtype=np.float32))
print(out.shape, out[:5])

# Decompression (if needed)
dec = Decompressor("/path/model.sdr", sparsity=0.02)
dec.decompress("/path/model.onnx")
```

## Wheels

Use cibuildwheel to produce manylinux x86_64/aarch64 wheels (ARM64 for Graviton):

```bash
python -m pip install cibuildwheel
CIBW_ARCHS="x86_64 aarch64" cibuildwheel --output-dir wheelhouse
```

Ensure `CORTEXSDR_LIB_DIR` is set so the build can locate `libcortexsdr_sdk*.so`. After build, run `auditwheel repair` if needed.
