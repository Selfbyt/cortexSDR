# CortexSDR

CortexSDR is a C++17 project for compressing model artifacts into `.sdr` archives and running sparse/on-demand inference workflows from those archives.

## Quick Start (Just Use It)

If you already have binaries built, the fastest path is:

```powershell
# Compress a model
.\build\Debug\cortexsdr_ai_compression_cli.exe -c "C:\path\model.onnx" onnx "C:\path\model.sdr" 0.02

# Decompress it back
.\build\Debug\cortexsdr_ai_compression_cli.exe -d "C:\path\model.sdr" "C:\path\model_out.onnx" 0.02

# Run text prompt mode on an .sdr model
.\build\Debug\cortexsdr_text_cli.exe -p "Hello from CortexSDR" "C:\path\model.sdr"
```

Notes:

- `sparsity` defaults to `0.02` (2%) if omitted.
- The AI compression CLI supports `onnx`, `gguf`, `tensorflow`, `pytorch`, and `hdf5`.
- For all CLI options, see `docs/cli_text_generation.md` and `docs/cli_usage.md`.

## What This Repo Builds

Default CMake configuration builds:

- `cortexsdr` (core library target)
- `cortexsdr_sdk` (static SDK library)
- `cortexsdr_sdk_shared` (shared SDK library)
- `cortexsdr_ai_compression_cli` (compress/decompress/inference CLI)
- `cortexsdr_text_cli` (interactive text/inference CLI)
- `cortexsdr_bench` (native benchmark runner CLI)

Optional build modes also exist for firmware, library packaging, and Python wrapper generation.

## Repository Layout

- `src/ai_compression/core` - compressor/decompressor core
- `src/ai_compression/parsers` - model parsers (ONNX, GGUF, TensorFlow, PyTorch, HDF5)
- `src/ai_compression/strategies` - compression strategies
- `src/ai_compression/kernels` - optimized kernels (BLAS/SIMD/sparse/attention/flash)
- `src/ai_compression/streaming` - streaming compression path
- `src/ai_compression/onnx_proto` - ONNX protobuf sources (`onnx.proto`, `onnx.pb.h`, `onnx.pb.cc`)
- `src/ai_compression/api` - C API and SDK surface
- `docs` - CLI and AI compression docs
- `examples` - sample integration code
- `firmware` - firmware target and resource monitoring components
- `python` - Python SDK bindings and packaging files

## Requirements

Toolchain:

- C++17 compiler (MSVC, GCC, or Clang)
- CMake 3.10+
- Git
- `pkg-config` / `pkgconf`

Required libraries:

- `zlib`
- `protobuf` (compiler + runtime)
- `nlohmann_json`
- `libzip`
- `hdf5`

Optional libraries/features:

- ONNX Runtime (runtime-backed ONNX path)
- LibTorch (PyTorch parser path)
- TensorFlow C++ source/libs (TensorFlow parser path, off by default)
- BLAS/OpenBLAS/MKL (performance acceleration)

## Build

### Windows (vcpkg recommended)

```powershell
git clone https://github.com/Selfbyt/cortexSDR.git
Set-Location cortexSDR

vcpkg install zlib protobuf nlohmann-json libzip hdf5

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
```

### Linux

Install equivalent dev packages for compiler/CMake/pkg-config plus zlib/protobuf/nlohmann-json/libzip/hdf5, then:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Useful CMake Options

Feature toggles:

- `-DENABLE_ONNX=ON|OFF`
- `-DENABLE_TENSORFLOW=ON|OFF` (default `OFF`)
- `-DENABLE_PYTORCH=ON|OFF`
- `-DENABLE_HDF5=ON|OFF`
- `-DENABLE_GGUF=ON|OFF`
- `-DENABLE_BLAS=ON|OFF`
- `-DENABLE_SIMD=ON|OFF`
- `-DUSE_OPENBLAS=ON|OFF`
- `-DUSE_MKL=ON|OFF`

Build modes:

- `-DBUILD_FIRMWARE=ON|OFF`
- `-DBUILD_LIBRARY=ON|OFF`
- `-DBUILD_PYTHON_WRAPPER=ON|OFF`
- `-DBUILD_DESKTOP=ON|OFF` (currently not used by active desktop targets)

## CLI Usage

### AI Compression CLI

Binary (Windows Visual Studio generator default):

- `build/Debug/cortexsdr_ai_compression_cli.exe`

Usage:

```powershell
# Compress
.\build\Debug\cortexsdr_ai_compression_cli.exe -c "C:\path\model.onnx" onnx "C:\path\model.sdr" 0.02

# Decompress
.\build\Debug\cortexsdr_ai_compression_cli.exe -d "C:\path\model.sdr" "C:\path\model_out.onnx" 0.02

# Inference (indices file)
.\build\Debug\cortexsdr_ai_compression_cli.exe -i "C:\path\model.sdr" "C:\path\input_indices.txt"
```

Accepted format values:

- `onnx`
- `gguf`
- `tensorflow`
- `pytorch`
- `hdf5`

### Text/Inference CLI

Binary (Windows Visual Studio generator default):

- `build/Debug/cortexsdr_text_cli.exe`

Example:

```powershell
.\build\Debug\cortexsdr_text_cli.exe -p "Hello from CortexSDR" "C:\path\model.sdr"
```

For full option details, see `docs/cli_text_generation.md`.

### Benchmark CLI

Binary (Windows Visual Studio generator default):

- `build/Debug/cortexsdr_bench.exe`

Example:

```powershell
.\build\Debug\cortexsdr_bench.exe --config benchmarks\suite.example.json
```

See `benchmarks/README.md` for the benchmark suite format and metrics.

## ONNX Protobuf Regeneration

If `src/ai_compression/onnx_proto/onnx.proto` changes, regenerate generated files with a matching protobuf toolchain:

```powershell
& "C:/tools/vcpkg/installed/x64-windows/tools/protobuf/protoc.exe" --proto_path="src/ai_compression/onnx_proto" --cpp_out="src/ai_compression/onnx_proto" "src/ai_compression/onnx_proto/onnx.proto"
```

Keeping `protoc` and runtime versions aligned avoids protobuf compile/link mismatches.

## Optional Components

- Firmware build helper: `build_firmware.sh`
- Library packaging helper: `build_library.sh`
- Python SDK build helper: `build_python.sh`
- Python binding docs: `python/README.md`

## Troubleshooting

- Protobuf mismatch (`protoc` vs linked runtime): regenerate `onnx.pb.*` with matching tooling.
- Missing ONNX Runtime: build continues, runtime-backed ONNX features are disabled.
- Missing LibTorch: PyTorch parser support is disabled.
- Missing BLAS/OpenBLAS/MKL: build continues with reduced performance paths.

## Additional Docs

- `src/ai_compression/README.md`
- `docs/cli_usage.md`
- `docs/cli_text_generation.md`
- `docs/ai_compression/compression_strategies.md`
- `CHANGELOG.md`

## License

MIT. See `LICENSE`.
