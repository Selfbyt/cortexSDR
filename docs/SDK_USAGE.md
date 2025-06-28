# CortexSDR SDK Usage Guide

This document provides instructions for using the CortexSDR SDK, which combines the C API and sparse inference engine into a unified library that can be used across different platforms including mobile applications and server-side deployments.

## Overview

The CortexSDR SDK provides a complete solution for AI model compression, decompression, and inference. It allows you to:

1. Compress AI models to reduce their size
2. Decompress compressed models
3. Run inference on compressed models directly without full decompression
4. Access advanced features like sparse inference for efficient model execution

## Installation

### Building from Source

To build the SDK from source:

```bash
# Clone the repository
git clone https://github.com/your-org/cortexSDR.git
cd cortexSDR

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DBUILD_SDK=ON

# Build
make

# Install (optional)
sudo make install
```

### Dependencies

The SDK requires the following dependencies:
- ZLIB
- Protobuf
- nlohmann_json

Optional dependencies for extended functionality:
- ONNX Runtime (for ONNX model support)
- TensorFlow (for TensorFlow model conversion)
- PyTorch (for PyTorch model conversion)

## Using the SDK

### Including in Your Project

#### CMake Projects

```cmake
find_package(cortexsdr_sdk REQUIRED)
target_link_libraries(your_project PRIVATE cortexsdr_sdk)
```

#### Manual Linking

```bash
g++ -o your_app your_app.cpp -lcortexsdr_sdk
```

### Basic Usage

Here's a simple example of how to use the SDK:

```c
#include "cortexsdr/cortex_sdk.h"
#include <stdio.h>

int main() {
    // Print SDK version
    printf("CortexSDR SDK Version: %s\n", cortex_sdk_version());
    
    // Initialize compression options
    CortexCompressionOptions options;
    CortexError error = cortex_compression_options_init(&options);
    if (error.code != CORTEX_SUCCESS) {
        printf("Error: %s\n", error.message);
        cortex_error_free(&error);
        return 1;
    }
    
    // Configure options
    options.sparsity = 0.02f; // 2% sparsity
    
    // Compress a model
    CortexCompressorHandle compressor = NULL;
    error = cortex_compressor_create("model.onnx", "onnx", &options, &compressor);
    if (error.code == CORTEX_SUCCESS) {
        error = cortex_compressor_compress(compressor, "compressed_model.cortex");
        cortex_compressor_free(compressor);
    }
    
    // Create inference engine
    CortexInferenceEngineHandle engine = NULL;
    error = cortex_inference_engine_create("compressed_model.cortex", &engine);
    if (error.code != CORTEX_SUCCESS) {
        printf("Error: %s\n", error.message);
        cortex_error_free(&error);
        return 1;
    }
    
    // Run inference
    float input_data[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float output_data[100];
    size_t actual_output_size = 0;
    
    error = cortex_inference_engine_run(
        engine,
        input_data,
        10,
        output_data,
        100,
        &actual_output_size
    );
    
    // Free resources
    cortex_inference_engine_free(engine);
    
    return 0;
}
```

## API Reference

### Error Handling

```c
typedef enum {
    CORTEX_SUCCESS = 0,
    CORTEX_ERROR_INVALID_ARGUMENT = -1,
    CORTEX_ERROR_FILE_IO = -2,
    CORTEX_ERROR_MEMORY = -3,
    CORTEX_ERROR_UNSUPPORTED_FORMAT = -4,
    CORTEX_ERROR_COMPRESSION = -5,
    CORTEX_ERROR_DECOMPRESSION = -6,
    CORTEX_ERROR_INFERENCE = -7,
    CORTEX_ERROR_UNKNOWN = -99
} CortexStatus;

typedef struct {
    const char* message;
    int code;
} CortexError;

void cortex_error_free(CortexError* error);
```

### Compression

```c
typedef struct CortexCompressor* CortexCompressorHandle;

typedef struct {
    size_t num_threads;
    int verbose;
    int show_stats;
    int use_delta_encoding;
    int use_rle;
    int compression_level;
    int use_quantization;
    int quantization_bits;
    float sparsity;
} CortexCompressionOptions;

CortexError cortex_compression_options_init(CortexCompressionOptions* options);

CortexError cortex_compressor_create(
    const char* model_path,
    const char* format,
    const CortexCompressionOptions* options,
    CortexCompressorHandle* handle
);

CortexError cortex_compressor_compress(
    CortexCompressorHandle handle,
    const char* output_path
);

CortexError cortex_compressor_get_stats(
    CortexCompressorHandle handle,
    size_t* original_size,
    size_t* compressed_size,
    double* compression_ratio,
    double* compression_time_ms
);

CortexError cortex_compressor_free(CortexCompressorHandle handle);
```

### Decompression

```c
typedef void* CortexDecompressorHandle;

CortexError cortex_decompressor_create(
    const char* compressed_path,
    CortexDecompressorHandle* handle,
    float sparsity
);

CortexError cortex_decompressor_decompress(
    CortexDecompressorHandle handle,
    const char* compressed_path,
    const char* output_path
);

CortexError cortex_decompressor_free(CortexDecompressorHandle handle);
```

### Inference

```c
typedef struct CortexInferenceEngine* CortexInferenceEngineHandle;

CortexError cortex_inference_engine_create(
    const char* compressed_model_path,
    CortexInferenceEngineHandle* handle
);

CortexError cortex_inference_engine_set_batch_size(
    CortexInferenceEngineHandle handle,
    size_t batch_size
);

CortexError cortex_inference_engine_enable_dropout(
    CortexInferenceEngineHandle handle,
    int enable
);

CortexError cortex_inference_engine_set_mode(
    CortexInferenceEngineHandle handle,
    int training_mode
);

CortexError cortex_inference_engine_run(
    CortexInferenceEngineHandle handle,
    const float* input_data,
    size_t input_size,
    float* output_data,
    size_t output_size,
    size_t* actual_output_size
);

CortexError cortex_inference_engine_run_layer(
    CortexInferenceEngineHandle handle,
    const char* layer_name,
    const float* input_data,
    size_t input_size,
    float* output_data,
    size_t output_size,
    size_t* actual_output_size
);

CortexError cortex_inference_engine_free(
    CortexInferenceEngineHandle handle
);
```

### Utility Functions

```c
const char* cortex_sdk_version();
```

## Advanced Usage

### Layer-by-Layer Inference

For more fine-grained control, you can run inference on specific layers:

```c
CortexError error = cortex_inference_engine_run_layer(
    engine,
    "layer_name",
    input_data,
    input_size,
    output_data,
    output_size,
    &actual_output_size
);
```

### Batch Processing

For batch processing, set the batch size before running inference:

```c
cortex_inference_engine_set_batch_size(engine, 32); // Process 32 samples at once
```

### Training Mode

The SDK supports both inference and training modes:

```c
// Enable training mode
cortex_inference_engine_set_mode(engine, 1);

// Enable dropout during training
cortex_inference_engine_enable_dropout(engine, 1);
```

## Performance Considerations

- **Sparsity**: Higher sparsity values (e.g., 0.05 for 5%) result in smaller models but may reduce accuracy
- **Quantization**: Enable quantization for further size reduction, especially for mobile deployments
- **Batch Size**: Larger batch sizes generally improve throughput but increase memory usage

## Troubleshooting

### Common Errors

- **CORTEX_ERROR_FILE_IO**: Check file paths and permissions
- **CORTEX_ERROR_UNSUPPORTED_FORMAT**: Verify that the model format is supported
- **CORTEX_ERROR_MEMORY**: Increase available memory or reduce batch size

### Memory Management

Always free resources when done:

```c
cortex_compressor_free(compressor);
cortex_decompressor_free(decompressor);
cortex_inference_engine_free(engine);
cortex_error_free(&error);
```

## Platform-Specific Notes

### Mobile (Android/iOS)

- Use static linking for mobile platforms
- Consider enabling quantization for reduced memory footprint

### Server Deployment

- Use dynamic linking for easier updates
- Configure thread count based on available CPU cores

## License

This SDK is licensed under [LICENSE TERMS]. See LICENSE file for details.
