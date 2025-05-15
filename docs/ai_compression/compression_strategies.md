# Compression Strategies

This document describes the compression strategies available in the CortexAI Compression system and provides guidance on their usage.

## Overview

The compression system supports multiple strategies optimized for different types of data found in AI models:

1. Gzip Strategy
2. Numerical RLE (Run-Length Encoding)
3. SDR (Sparse Distributed Representation) Index Storage
4. Quantized Tensor Strategy

Each strategy is designed for specific use cases and data types. The system automatically selects appropriate strategies based on the tensor type and characteristics.

## Strategy Details

### 1. Gzip Strategy

**Best for**: General-purpose compression, metadata, and configuration files
**Implementation**: `GzipStrategy`

- Uses zlib's deflate algorithm
- Provides good compression for text-based data
- Default compression level: 6 (configurable 1-9)
- Thread-safe implementation

Example usage:
```cpp
auto gzipStrategy = std::make_shared<GzipStrategy>();
compressor.registerStrategy(SegmentType::METADATA_JSON, gzipStrategy);
```

### 2. Numerical RLE Strategy

**Best for**: Weight tensors with repeated values
**Implementation**: `NumericalRLEStrategy`

- Optimized for floating-point and integer data
- Efficient for layers with repeated weights (e.g., padding)
- Supports multiple numeric types (FP32, FP16, INT8)
- Automatically detects run patterns

Performance characteristics:
- Best case: O(1) for constant tensors
- Worst case: O(n) for random data
- Memory overhead: Minimal

### 3. SDR Index Storage Strategy

**Best for**: Sparse tensors and attention matrices
**Implementation**: `SDRIndexStorageStrategy`

- Stores only non-zero element indices
- Optional delta encoding for sorted indices
- Efficient for sparse attention patterns
- Configurable sparsity threshold

Compression ratio depends on:
- Sparsity level
- Index distribution
- Whether indices are sorted

### 4. Quantized Tensor Strategy

**Best for**: Dense floating-point weight tensors
**Implementation**: `QuantizedTensorStrategy`

Features:
- Supports 8-bit and 4-bit quantization
- Symmetric and asymmetric quantization
- Per-tensor scaling
- Optional bias correction

Configuration options:
```cpp
// 8-bit symmetric quantization
auto quantStrategy = std::make_shared<QuantizedTensorStrategy>(8, true);

// 4-bit asymmetric quantization
auto quantStrategy = std::make_shared<QuantizedTensorStrategy>(4, false);
```

## Performance Benchmarks

| Strategy | Data Type | Compression Ratio | Speed (MB/s) | Quality Loss |
|----------|-----------|-------------------|--------------|--------------|
| Gzip | JSON | 3-5x | 50-100 | Lossless |
| RLE | FP32 | 2-20x | 200-400 | Lossless |
| SDR | Sparse | 10-50x | 150-300 | Lossless |
| Quantized (8-bit) | FP32 | 4x | 300-500 | ~0.5% |
| Quantized (4-bit) | FP32 | 8x | 350-550 | ~2% |

## Best Practices

1. **Strategy Selection**:
   - Use Gzip for metadata and small tensors
   - Use RLE for layers with repetitive patterns
   - Use SDR for sparse layers (sparsity > 70%)
   - Use Quantization for dense weight matrices

2. **Performance Optimization**:
   - Enable parallel compression for large models
   - Use chunked compression for better memory usage
   - Consider model structure when selecting chunk size

3. **Quality Control**:
   - Monitor quantization error for critical layers
   - Use symmetric quantization for attention layers
   - Preserve accuracy of key model components

4. **Memory Management**:
   - Use streaming compression for large models
   - Process tensors in chunks when possible
   - Monitor peak memory usage

## Example Configurations

### High Compression Priority
```cpp
// Maximize compression ratio
auto compressor = std::make_unique<AICompressor>(parser);
compressor->registerStrategy(SegmentType::WEIGHTS_FP32, 
    std::make_shared<QuantizedTensorStrategy>(4, false));
compressor->registerStrategy(SegmentType::SPARSE_INDICES,
    std::make_shared<SDRIndexStorageStrategy>());
```

### High Speed Priority
```cpp
// Maximize compression speed
auto compressor = std::make_unique<AICompressor>(parser);
compressor->registerStrategy(SegmentType::WEIGHTS_FP32,
    std::make_shared<QuantizedTensorStrategy>(8, true));
compressor->setCompressionThreads(4);
```

### Balanced Configuration
```cpp
// Balance between speed and ratio
auto compressor = std::make_unique<AICompressor>(parser);
compressor->registerStrategy(SegmentType::WEIGHTS_FP32,
    std::make_shared<QuantizedTensorStrategy>(8, false));
compressor->registerStrategy(SegmentType::SPARSE_INDICES,
    std::make_shared<SDRIndexStorageStrategy>());
compressor->setCompressionThreads(2);
```

## Troubleshooting

Common issues and solutions:

1. **High Memory Usage**
   - Use chunked compression
   - Reduce parallel compression threads
   - Enable streaming mode

2. **Poor Compression Ratio**
   - Check tensor sparsity levels
   - Adjust quantization parameters
   - Consider tensor reordering

3. **Slow Compression Speed**
   - Increase chunk size
   - Enable parallel compression
   - Use faster strategies for non-critical layers

4. **Quality Loss**
   - Use symmetric quantization
   - Increase quantization bits
   - Preserve critical layer precision 