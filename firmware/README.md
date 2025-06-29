# CortexSDR Firmware

This directory contains the firmware implementation for the CortexSDR project, enabling efficient model inference on embedded devices and servers.

## Overview

The CortexSDR firmware provides a lightweight, optimized implementation of the sparse inference engine for running compressed `.sdr` models on various hardware platforms. It supports:

- Server-side deployment (x86_64)
- Embedded ARM devices (Cortex-M4)
- RISC-V platforms

## Directory Structure

```
firmware/
├── config/               # Build configuration files
│   └── firmware_config.cmake  # Target-specific build settings
├── include/              # Public header files
│   └── cortex_firmware.h      # Main firmware interface
├── src/                  # Source code
│   ├── cortex_firmware.cpp    # Implementation of firmware interface
│   └── firmware_main.cpp      # Main entry point
└── CMakeLists.txt        # Build system configuration
```

## Building the Firmware

The firmware can be built using the provided `build_firmware.sh` script in the project root:

```bash
# Build for server (default)
./build_firmware.sh

# Build for ARM Cortex-M4
./build_firmware.sh -t ARM_CORTEX_M4

# Build for RISC-V
./build_firmware.sh -t RISC_V
```

The build output will be placed in the `build_firmware` directory.

## Integration

### Embedded Integration

To integrate the CortexSDR firmware into your embedded project:

1. Include the `cortex_firmware.h` header
2. Link against the built firmware library
3. Initialize the firmware with `CortexFirmware::initialize()`
4. Load your model with `CortexFirmware::loadModel()`
5. Run inference with `CortexFirmware::runInference()`
6. Monitor resource usage with `CortexFirmware::getResourceStats()`
7. Clean up with `CortexFirmware::shutdown()`

### Server Integration

For server-side deployment, the process is similar to embedded integration, but you'll use the x86_64 build target.

## Memory Optimization

The firmware is designed to minimize memory usage through:

- Sparse tensor representation
- Quantized weights
- Streaming model loading
- Efficient memory allocation

## Performance Considerations

For optimal performance:

- Pre-process input data to match the expected model input format
- Use batched inference when possible
- Monitor memory usage with `getResourceStats()` to avoid OOM conditions
- Consider model pruning to reduce memory footprint

## Resource Monitoring

The firmware includes a comprehensive resource monitoring system that tracks:

- Memory usage (current and peak)
- CPU usage
- Custom resources (e.g., model loading time, inference time)
- Resource usage history
- Event markers for important operations

Resource monitoring is automatically enabled when the firmware is initialized. You can access resource statistics through the `getResourceStats` function.

Example:

```cpp
uint32_t memoryUsageKB;
float cpuUsagePercent;
CortexFirmware::getResourceStats(memoryUsageKB, cpuUsagePercent);
```

The resource monitor also supports custom resources and event markers:

```cpp
// Register a custom resource
g_resourceMonitor->registerCustomResource("CustomMetric", 
    []() -> double { return calculateMetric(); },
    "units");

// Mark an event
g_resourceMonitor->markEvent("Important Operation Completed");
```

To test the resource monitoring functionality, run:

```bash
./build/firmware/resource_monitor_test
```

## Error Handling

The firmware uses error codes to indicate success or failure:

- `CORTEX_FW_SUCCESS`: Operation completed successfully
- `CORTEX_FW_ERROR_INIT_FAILED`: Initialization failed
- `CORTEX_FW_ERROR_MEMORY_EXCEEDED`: Memory limit exceeded
- `CORTEX_FW_ERROR_INVALID_MODEL`: Invalid model format
- `CORTEX_FW_ERROR_INFERENCE_FAILED`: Inference operation failed

## Example Usage

```cpp
#include "cortex_firmware.h"
#include <iostream>
#include <vector>

int main() {
    // Initialize firmware
    if (CortexFirmware::initialize() != CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to initialize firmware" << std::endl;
        return 1;
    }
    
    // Load model (from file or memory)
    std::vector<uint8_t> modelData = loadModelFromFile("model.sdr");
    if (CortexFirmware::loadModel(modelData.data(), modelData.size()) != CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to load model" << std::endl;
        CortexFirmware::shutdown();
        return 1;
    }
    
    // Prepare input and output buffers
    std::vector<float> inputData = {1.0f, 2.0f, 3.0f};
    std::vector<size_t> inputShape = {1, 3};
    std::vector<float> outputData(10);
    std::vector<size_t> outputShape;
    
    // Run inference
    if (CortexFirmware::runInference(
        inputData.data(), inputShape, 
        outputData.data(), outputShape) != CORTEX_FW_SUCCESS) {
        std::cerr << "Inference failed" << std::endl;
        CortexFirmware::shutdown();
        return 1;
    }
    
    // Process results
    for (size_t i = 0; i < outputData.size(); i++) {
        std::cout << "Output[" << i << "] = " << outputData[i] << std::endl;
    }
    
    // Clean up
    CortexFirmware::shutdown();
    return 0;
}
```
