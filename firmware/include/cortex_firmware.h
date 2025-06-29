#ifndef CORTEX_FIRMWARE_H
#define CORTEX_FIRMWARE_H

#include <cstdint>
#include <vector>
#include <string>

// Firmware version and configuration
#define CORTEX_FIRMWARE_VERSION "1.0.0"
#define CORTEX_FIRMWARE_NAME "CortexSDR"
#define CORTEX_FIRMWARE_BUILD_DATE __DATE__
#define CORTEX_FIRMWARE_BUILD_TIME __TIME__

// Hardware configuration
#define MEMORY_LIMIT_KB 512
#define MAX_MODEL_SIZE_MB 64
#define MAX_TENSOR_DIMS 8

// Error codes
#define CORTEX_FW_SUCCESS 0
#define CORTEX_FW_ERROR_INIT_FAILED 1
#define CORTEX_FW_ERROR_MEMORY_EXCEEDED 2
#define CORTEX_FW_ERROR_INVALID_MODEL 3
#define CORTEX_FW_ERROR_INFERENCE_FAILED 4

// Forward declarations
namespace CortexFirmware {

/**
 * Initialize the firmware and hardware components
 * @return Error code (0 for success)
 */
int initialize();

/**
 * Load a compressed model from memory
 * @param modelData Pointer to model data in memory
 * @param modelSize Size of model data in bytes
 * @return Error code (0 for success)
 */
int loadModel(const uint8_t* modelData, size_t modelSize);

/**
 * Run inference on the loaded model
 * @param inputData Input tensor data
 * @param inputShape Shape of input tensor
 * @param outputData Output tensor data (pre-allocated)
 * @param outputShape Shape of output tensor
 * @return Error code (0 for success)
 */
int runInference(
    const float* inputData, 
    const std::vector<size_t>& inputShape,
    float* outputData,
    std::vector<size_t>& outputShape
);

/**
 * Get resource usage statistics
 * @param memoryUsageKB Memory usage in KB
 * @param cpuUsagePercent CPU usage percentage
 * @return Error code (0 for success)
 */
int getResourceStats(uint32_t& memoryUsageKB, float& cpuUsagePercent);

/**
 * Shutdown the firmware and hardware components
 * @return Error code (0 for success)
 */
int shutdown();

} // namespace CortexFirmware

#endif // CORTEX_FIRMWARE_H
