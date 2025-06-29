#include "../include/cortex_firmware.h"
#include "../include/cortex_resource_monitor.h"
#include "../../src/ai_compression/SparseInferenceEngine.hpp"
#include "../../src/ai_compression/AIDecompressor.hpp"
#include <iostream>
#include <chrono>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace CortexFirmware {

// Global variables
static std::unique_ptr<CortexAICompression::SDRModelLoader> g_modelLoader;
static std::unique_ptr<CortexAICompression::SDRInferenceEngine> g_inferenceEngine;
static std::unique_ptr<ResourceMonitor> g_resourceMonitor;
static bool g_initialized = false;
static uint32_t g_peakMemoryUsage = 0;
static auto g_startTime = std::chrono::high_resolution_clock::now();

// Internal helper functions
namespace {
    // Memory tracking class to monitor allocations
    class MemoryTracker {
    public:
        static void recordAllocation(size_t bytes) {
            currentUsage_ += bytes;
            peakUsage_ = std::max(peakUsage_, currentUsage_);
        }
        
        static void recordDeallocation(size_t bytes) {
            if (bytes <= currentUsage_) {
                currentUsage_ -= bytes;
            } else {
                // This shouldn't happen, but protect against underflow
                currentUsage_ = 0;
            }
        }
        
        static uint32_t getCurrentUsageKB() {
            return static_cast<uint32_t>(currentUsage_ / 1024);
        }
        
        static uint32_t getPeakUsageKB() {
            return static_cast<uint32_t>(peakUsage_ / 1024);
        }
        
    private:
        static size_t currentUsage_;
        static size_t peakUsage_;
    };
    
    size_t MemoryTracker::currentUsage_ = 0;
    size_t MemoryTracker::peakUsage_ = 0;
    
    void updateMemoryUsage() {
        g_peakMemoryUsage = MemoryTracker::getPeakUsageKB();
    }
    
    uint32_t estimateMemoryUsage() {
        // Use our memory tracker for accurate usage reporting
        return MemoryTracker::getCurrentUsageKB();
    }
    
    // Custom memory buffer for direct memory loading
    class MemoryBuffer {
    public:
        MemoryBuffer(const uint8_t* data, size_t size) : data_(data), size_(size), pos_(0) {}
        
        size_t read(char* buffer, size_t bytes) {
            if (pos_ >= size_) return 0;
            
            size_t bytesToRead = std::min(bytes, size_ - pos_);
            std::memcpy(buffer, data_ + pos_, bytesToRead);
            pos_ += bytesToRead;
            return bytesToRead;
        }
        
        bool eof() const {
            return pos_ >= size_;
        }
        
    private:
        const uint8_t* data_;
        size_t size_;
        size_t pos_;
    };
}

int initialize() {
    if (g_initialized) {
        std::cout << "Firmware already initialized" << std::endl;
        return CORTEX_FW_SUCCESS;
    }
    
    try {
        // Print firmware information
        std::cout << "Initializing " << CORTEX_FIRMWARE_NAME << " firmware v" 
                  << CORTEX_FIRMWARE_VERSION << std::endl;
        std::cout << "Build date: " << CORTEX_FIRMWARE_BUILD_DATE << " " 
                  << CORTEX_FIRMWARE_BUILD_TIME << std::endl;
        
        // Reset memory tracking
        g_peakMemoryUsage = 0;
        g_startTime = std::chrono::high_resolution_clock::now();
        g_lastCpuCheckTime = g_startTime;
        
        // Initialize CPU usage tracking
        g_lastTotalTime = 0;
        g_lastActiveTime = 0;
        g_lastCpuUsage = 0.0f;
        
        // Check available hardware resources
        uint32_t availableMemoryKB = 0;
        
#ifdef __linux__
        // On Linux, check available memory from /proc/meminfo
        try {
            std::ifstream meminfo("/proc/meminfo");
            if (meminfo.is_open()) {
                std::string line;
                while (std::getline(meminfo, line)) {
                    if (line.find("MemAvailable") != std::string::npos) {
                        std::istringstream ss(line);
                        std::string key;
                        ss >> key >> availableMemoryKB;
                        break;
                    }
                }
                meminfo.close();
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to read system memory info: " << e.what() << std::endl;
        }
#endif
        
        if (availableMemoryKB > 0) {
            std::cout << "Available system memory: " << availableMemoryKB << " KB" << std::endl;
            
            // Check if we have enough memory for our firmware
            if (availableMemoryKB < MEMORY_LIMIT_KB) {
                std::cerr << "Warning: Available memory " << availableMemoryKB 
                          << " KB is less than recommended " << MEMORY_LIMIT_KB << " KB" << std::endl;
            }
        } else {
            std::cout << "Memory limit set to: " << MEMORY_LIMIT_KB << " KB" << std::endl;
        }
        
        // Initialize hardware abstraction layer if needed
        // This would be platform-specific code to initialize hardware
        // such as GPU, NPU, or other accelerators
        
        // For now, we'll just log that we're using CPU inference
        std::cout << "Using CPU for inference operations" << std::endl;
        
        // Initialize the resource monitor with default configuration
        ResourceMonitorConfig resourceConfig;
        resourceConfig.enableMemoryTracking = true;
        resourceConfig.enableCpuTracking = true;
        resourceConfig.enableGpuTracking = false;  // Enable if GPU support is added
        resourceConfig.enableTemperatureTracking = false;
        resourceConfig.samplingIntervalMs = 1000;  // Sample every second
        resourceConfig.historySize = 60;  // Keep 1 minute of history
        
        g_resourceMonitor = std::make_unique<ResourceMonitor>(resourceConfig);
        
        // Start monitoring resources
        g_resourceMonitor->start();
        std::cout << "Resource monitoring started" << std::endl;
        
        // Register a custom resource for model loading time
        g_resourceMonitor->registerCustomResource("ModelLoadTime", 
            []() -> double { return 0.0; },  // Will be updated during model loading
            "ms");
        
        // Mark as initialized
        g_initialized = true;
        
        // Mark initialization event
        g_resourceMonitor->markEvent("Firmware Initialized");
        
        std::cout << "Firmware initialization complete" << std::endl;
        return CORTEX_FW_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return CORTEX_FW_ERROR_INIT_FAILED;
    }
}

int loadModel(const uint8_t* modelData, size_t modelSize) {
    if (!g_initialized) {
        std::cerr << "Firmware not initialized" << std::endl;
        return CORTEX_FW_ERROR_INIT_FAILED;
    }
    
    if (!modelData || modelSize == 0) {
        std::cerr << "Invalid model data or size" << std::endl;
        return CORTEX_FW_ERROR_INVALID_MODEL;
    }
    
    try {
        std::cout << "Loading model, size: " << modelSize << " bytes" << std::endl;
        
        // Mark model loading event in resource monitor
        if (g_resourceMonitor) {
            g_resourceMonitor->markEvent("Model Loading Started");
        }
        
        // Track memory usage during model loading
        MemoryTracker::recordAllocation(modelSize);
        
        // Create a memory buffer from the model data
        MemoryBuffer modelBuffer(modelData, modelSize);
        
        // Create model loader if needed
        if (!g_modelLoader) {
            g_modelLoader = std::make_unique<CortexAICompression::SDRModelLoader>();
        }
        
        // Load the model from memory
        auto loadStartTime = std::chrono::high_resolution_clock::now();
        
        // Use the model loader to load the model from memory
        bool loadSuccess = g_modelLoader->loadFromMemory(modelData, modelSize);
        
        if (!loadSuccess) {
            std::cerr << "Failed to load model" << std::endl;
            MemoryTracker::recordDeallocation(modelSize); // Release memory tracking
            
            if (g_resourceMonitor) {
                g_resourceMonitor->markEvent("Model Loading Failed");
            }
            
            return CORTEX_FW_ERROR_INVALID_MODEL;
        }
        
        // Create inference engine with the loaded model
        g_inferenceEngine = std::make_unique<CortexAICompression::SDRInferenceEngine>(
            g_modelLoader->getModel());
        
        auto loadEndTime = std::chrono::high_resolution_clock::now();
        auto loadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            loadEndTime - loadStartTime).count();
        
        // Update model loading time in resource monitor
        if (g_resourceMonitor) {
            // Update the custom resource for model loading time
            g_resourceMonitor->registerCustomResource("ModelLoadTime", 
                [loadDuration]() -> double { return static_cast<double>(loadDuration); },
                "ms");
            
            g_resourceMonitor->markEvent("Model Loading Completed");
        }
        
        std::cout << "Model loaded successfully in " << loadDuration << " ms" << std::endl;
        
        // Update peak memory usage
        updateMemoryUsage();
        
        return CORTEX_FW_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        
        // Clean up resources on failure
        g_inferenceEngine.reset();
        g_modelLoader.reset();
        
        if (g_resourceMonitor) {
            g_resourceMonitor->markEvent("Model Loading Exception: " + std::string(e.what()));
        }
        
        return CORTEX_FW_ERROR_INVALID_MODEL;
    }
}

int runInference(
    const float* inputData, 
    const std::vector<size_t>& inputShape,
    float* outputData,
    std::vector<size_t>& outputShape
) {
    if (!g_initialized || !g_inferenceEngine) {
        std::cerr << "Firmware not initialized or model not loaded" << std::endl;
        return CORTEX_FW_ERROR_INIT_FAILED;
    }
    
    try {
        // Validate input shape dimensions
        if (inputShape.size() > MAX_TENSOR_DIMS) {
        
        std::cout << "Inference completed in " << inferenceDuration << " ms" << std::endl;
        
        // Update memory usage
        updateMemoryUsage();
        
        // Check for resource warnings
        if (g_resourceMonitor) {
            ResourceUsage memoryUsage = g_resourceMonitor->getResourceUsage(ResourceType::MEMORY);
            ResourceUsage cpuUsage = g_resourceMonitor->getResourceUsage(ResourceType::CPU);
            
            // Log warnings if resources are running high during inference
            if (cpuUsage.current > 90.0) {
                std::cout << "WARNING: High CPU usage during inference: " << cpuUsage.current << "%" << std::endl;
            }
            
            if (memoryUsage.current > MEMORY_LIMIT_KB * 0.9) {
                std::cout << "WARNING: High memory usage during inference: " << memoryUsage.current 
                          << " KB (" << (memoryUsage.current * 100 / MEMORY_LIMIT_KB) << "% of limit)" << std::endl;
            }
        }
        
        // Clean up temporary memory
        MemoryTracker::recordDeallocation(inputSize * sizeof(float));
        
        return CORTEX_FW_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return CORTEX_FW_ERROR_INFERENCE_FAILED;
    }
}

// CPU usage tracking variables
static std::chrono::time_point<std::chrono::high_resolution_clock> g_lastCpuCheckTime;
static float g_lastCpuUsage = 0.0f;
static uint64_t g_lastTotalTime = 0;
static uint64_t g_lastActiveTime = 0;

int getResourceStats(uint32_t& memoryUsageKB, float& cpuUsagePercent) {
    if (!g_initialized) {
        std::cerr << "Firmware not initialized" << std::endl;
        return CORTEX_FW_ERROR_INIT_FAILED;
    }
    
    if (!g_resourceMonitor) {
        std::cerr << "Resource monitor not initialized" << std::endl;
        return CORTEX_FW_ERROR_INIT_FAILED;
    }
    
    try {
        // Get memory usage from resource monitor
        ResourceUsage memoryUsage = g_resourceMonitor->getResourceUsage(ResourceType::MEMORY);
        memoryUsageKB = static_cast<uint32_t>(memoryUsage.current);
        
        // Get CPU usage from resource monitor
        ResourceUsage cpuUsage = g_resourceMonitor->getResourceUsage(ResourceType::CPU);
        cpuUsagePercent = static_cast<float>(cpuUsage.current);
        
        // Update global peak memory usage for backward compatibility
        g_peakMemoryUsage = static_cast<uint32_t>(memoryUsage.peak);
        
        // Log additional resource information if available
        if (cpuUsage.current > 80.0) {
            std::cout << "WARNING: High CPU usage detected: " << cpuUsage.current << "%" << std::endl;
        }
        
        if (memoryUsage.current > MEMORY_LIMIT_KB * 0.9) {
            std::cout << "WARNING: High memory usage detected: " << memoryUsage.current 
                      << " KB (" << (memoryUsage.current * 100 / MEMORY_LIMIT_KB) << "% of limit)" << std::endl;
        }
        
        return CORTEX_FW_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error getting resource stats: " << e.what() << std::endl;
        
        // Fallback to legacy methods if resource monitor fails
        memoryUsageKB = estimateMemoryUsage();
        updateMemoryUsage();
        cpuUsagePercent = g_lastCpuUsage;
        
        return CORTEX_FW_ERROR_RESOURCE_STATS_FAILED;
    }
}

int shutdown() {
    if (!g_initialized) {
        std::cerr << "Firmware not initialized" << std::endl;
        return CORTEX_FW_ERROR_INIT_FAILED;
    }
    
    try {
        // Mark shutdown event in resource monitor
        if (g_resourceMonitor) {
            g_resourceMonitor->markEvent("Firmware Shutdown");
        }
        
        // Get final resource stats before shutdown
        uint32_t finalMemoryUsage = 0;
        float finalCpuUsage = 0.0f;
        getResourceStats(finalMemoryUsage, finalCpuUsage);
        
        // Print resource summary if available
        if (g_resourceMonitor) {
            std::cout << "\nResource Usage Summary:\n" << g_resourceMonitor->getResourceSummary() << std::endl;
        }
        
        // Clean up resources in reverse order of creation
        std::cout << "Shutting down inference engine..." << std::endl;
        g_inferenceEngine.reset();
        
        std::cout << "Shutting down model loader..." << std::endl;
        g_modelLoader.reset();
        
        // Stop and clean up resource monitor
        if (g_resourceMonitor) {
            std::cout << "Stopping resource monitor..." << std::endl;
            g_resourceMonitor->stop();
            g_resourceMonitor.reset();
        }
        
        // Reset initialization flag
        g_initialized = false;
        
        // Log final statistics
        auto totalRuntime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - g_startTime).count();
        
        std::cout << "Firmware shutdown complete" << std::endl;
        std::cout << "Total runtime: " << totalRuntime << " seconds" << std::endl;
        std::cout << "Peak memory usage: " << g_peakMemoryUsage << " KB" << std::endl;
        std::cout << "Final memory usage: " << finalMemoryUsage << " KB" << std::endl;
        
        return CORTEX_FW_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error during shutdown: " << e.what() << std::endl;
        
        // Try to clean up resources even if there was an error
        g_inferenceEngine.reset();
        g_modelLoader.reset();
        
        if (g_resourceMonitor) {
            g_resourceMonitor->stop();
            g_resourceMonitor.reset();
        }
        
        // Still mark as not initialized even if there was an error
        g_initialized = false;
        return CORTEX_FW_ERROR_INIT_FAILED;
    }
}

} // namespace CortexFirmware
