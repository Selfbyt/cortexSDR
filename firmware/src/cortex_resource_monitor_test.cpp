#include "../include/cortex_firmware.h"
#include "../include/cortex_resource_monitor.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>

// Test function to stress CPU
void stressCPU(int durationMs) {
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        // Perform some CPU-intensive calculations
        double result = 0;
        for (int i = 0; i < 10000; i++) {
            result += std::sin(i) * std::cos(i);
        }
        
        // Prevent compiler from optimizing out the calculation
        if (result == 0.12345) {
            std::cout << "This will never print" << std::endl;
        }
        
        // Check if we've reached the duration
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= durationMs) {
            break;
        }
    }
}

// Test function to allocate memory
void* allocateMemory(size_t sizeKB) {
    size_t sizeBytes = sizeKB * 1024;
    void* ptr = malloc(sizeBytes);
    if (ptr) {
        // Touch the memory to ensure it's actually allocated
        memset(ptr, 1, sizeBytes);
    }
    return ptr;
}

// Test the resource monitor
int main() {
    std::cout << "Starting Resource Monitor Test" << std::endl;
    
    // Initialize the firmware
    int result = CortexFirmware::initialize();
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to initialize firmware" << std::endl;
        return 1;
    }
    
    std::cout << "Firmware initialized successfully" << std::endl;
    
    // Get initial resource stats
    uint32_t initialMemoryKB = 0;
    float initialCpuPercent = 0.0f;
    result = CortexFirmware::getResourceStats(initialMemoryKB, initialCpuPercent);
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to get initial resource stats" << std::endl;
        CortexFirmware::shutdown();
        return 1;
    }
    
    std::cout << "Initial Memory Usage: " << initialMemoryKB << " KB" << std::endl;
    std::cout << "Initial CPU Usage: " << initialCpuPercent << "%" << std::endl;
    
    // Test 1: Stress CPU
    std::cout << "\nTest 1: Stressing CPU for 5 seconds..." << std::endl;
    stressCPU(5000);
    
    // Get resource stats after CPU stress
    uint32_t cpuStressMemoryKB = 0;
    float cpuStressPercent = 0.0f;
    result = CortexFirmware::getResourceStats(cpuStressMemoryKB, cpuStressPercent);
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to get resource stats after CPU stress" << std::endl;
        CortexFirmware::shutdown();
        return 1;
    }
    
    std::cout << "Memory Usage after CPU stress: " << cpuStressMemoryKB << " KB" << std::endl;
    std::cout << "CPU Usage after CPU stress: " << cpuStressPercent << "%" << std::endl;
    
    // Test 2: Allocate memory
    std::cout << "\nTest 2: Allocating 100MB of memory..." << std::endl;
    void* memPtr = allocateMemory(100 * 1024); // 100MB
    if (!memPtr) {
        std::cerr << "Failed to allocate memory" << std::endl;
        CortexFirmware::shutdown();
        return 1;
    }
    
    // Get resource stats after memory allocation
    uint32_t memAllocMemoryKB = 0;
    float memAllocCpuPercent = 0.0f;
    result = CortexFirmware::getResourceStats(memAllocMemoryKB, memAllocCpuPercent);
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to get resource stats after memory allocation" << std::endl;
        free(memPtr);
        CortexFirmware::shutdown();
        return 1;
    }
    
    std::cout << "Memory Usage after allocation: " << memAllocMemoryKB << " KB" << std::endl;
    std::cout << "CPU Usage after allocation: " << memAllocCpuPercent << "%" << std::endl;
    
    // Free the allocated memory
    free(memPtr);
    
    // Test 3: Simulate model loading and inference
    std::cout << "\nTest 3: Simulating model loading and inference..." << std::endl;
    
    // Create a dummy model (just an array of floats)
    const size_t modelSize = 10 * 1024 * 1024; // 10MB
    uint8_t* dummyModel = new uint8_t[modelSize];
    for (size_t i = 0; i < modelSize; i++) {
        dummyModel[i] = static_cast<uint8_t>(i % 256);
    }
    
    // Load the dummy model
    std::cout << "Loading dummy model..." << std::endl;
    result = CortexFirmware::loadModel(dummyModel, modelSize);
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to load dummy model" << std::endl;
        delete[] dummyModel;
        CortexFirmware::shutdown();
        return 1;
    }
    
    // Get resource stats after model loading
    uint32_t modelLoadMemoryKB = 0;
    float modelLoadCpuPercent = 0.0f;
    result = CortexFirmware::getResourceStats(modelLoadMemoryKB, modelLoadCpuPercent);
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to get resource stats after model loading" << std::endl;
        delete[] dummyModel;
        CortexFirmware::shutdown();
        return 1;
    }
    
    std::cout << "Memory Usage after model loading: " << modelLoadMemoryKB << " KB" << std::endl;
    std::cout << "CPU Usage after model loading: " << modelLoadCpuPercent << "%" << std::endl;
    
    // Create dummy input and output data for inference
    const size_t inputSize = 1024;
    const size_t outputSize = 512;
    float* inputData = new float[inputSize];
    float* outputData = new float[outputSize];
    
    for (size_t i = 0; i < inputSize; i++) {
        inputData[i] = static_cast<float>(i) / inputSize;
    }
    
    // Run inference
    std::cout << "Running inference..." << std::endl;
    result = CortexFirmware::runInference(inputData, inputSize, outputData, outputSize);
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to run inference" << std::endl;
        delete[] dummyModel;
        delete[] inputData;
        delete[] outputData;
        CortexFirmware::shutdown();
        return 1;
    }
    
    // Get resource stats after inference
    uint32_t inferenceMemoryKB = 0;
    float inferenceCpuPercent = 0.0f;
    result = CortexFirmware::getResourceStats(inferenceMemoryKB, inferenceCpuPercent);
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to get resource stats after inference" << std::endl;
        delete[] dummyModel;
        delete[] inputData;
        delete[] outputData;
        CortexFirmware::shutdown();
        return 1;
    }
    
    std::cout << "Memory Usage after inference: " << inferenceMemoryKB << " KB" << std::endl;
    std::cout << "CPU Usage after inference: " << inferenceCpuPercent << "%" << std::endl;
    
    // Clean up
    delete[] dummyModel;
    delete[] inputData;
    delete[] outputData;
    
    // Shutdown the firmware
    std::cout << "\nShutting down firmware..." << std::endl;
    result = CortexFirmware::shutdown();
    if (result != CortexFirmware::CORTEX_FW_SUCCESS) {
        std::cerr << "Failed to shutdown firmware" << std::endl;
        return 1;
    }
    
    std::cout << "Resource Monitor Test completed successfully" << std::endl;
    return 0;
}
