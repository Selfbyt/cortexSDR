#include "../include/cortex_resource_monitor.h"
#include "../include/cortex_resource_monitor_impl.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <cstring>

using namespace CortexFirmware;

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
    std::cout << "Starting Standalone Resource Monitor Test" << std::endl;
    
    // Initialize the resource monitor with default configuration
    ResourceMonitorConfig config;
    config.enableMemoryTracking = true;
    config.enableCpuTracking = true;
    config.enableGpuTracking = false;
    config.enableTemperatureTracking = false;
    config.samplingIntervalMs = 500;  // Sample every half second
    config.historySize = 60;  // Keep 30 seconds of history
    
    ResourceMonitor monitor(config);
    
    // Start monitoring
    monitor.start();
    std::cout << "Resource monitoring started" << std::endl;
    
    // Get initial resource stats
    ResourceUsage initialMemory = monitor.getResourceUsage(ResourceType::MEMORY);
    ResourceUsage initialCpu = monitor.getResourceUsage(ResourceType::CPU);
    
    std::cout << "Initial Memory Usage: " << initialMemory.current << " KB" << std::endl;
    std::cout << "Initial CPU Usage: " << initialCpu.current << "%" << std::endl;
    
    // Test 1: Stress CPU
    std::cout << "\nTest 1: Stressing CPU for 5 seconds..." << std::endl;
    monitor.markEvent("CPU Stress Test Started");
    stressCPU(5000);
    monitor.markEvent("CPU Stress Test Completed");
    
    // Get resource stats after CPU stress
    ResourceUsage cpuStressMemory = monitor.getResourceUsage(ResourceType::MEMORY);
    ResourceUsage cpuStressCpu = monitor.getResourceUsage(ResourceType::CPU);
    
    std::cout << "Memory Usage after CPU stress: " << cpuStressMemory.current << " KB" << std::endl;
    std::cout << "CPU Usage after CPU stress: " << cpuStressCpu.current << "%" << std::endl;
    
    // Test 2: Allocate memory
    std::cout << "\nTest 2: Allocating 100MB of memory..." << std::endl;
    monitor.markEvent("Memory Allocation Test Started");
    void* memPtr = allocateMemory(100 * 1024); // 100MB
    if (!memPtr) {
        std::cerr << "Failed to allocate memory" << std::endl;
        monitor.stop();
        return 1;
    }
    
    // Sleep to allow resource monitor to detect the change
    std::this_thread::sleep_for(std::chrono::seconds(2));
    monitor.markEvent("Memory Allocation Test Completed");
    
    // Get resource stats after memory allocation
    ResourceUsage memAllocMemory = monitor.getResourceUsage(ResourceType::MEMORY);
    ResourceUsage memAllocCpu = monitor.getResourceUsage(ResourceType::CPU);
    
    std::cout << "Memory Usage after allocation: " << memAllocMemory.current << " KB" << std::endl;
    std::cout << "CPU Usage after allocation: " << memAllocCpu.current << "%" << std::endl;
    
    // Free the allocated memory
    free(memPtr);
    
    // Test 3: Custom resource tracking
    std::cout << "\nTest 3: Testing custom resource tracking..." << std::endl;
    
    // Register a custom resource
    int customValue = 0;
    monitor.registerCustomResource("TestMetric", 
        [&customValue]() -> double { return customValue; },
        "units");
    
    // Update the custom resource over time
    for (int i = 0; i < 10; i++) {
        customValue = i * 10;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        if (i == 5) {
            monitor.markEvent("Custom Resource Halfway Point");
        }
    }
    
    // Get the resource summary
    std::cout << "\nResource Usage Summary:" << std::endl;
    std::cout << monitor.getResourceSummary() << std::endl;
    
    // Stop monitoring
    std::cout << "\nStopping resource monitor..." << std::endl;
    monitor.stop();
    
    std::cout << "Standalone Resource Monitor Test completed successfully" << std::endl;
    return 0;
}
