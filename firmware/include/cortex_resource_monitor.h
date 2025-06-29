#ifndef CORTEX_RESOURCE_MONITOR_H
#define CORTEX_RESOURCE_MONITOR_H

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <functional>

namespace CortexFirmware {

// Resource types that can be monitored
enum class ResourceType {
    MEMORY,
    CPU,
    GPU,
    TEMPERATURE,
    POWER,
    NETWORK,
    DISK,
    CUSTOM
};

// Resource usage data structure
struct ResourceUsage {
    double current;      // Current usage value
    double peak;         // Peak usage value
    double average;      // Average usage value
    std::string unit;    // Unit of measurement (e.g., "KB", "%", "W")
    
    ResourceUsage() : current(0), peak(0), average(0), unit("") {}
};

// Resource monitoring configuration
struct ResourceMonitorConfig {
    bool enableMemoryTracking = true;
    bool enableCpuTracking = true;
    bool enableGpuTracking = false;
    bool enableTemperatureTracking = false;
    bool enablePowerTracking = false;
    bool enableNetworkTracking = false;
    bool enableDiskTracking = false;
    
    uint32_t samplingIntervalMs = 1000;  // How often to sample resource usage
    uint32_t historySize = 60;           // How many samples to keep in history
};

// Forward declaration of implementation class
class ResourceMonitorImpl;

/**
 * ResourceMonitor - A class to monitor system resources during inference
 * 
 * This class provides functionality to track memory usage, CPU usage,
 * and other system resources during model inference. It supports both
 * real-time monitoring and historical data collection.
 */
class ResourceMonitor {
public:
    // Constructor with optional configuration
    explicit ResourceMonitor(const ResourceMonitorConfig& config = ResourceMonitorConfig());
    
    // Destructor
    ~ResourceMonitor();
    
    // Start monitoring resources
    void start();
    
    // Stop monitoring resources
    void stop();
    
    // Get current usage for a specific resource
    ResourceUsage getResourceUsage(ResourceType type);
    
    // Get historical usage data for a specific resource
    std::vector<ResourceUsage> getResourceHistory(ResourceType type);
    
    // Register a custom resource to monitor
    void registerCustomResource(const std::string& name, 
                               std::function<double()> measurementFunction,
                               const std::string& unit);
    
    // Get current usage for a custom resource
    ResourceUsage getCustomResourceUsage(const std::string& name);
    
    // Mark a significant event for correlation with resource usage
    void markEvent(const std::string& eventName);
    
    // Get a summary of all resource usage
    std::string getResourceSummary();
    
    // Reset peak measurements
    void resetPeaks();
    
    // Check if a resource exceeds a threshold
    bool isResourceExceeded(ResourceType type, double threshold);
    
    // Set a callback to be called when a resource exceeds a threshold
    void setThresholdCallback(ResourceType type, double threshold, 
                             std::function<void(ResourceType, double)> callback);

private:
    // Pointer to implementation (PIMPL pattern)
    std::unique_ptr<ResourceMonitorImpl> impl_;
};

} // namespace CortexFirmware

#endif // CORTEX_RESOURCE_MONITOR_H
