#pragma once

#include "cortex_resource_monitor.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>

namespace CortexFirmware {

class ResourceMonitorImpl {
public:
    ResourceMonitorImpl(const ResourceMonitorConfig& config);
    ~ResourceMonitorImpl();

    void start();
    void stop();
    
    ResourceUsage getResourceUsage(ResourceType type);
    std::vector<ResourceUsage> getResourceHistory(ResourceType type);
    std::string getResourceSummary();
    ResourceUsage getCustomResourceUsage(const std::string& name);
    
    void markEvent(const std::string& eventName);
    void registerCustomResource(const std::string& name, 
                               std::function<double()> measureFunction,
                               const std::string& units);
    
    void setThresholdCallback(ResourceType type, double threshold, 
                               std::function<void(ResourceType, double)> callback);
    
    void resetPeaks();
    bool isResourceExceeded(ResourceType type, double threshold);

private:
    // Configuration
    ResourceMonitorConfig config;
    
    // Resource data storage
    struct ResourceData {
        std::string name;
        std::string unit;
        double current = 0.0;
        double peak = 0.0;
        std::vector<double> history;
        size_t historyIndex = 0;
    };
    
    std::map<ResourceType, ResourceData> resourceUsage;
    
    // Custom resources
    struct CustomResource {
        std::string name;
        std::string unit;
        std::function<double()> measurementFunction;
        double current = 0.0;
        double peak = 0.0;
        std::vector<double> history;
        size_t historyIndex = 0;
    };
    std::map<std::string, CustomResource> customResources;
    
    // Event tracking
    struct Event {
        std::chrono::system_clock::time_point timestamp;
        std::string name;
    };
    std::vector<Event> events;
    
    // Threshold monitoring
    struct ThresholdCallback {
        double threshold;
        std::function<void(ResourceType, double)> callback;
        bool triggered = false;
    };
    std::map<ResourceType, ThresholdCallback> thresholds;
    
    // Monitoring thread
    std::thread monitorThread;
    std::atomic<bool> running;
    std::mutex resourceMutex;
    
    // Monitoring functions
    void monitorLoop();
    void updateResourceUsage();
    void checkThresholds();
    
    // Resource measurement functions
    double measureMemoryUsage();
    double measureCpuUsage();
    double measureGpuUsage();
    double measureTemperature();
    double measurePowerUsage();
    double measureNetworkUsage();
    double measureDiskUsage();
    
    // Helper functions
    void updateResourceData(ResourceType type, double value);
    void updateCustomResourceData(const std::string& name);
    double calculateAverage(const std::vector<double>& values);
    
    // CPU usage tracking variables
    unsigned long long lastTotalUser = 0;
    unsigned long long lastTotalUserLow = 0;
    unsigned long long lastTotalSys = 0;
    unsigned long long lastTotalIdle = 0;
};

} // namespace CortexFirmware
