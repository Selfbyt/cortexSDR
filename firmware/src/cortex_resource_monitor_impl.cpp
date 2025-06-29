#include "../include/cortex_resource_monitor_impl.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <functional>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace CortexFirmware {

// Implementation of ResourceMonitorImpl methods
ResourceMonitorImpl::ResourceMonitorImpl(const ResourceMonitorConfig& config) 
    : config(config), running(false) {
    // Initialize resource maps
    resourceUsage[ResourceType::MEMORY] = ResourceData{"Memory", "KB"};
    resourceUsage[ResourceType::CPU] = ResourceData{"CPU", "%"};
    resourceUsage[ResourceType::GPU] = ResourceData{"GPU", "%"};
    resourceUsage[ResourceType::TEMPERATURE] = ResourceData{"Temperature", "°C"};
    resourceUsage[ResourceType::POWER] = ResourceData{"Power", "W"};
    resourceUsage[ResourceType::NETWORK] = ResourceData{"Network", "KB/s"};
    resourceUsage[ResourceType::DISK] = ResourceData{"Disk", "KB/s"};
    
    // Initialize history storage based on config
    for (auto& [type, data] : resourceUsage) {
        data.history.resize(config.historySize);
    }
    
    // Initialize CPU measurement variables
    lastTotalUser = 0;
    lastTotalSys = 0;
}

ResourceMonitorImpl::~ResourceMonitorImpl() {
    stop();
}

void ResourceMonitorImpl::start() {
    if (running) return;
    
    running = true;
    monitorThread = std::thread(&ResourceMonitorImpl::monitorLoop, this);
}

void ResourceMonitorImpl::stop() {
    if (!running) return;
    
    running = false;
    if (monitorThread.joinable()) {
        monitorThread.join();
    }
}

ResourceUsage ResourceMonitorImpl::getResourceUsage(ResourceType type) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    if (resourceUsage.find(type) == resourceUsage.end()) {
        return ResourceUsage();
    }
    
    ResourceUsage usage;
    usage.current = resourceUsage[type].current;
    usage.peak = resourceUsage[type].peak;
    usage.average = calculateAverage(resourceUsage[type].history);
    usage.unit = resourceUsage[type].unit;
    return usage;
}

std::vector<ResourceUsage> ResourceMonitorImpl::getResourceHistory(ResourceType type) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    if (resourceUsage.find(type) == resourceUsage.end()) {
        return {};
    }
    
    std::vector<ResourceUsage> history;
    for (double value : resourceUsage[type].history) {
        ResourceUsage usage;
        usage.current = value;
        usage.unit = resourceUsage[type].unit;
        history.push_back(usage);
    }
    return history;
}

void ResourceMonitorImpl::registerCustomResource(const std::string& name, 
                          std::function<double()> measurementFunction,
                          const std::string& unit) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    customResources[name] = CustomResource{
        name,
        unit,
        measurementFunction,
        0.0,  // current
        0.0,  // peak
        std::vector<double>(config.historySize, 0.0),  // history
        0     // historyIndex
    };
}

ResourceUsage ResourceMonitorImpl::getCustomResourceUsage(const std::string& name) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    if (customResources.find(name) == customResources.end()) {
        return ResourceUsage();
    }
    
    ResourceUsage usage;
    usage.current = customResources[name].current;
    usage.peak = customResources[name].peak;
    usage.average = calculateAverage(customResources[name].history);
    usage.unit = customResources[name].unit;
    return usage;
}

void ResourceMonitorImpl::markEvent(const std::string& eventName) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    events.push_back({
        std::chrono::system_clock::now(),
        eventName
    });
    
    std::cout << "[ResourceMonitor] Event marked: " << eventName << std::endl;
}

std::string ResourceMonitorImpl::getResourceSummary() {
    std::lock_guard<std::mutex> lock(resourceMutex);
    std::stringstream ss;
    
    ss << "=== Resource Usage Summary ===\n";
    
    // Standard resources
    for (const auto& [type, data] : resourceUsage) {
        if (data.current > 0) {
            ss << data.name << ": " 
               << std::fixed << std::setprecision(2) << data.current << " " << data.unit
               << " (Peak: " << data.peak << " " << data.unit << ")\n";
        }
    }
    
    // Custom resources
    for (const auto& [name, resource] : customResources) {
        ss << name << ": " 
           << std::fixed << std::setprecision(2) << resource.current << " " << resource.unit
           << " (Peak: " << resource.peak << " " << resource.unit << ")\n";
    }
    
    return ss.str();
}

void ResourceMonitorImpl::resetPeaks() {
    std::lock_guard<std::mutex> lock(resourceMutex);
    
    for (auto& [type, data] : resourceUsage) {
        data.peak = data.current;
    }
    
    for (auto& [name, resource] : customResources) {
        resource.peak = resource.current;
    }
}

bool ResourceMonitorImpl::isResourceExceeded(ResourceType type, double threshold) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    if (resourceUsage.find(type) == resourceUsage.end()) {
        return false;
    }
    
    return resourceUsage[type].current > threshold;
}

void ResourceMonitorImpl::setThresholdCallback(ResourceType type, double threshold, 
                        std::function<void(ResourceType, double)> callback) {
    std::lock_guard<std::mutex> lock(resourceMutex);
    thresholds[type] = {threshold, callback};
}

double ResourceMonitorImpl::calculateAverage(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

void ResourceMonitorImpl::updateResourceData(ResourceType type, double value) {
    if (resourceUsage.find(type) == resourceUsage.end()) return;
    
    auto& data = resourceUsage[type];
    data.current = value;
    data.peak = std::max(data.peak, value);
    
    // Update history
    data.history[data.historyIndex] = value;
    data.historyIndex = (data.historyIndex + 1) % data.history.size();
}

void ResourceMonitorImpl::updateCustomResourceData(const std::string& name) {
    if (customResources.find(name) == customResources.end()) return;
    
    auto& resource = customResources[name];
    double value = resource.measurementFunction();
    
    resource.current = value;
    resource.peak = std::max(resource.peak, value);
    
    // Update history
    resource.history[resource.historyIndex] = value;
    resource.historyIndex = (resource.historyIndex + 1) % resource.history.size();
}

void ResourceMonitorImpl::monitorLoop() {
    while (running) {
        // Collect resource usage data
        updateResourceUsage();
        
        // Check thresholds and call callbacks if needed
        checkThresholds();
        
        // Sleep for the configured interval
        std::this_thread::sleep_for(std::chrono::milliseconds(config.samplingIntervalMs));
    }
}

void ResourceMonitorImpl::updateResourceUsage() {
    std::lock_guard<std::mutex> lock(resourceMutex);
    
    // Update memory usage
    if (config.enableMemoryTracking) {
        double memoryUsage = measureMemoryUsage();
        updateResourceData(ResourceType::MEMORY, memoryUsage);
    }
    
    // Update CPU usage
    if (config.enableCpuTracking) {
        double cpuUsage = measureCpuUsage();
        updateResourceData(ResourceType::CPU, cpuUsage);
    }
    
    // Update GPU usage if enabled
    if (config.enableGpuTracking) {
        double gpuUsage = measureGpuUsage();
        updateResourceData(ResourceType::GPU, gpuUsage);
    }
    
    // Update temperature if enabled
    if (config.enableTemperatureTracking) {
        double temperature = measureTemperature();
        updateResourceData(ResourceType::TEMPERATURE, temperature);
    }
    
    // Update power usage if enabled
    if (config.enablePowerTracking) {
        double power = measurePowerUsage();
        updateResourceData(ResourceType::POWER, power);
    }
    
    // Update network usage if enabled
    if (config.enableNetworkTracking) {
        double network = measureNetworkUsage();
        updateResourceData(ResourceType::NETWORK, network);
    }
    
    // Update disk usage if enabled
    if (config.enableDiskTracking) {
        double disk = measureDiskUsage();
        updateResourceData(ResourceType::DISK, disk);
    }
    
    // Update custom resources
    for (const auto& [name, _] : customResources) {
        updateCustomResourceData(name);
    }
}

void ResourceMonitorImpl::checkThresholds() {
    std::lock_guard<std::mutex> lock(resourceMutex);
    
    for (const auto& [type, callback] : thresholds) {
        if (resourceUsage.find(type) != resourceUsage.end()) {
            double value = resourceUsage[type].current;
            if (value > callback.threshold) {
                // Call the callback outside the lock to prevent deadlocks
                resourceMutex.unlock();
                callback.callback(type, value);
                resourceMutex.lock();
            }
        }
    }
}

// Platform-specific resource measurement methods
double ResourceMonitorImpl::measureMemoryUsage() {
    // Get process memory usage
    double memoryUsageKB = 0.0;
    
#ifdef __linux__
    // On Linux, read from /proc/self/status
    try {
        std::ifstream statusFile("/proc/self/status");
        if (statusFile.is_open()) {
            std::string line;
            while (std::getline(statusFile, line)) {
                if (line.find("VmRSS:") != std::string::npos) {
                    std::istringstream ss(line);
                    std::string key;
                    ss >> key >> memoryUsageKB;
                    break;
                }
            }
            statusFile.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading memory usage: " << e.what() << std::endl;
    }
#else
    // Fallback for other platforms
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        memoryUsageKB = usage.ru_maxrss;
    }
#endif

    return memoryUsageKB;
}

double ResourceMonitorImpl::measureCpuUsage() {
    double cpuUsage = 0.0;
    
#ifdef __linux__
    // On Linux, read from /proc/stat
    try {
        std::ifstream statFile("/proc/stat");
        if (statFile.is_open()) {
            std::string line;
            std::getline(statFile, line);
            statFile.close();
            
            std::istringstream ss(line);
            std::string cpu;
            uint64_t user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
            
            ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
            
            uint64_t idleAllTime = idle + iowait;
            uint64_t systemAllTime = system + irq + softirq;
            uint64_t virtAllTime = guest + guest_nice;
            uint64_t totalTime = user + nice + systemAllTime + idleAllTime + steal + virtAllTime;
            
            uint64_t activeTime = totalTime - idleAllTime;
            
            if (lastTotalUser > 0) {
                uint64_t totalTimeDelta = totalTime - lastTotalUser;
                uint64_t activeTimeDelta = activeTime - lastTotalSys;
                
                if (totalTimeDelta > 0) {
                    cpuUsage = 100.0 * activeTimeDelta / totalTimeDelta;
                }
            }
            
            lastTotalUser = totalTime;
            lastTotalSys = activeTime;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading CPU usage: " << e.what() << std::endl;
    }
#else
    // Fallback for other platforms
    cpuUsage = 0.0;  // Not implemented
#endif

    return cpuUsage;
}

double ResourceMonitorImpl::measureGpuUsage() {
    // GPU usage measurement is platform-specific and requires external libraries
    // This is a placeholder implementation
    return 0.0;
}

double ResourceMonitorImpl::measureTemperature() {
    // Temperature measurement is platform-specific
    // This is a placeholder implementation
    return 0.0;
}

double ResourceMonitorImpl::measurePowerUsage() {
    // Power usage measurement is platform-specific
    // This is a placeholder implementation
    return 0.0;
}

double ResourceMonitorImpl::measureNetworkUsage() {
    // Network usage measurement is platform-specific
    // This is a placeholder implementation
    return 0.0;
}

double ResourceMonitorImpl::measureDiskUsage() {
    // Disk usage measurement is platform-specific
    // This is a placeholder implementation
    return 0.0;
}

} // namespace CortexFirmware
