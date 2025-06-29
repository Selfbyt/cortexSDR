#include "../include/cortex_resource_monitor.h"
#include "../include/cortex_resource_monitor_impl.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace CortexFirmware {

// Forward declaration of implementation class
class ResourceMonitorImpl;

// ResourceMonitor implementation
ResourceMonitor::ResourceMonitor(const ResourceMonitorConfig& config)
    : impl_(new ResourceMonitorImpl(config)) {
}

ResourceMonitor::~ResourceMonitor() {
    // Smart pointer will handle cleanup
}

void ResourceMonitor::start() {
    impl_->start();
}

void ResourceMonitor::stop() {
    impl_->stop();
}

ResourceUsage ResourceMonitor::getResourceUsage(ResourceType type) {
    return impl_->getResourceUsage(type);
}

std::vector<ResourceUsage> ResourceMonitor::getResourceHistory(ResourceType type) {
    return impl_->getResourceHistory(type);
}

void ResourceMonitor::registerCustomResource(const std::string& name, 
                                          std::function<double()> measurementFunction,
                                          const std::string& unit) {
    impl_->registerCustomResource(name, measurementFunction, unit);
}

ResourceUsage ResourceMonitor::getCustomResourceUsage(const std::string& name) {
    return impl_->getCustomResourceUsage(name);
}

void ResourceMonitor::markEvent(const std::string& eventName) {
    impl_->markEvent(eventName);
}

std::string ResourceMonitor::getResourceSummary() {
    return impl_->getResourceSummary();
}

void ResourceMonitor::resetPeaks() {
    impl_->resetPeaks();
}

bool ResourceMonitor::isResourceExceeded(ResourceType type, double threshold) {
    return impl_->isResourceExceeded(type, threshold);
}

void ResourceMonitor::setThresholdCallback(ResourceType type, double threshold, 
                                        std::function<void(ResourceType, double)> callback) {
    impl_->setThresholdCallback(type, threshold, callback);
}

} // namespace CortexFirmware
