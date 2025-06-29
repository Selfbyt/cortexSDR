#!/bin/bash

# Simple script to build and run the standalone resource monitor test

echo "Building standalone resource monitor test..."

# Create build directory if it doesn't exist
mkdir -p build_test

# Compile the test
g++ -std=c++17 -o build_test/standalone_resource_monitor_test \
    src/standalone_resource_monitor_test.cpp \
    src/cortex_resource_monitor.cpp \
    src/cortex_resource_monitor_impl.cpp \
    -I./include -I../src -I../include -pthread

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Running test..."
    echo "-----------------------------------"
    ./build_test/standalone_resource_monitor_test
    echo "-----------------------------------"
    echo "Test completed."
else
    echo "Build failed!"
    exit 1
fi
