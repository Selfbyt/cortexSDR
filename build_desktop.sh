#!/bin/bash

# CortexSDR Desktop Application Build Script
# This script builds the Qt-based desktop application for benchmarking

set -e

echo "Building CortexSDR Desktop Application..."

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the cortexSDR root directory"
    exit 1
fi

# Check for Qt6 (required components)
if ! pkg-config --exists Qt6Core Qt6Widgets Qt6Network; then
    echo "Error: Qt6 development packages not found."
    echo "Please install Qt6 development packages:"
    echo "  Ubuntu/Debian: sudo apt-get install qt6-base-dev qt6-network-dev"
    echo "  Fedora: sudo dnf install qt6-qtbase-devel qt6-qtnetwork-devel"
    echo "  Arch: sudo pacman -S qt6-base qt6-network"
    exit 1
fi

# Check for Qt6 Charts (optional)
if pkg-config --exists Qt6Charts; then
    echo "Qt6 Charts found - charts will be available"
else
    echo "Qt6 Charts not found - charts will be disabled"
fi

# Create build directory
BUILD_DIR="build_desktop"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_DESKTOP=ON \
    -DENABLE_ONNX=ON \
    -DENABLE_GGUF=ON \
    -DENABLE_TENSORFLOW=ON \
    -DCMAKE_PREFIX_PATH=$(pkg-config --variable=prefix Qt6Core)

# Build
echo "Building desktop application..."
make -j$(nproc)

echo ""
echo "Build completed successfully!"
echo ""
echo "Desktop application: $BUILD_DIR/cortexsdr_desktop"
echo ""
echo "To run the application:"
echo "  cd $BUILD_DIR"
echo "  ./cortexsdr_desktop"
echo ""
echo "Features:"
echo "  - Model compression with configurable parameters"
echo "  - Text generation inference"
echo "  - Audio model inference"
echo "  - Performance benchmarking"
echo "  - Real-time system monitoring"
echo "  - Results visualization with charts"
echo "" 