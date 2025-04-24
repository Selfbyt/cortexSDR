#!/bin/bash

# Build script for cortexSDR library example
echo "Building cortexSDR library example..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the example
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Example build successful!"
    echo ""
    echo "To run the example:"
    echo "./sdr_example"
else
    echo "Example build failed!"
    exit 1
fi

echo "Build complete!"
