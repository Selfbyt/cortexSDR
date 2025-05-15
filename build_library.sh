#!/bin/bash

# Build script for cortexSDR library
echo "Building cortexSDR as a library..."

# Create build directory if it doesn't exist
mkdir -p build_library
cd build_library

# Configure with library options
cmake -DBUILD_LIBRARY=ON -DBUILD_DESKTOP=OFF -DBUILD_FIRMWARE=OFF -DCMAKE_INSTALL_PREFIX=/usr/local ..

# Build the library
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Library build successful!"
    echo ""
    echo "To install the library system-wide, run:"
    echo "sudo make install"
    echo ""
    echo "To use in your CMake project, add:"
    echo "find_package(cortexsdr REQUIRED)"
    echo "target_link_libraries(your_target PRIVATE cortexsdr::cortexsdr)"
else
    echo "Library build failed!"
    exit 1
fi

echo "Build complete!"
