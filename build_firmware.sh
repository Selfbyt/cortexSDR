#!/bin/bash

# Build script for cortexSDR firmware
echo "Building cortexSDR firmware..."

# Create build directory if it doesn't exist
mkdir -p build_firmware
cd build_firmware

# Configure with firmware options
cmake -DCMAKE_TOOLCHAIN_FILE=../firmware_config.cmake ..

# Build the firmware
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Firmware build successful!"
    echo "Firmware binary: $(pwd)/cortexsdr_firmware"
    echo "Firmware HEX file: $(pwd)/cortexsdr_firmware.hex"
else
    echo "Firmware build failed!"
    exit 1
fi

# Print firmware size information if objdump is available
if command -v objdump &> /dev/null; then
    echo "Firmware size information:"
    size cortexsdr_firmware
fi

echo "Build complete!"
