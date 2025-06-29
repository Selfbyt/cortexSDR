#!/bin/bash

# Build script for cortexSDR firmware
echo "Building cortexSDR firmware..."

# Create build directory if it doesn't exist
mkdir -p build_firmware
cd build_firmware

# Determine target platform
TARGET="SERVER_X86_64"

while getopts "t:" opt; do
  case $opt in
    t) TARGET=$OPTARG ;;
    *) echo "Usage: $0 [-t TARGET]" >&2
       echo "Available targets: SERVER_X86_64, ARM_CORTEX_M4, RISC_V" >&2
       exit 1 ;;
  esac
done

echo "Building for target: $TARGET"

# Configure with firmware options
cmake -DCMAKE_TOOLCHAIN_FILE=../firmware/config/firmware_config.cmake \
      -DCORTEX_TARGET_PLATFORM=$TARGET \
      -DCMAKE_BUILD_TYPE=Release ..

# Build the firmware
make -j$(nproc) cortexsdr_firmware_$TARGET

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Firmware build successful!"
    echo "Firmware binary: $(pwd)/cortexsdr_firmware_$TARGET"
    
    # Generate HEX file for embedded targets
    if [ "$TARGET" != "SERVER_X86_64" ] && command -v objcopy &> /dev/null; then
        objcopy -O ihex cortexsdr_firmware_$TARGET cortexsdr_firmware_$TARGET.hex
        echo "Firmware HEX file: $(pwd)/cortexsdr_firmware_$TARGET.hex"
    fi
else
    echo "Firmware build failed!"
    exit 1
fi

# Print firmware size information
if command -v size &> /dev/null; then
    echo "Firmware size information:"
    size cortexsdr_firmware_$TARGET
fi

echo "Build complete!"
