#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Clean build if requested
if [ "$1" == "clean" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Check for local libraries
ONNX_FLAGS="-DENABLE_ONNX=ON"
PYTORCH_FLAGS="-DENABLE_PYTORCH=ON"
TF_FLAGS="-DENABLE_TENSORFLOW=ON"

# Build with all available ML libraries
echo -e "${GREEN}Configuring project with ONNX, PyTorch, and TensorFlow support...${NC}"
cmake $ONNX_FLAGS $PYTORCH_FLAGS $TF_FLAGS ..

# Build with multiple cores if available
NUM_CORES=$(nproc)
if [ $NUM_CORES -gt 1 ]; then
    make -j$NUM_CORES
else
    make
fi

echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}Executables are in the build directory:${NC}"
echo -e "  - ${YELLOW}cortexsdr_cli${NC}: Main CLI tool"
echo -e "  - ${YELLOW}cortexsdr_model_converter${NC}: Tool to convert models to ONNX format"

# Remind about sparsity parameter
echo -e "\n${GREEN}Remember:${NC} You can use the ${YELLOW}--sparsity${NC} or ${YELLOW}-s${NC} parameter with cortexsdr_cli to control the fraction of active bits in the SDR encoding (default 2%)."
echo -e "This helps tune compression and achieve higher compression ratios."
