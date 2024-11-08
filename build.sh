#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Create build directory
mkdir -p build
cd build

# Build
echo -e "${GREEN}Building project...${NC}"
cmake ..
make -j$(nproc)

echo -e "${GREEN}Build complete!${NC}"
