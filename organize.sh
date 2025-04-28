#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Core structure
create_structure() {
    local dirs=(
        "include"      # Public API headers
        "src"         # Source files
        "test"        # Test UI and unit tests
        "build"       # Build output
    )

    echo -e "${BLUE}Creating core structure...${NC}"
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        echo -e "${GREEN}Created: $dir${NC}"
    done

    # Create src subdirectories
    mkdir -p src/encoders

    # Setup minimal test directory
    mkdir -p test/build
}

# Generate build files
setup_build() {
    # Main CMakeLists.txt
    cat > CMakeLists.txt << 'EOL'
cmake_minimum_required(VERSION 3.10)
project(cortexsdr VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library target
add_library(${PROJECT_NAME}
    src/encoders/AudioEncoding.cpp
    src/encoders/DateTimeEncoding.cpp
    src/encoders/ImageEncoding.cpp
    src/encoders/NumberEncoding.cpp
    src/encoders/VideoEncoding.cpp
    src/encoders/WordEncoding.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Test UI
add_subdirectory(test)
EOL

    # Test CMakeLists.txt
    cat > test/CMakeLists.txt << 'EOL'
find_package(Qt6 COMPONENTS Widgets REQUIRED)

add_executable(sdr_test
    SDRWindow.cpp
    test.cpp
)

target_link_libraries(sdr_test
    PRIVATE
        cortexsdr
        Qt6::Widgets
)
EOL
}

# Setup git configuration
setup_git() {
    # Optimized .gitignore
    cat > .gitignore << 'EOL'
build/
.vscode/
.idea/
*.o
moc_*
*.pro.user
*.user
.DS_Store
EOL

    git init
    git add .
    git commit -m "Initial structure"
}

# Build script
create_build_script() {
    cat > build.sh << 'EOL'
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
EOL

    chmod +x build.sh
}

# Main execution
main() {
    echo -e "${BLUE}Setting up CortexSDR...${NC}"
    
    create_structure
    setup_build
    setup_git
    create_build_script
    
    echo -e "${GREEN}Setup complete! Structure created:${NC}"
    tree -L 2
}

# Check for required tools
check_requirements() {
    local required_tools=("cmake" "git" "g++")
    local missing_tools=()

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing required tools: ${missing_tools[*]}${NC}"
        exit 1
    fi
}

# Execute with error handling
{
    check_requirements
    main
} || {
    echo -e "${RED}Setup failed!${NC}"
    exit 1
}