cmake_minimum_required(VERSION 3.10)
project(cortexsdr_example VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the cortexSDR package
find_package(cortexsdr REQUIRED)

# Create example executable
add_executable(sdr_example main.cpp)

# Link against cortexSDR
target_link_libraries(sdr_example PRIVATE cortexsdr::cortexsdr)

# Print status
message(STATUS "cortexSDR found: ${cortexsdr_VERSION}")
message(STATUS "cortexSDR include dirs: ${cortexsdr_INCLUDE_DIRS}")
message(STATUS "cortexSDR libraries: ${cortexsdr_LIBRARIES}")
