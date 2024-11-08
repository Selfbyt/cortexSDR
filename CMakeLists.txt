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
    src/encoders/SpecialCharacterEncoding.cpp
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
