cmake_minimum_required(VERSION 3.16)
project(CortexSDRDesktop)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt6 components
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Network Charts)

# Find required packages
find_package(ZLIB REQUIRED)
find_package(Protobuf REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ai_compression
    ${CMAKE_SOURCE_DIR}/src/ai_compression/api
    ${CMAKE_SOURCE_DIR}/src/ai_compression/core
    ${CMAKE_SOURCE_DIR}/src/ai_compression/strategies
    ${CMAKE_SOURCE_DIR}/src/ai_compression/parsers
    ${CMAKE_SOURCE_DIR}/src/ai_compression/utils
    ${CMAKE_SOURCE_DIR}/src/ai_compression/streaming
    ${CMAKE_SOURCE_DIR}/src/ai_compression/onnx_proto
)

# Desktop application sources
set(DESKTOP_SOURCES
    main.cpp
    MainWindow.cpp
    ModelManager.cpp
    CompressionWidget.cpp
    InferenceWidget.cpp
    BenchmarkWidget.cpp
    ResultsWidget.cpp
    TextGenerationWidget.cpp
    AudioInferenceWidget.cpp
    ChatWidget.cpp
    ModelDownloader.cpp
    PerformanceMonitor.cpp
)

# Create executable
add_executable(cortexsdr_desktop ${DESKTOP_SOURCES})

# Link libraries
target_link_libraries(cortexsdr_desktop
    Qt6::Core
    Qt6::Widgets
    Qt6::Network
    Qt6::Charts
    cortexsdr
    ${PROTOBUF_LIBRARIES}
    ${ZLIB_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Set properties for Qt
set_target_properties(cortexsdr_desktop PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
)

# Install target
install(TARGETS cortexsdr_desktop
    RUNTIME DESTINATION bin
) 