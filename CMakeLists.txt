cmake_minimum_required(VERSION 3.10)
project(cortexsdr VERSION 1.0.0 LANGUAGES CXX)

# Build type options
option(BUILD_FIRMWARE "Build as firmware for embedded devices" OFF)
option(BUILD_DESKTOP "Build as desktop application" ON)
option(BUILD_LIBRARY "Build as a standalone library" OFF)
option(BUILD_PYTHON_WRAPPER "Build Python bindings" OFF)

# Set version information
set(CORTEXSDR_VERSION_MAJOR 1)
set(CORTEXSDR_VERSION_MINOR 0)
set(CORTEXSDR_VERSION_PATCH 0)
set(CORTEXSDR_VERSION "${CORTEXSDR_VERSION_MAJOR}.${CORTEXSDR_VERSION_MINOR}.${CORTEXSDR_VERSION_PATCH}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Firmware specific settings
if(BUILD_FIRMWARE)
    # Disable desktop dependencies
    set(BUILD_DESKTOP OFF)
    
    # Set firmware-specific compiler flags
    add_compile_options(-Os -ffunction-sections -fdata-sections)
    add_link_options(-Wl,--gc-sections)
    
    # Define firmware target
    add_compile_definitions(CORTEX_SDR_FIRMWARE)
endif()

# Desktop application settings
if(BUILD_DESKTOP)
    # Find Qt packages for desktop build
    find_package(Qt6 COMPONENTS Widgets REQUIRED)
    
    # Enable Qt features
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
endif()

# Define source files
set(COMMON_SOURCES
    # Text encoders
    src/encoders/text/WordEncoding.cpp
    src/encoders/text/CharacterEncoding.cpp
    src/encoders/text/SpecialCharEncoding.cpp
    
    # Numeric encoders
    src/encoders/numeric/NumberEncoding.cpp
    src/encoders/numeric/DateTimeEncoding.cpp
    src/encoders/numeric/GeoEncoding.cpp
    
    # Media encoders
    src/encoders/media/ImageEncoding.cpp
    src/encoders/media/AudioEncoding.cpp
    src/encoders/media/VideoEncoding.cpp
    
    # Adapters
    src/encoders/adapters/SpecialCharEncodingAdapter.cpp
    src/encoders/adapters/GeoEncodingAdapter.cpp
    # Core
    src/cortexSDR.cpp
    src/rle.cpp

    # AI Compression
    src/ai_compression/core/AICompressor.cpp
    src/ai_compression/core/AIDecompressor.cpp
    src/ai_compression/core/ArchiveConstants.cpp
    src/ai_compression/strategies/GzipStrategy.cpp
    src/ai_compression/strategies/NumericalRLE.cpp
    src/ai_compression/strategies/SDRIndexStorage.cpp
    src/ai_compression/strategies/MetadataSDRStrategy.cpp
    src/ai_compression/strategies/AdaptiveSDRStrategy.cpp
    src/ai_compression/streaming/StreamingCompressor.cpp
    src/ai_compression/parsers/GGUFModelParser.cpp
    src/ai_compression/parsers/ONNXModelParser.cpp
    src/ai_compression/utils/ModelConverter.cpp
    src/ai_compression/api/c_api.cpp
    src/ai_compression/onnx_proto/onnx.pb.cc
    src/ai_compression/SparseInferenceEngine.cpp
)

# Add UI sources only for desktop build
if(BUILD_DESKTOP)
    set(UI_SOURCES
        src/SDRWindow.cpp
    )
endif()

# Find required packages
find_package(ZLIB REQUIRED)
find_package(Protobuf REQUIRED) # Protobuf is often a dependency for ONNX
find_package(nlohmann_json REQUIRED)

# Optional model format support
option(ENABLE_GGUF "Enable GGUF model support" ON)
option(ENABLE_ONNX "Enable ONNX model support" ON)
option(ENABLE_TENSORFLOW "Enable TensorFlow model conversion support" ON)
option(ENABLE_PYTORCH "Enable PyTorch model conversion support" ON)

# Set the module path to include our custom FindONNXRuntime.cmake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- ONNX Configuration ---
set(ENABLE_ONNX_PROTOBUF OFF) # Default to OFF, enable if ONNX found
if(ENABLE_ONNX)
    # 1. Find ONNX Protobuf library (for model building/manipulation)
    find_package(ONNX) # Standard CMake find module for ONNX protobuf library
    if(ONNX_FOUND)
        message(STATUS "Found ONNX Protobuf library: ${ONNX_LIBRARIES}")
        message(STATUS "ONNX Protobuf includes: ${ONNX_INCLUDE_DIRS}")
        list(APPEND EXTRA_LIBS ${ONNX_LIBRARIES})
        list(APPEND EXTRA_INCLUDES ${ONNX_INCLUDE_DIRS})
        add_definitions(-DENABLE_ONNX_PROTOBUF) # Define if ONNX Protobuf is found
        set(ENABLE_ONNX_PROTOBUF ON)
    else()
        message(WARNING "ONNX Protobuf library not found. ONNX model reconstruction will not be possible.")
    endif()

    # 2. Find ONNX Runtime library (for inference/parsing)
    # Try to use local ONNX Runtime installation first
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/onnxruntime-linux-x64-gpu-1.21.1")
    if(EXISTS "${ONNXRUNTIME_ROOT}/include" AND EXISTS "${ONNXRUNTIME_ROOT}/lib")
        set(ONNXRuntime_FOUND TRUE)
        set(ONNXRuntime_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
        
        # Use only the main ONNX Runtime library, not the GPU providers to avoid CUDA dependencies
        set(ONNXRuntime_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
        
        add_definitions(-DENABLE_ONNX)
        list(APPEND EXTRA_LIBS ${ONNXRuntime_LIBRARIES})
        list(APPEND EXTRA_INCLUDES ${ONNXRuntime_INCLUDE_DIRS})
        
        # Make sure we include the ONNX Runtime headers in all targets that need it
        # We already added ONNX_INCLUDE_DIRS and ONNXRuntime_INCLUDE_DIRS to EXTRA_INCLUDES

        message(STATUS "Using local ONNX Runtime installation at ${ONNXRUNTIME_ROOT}")
        message(STATUS "ONNX Runtime library: ${ONNXRuntime_LIBRARIES}")
    else()
        # Fall back to system ONNX Runtime
        find_package(ONNXRuntime) # Use the custom Find module if needed
        if(ONNXRuntime_FOUND)
            add_definitions(-DENABLE_ONNX) # Keep this for runtime features
            list(APPEND EXTRA_LIBS ${ONNXRuntime_LIBRARIES})
            list(APPEND EXTRA_INCLUDES ${ONNXRuntime_INCLUDE_DIRS})
            message(STATUS "Found system ONNXRuntime: ${ONNXRuntime_LIBRARIES}")
            message(STATUS "ONNXRuntime include dirs: ${ONNXRuntime_INCLUDE_DIRS}")
        else()
            message(WARNING "ONNX Runtime not found. ONNX model parsing/runtime support will be disabled.")
            # We might still want ONNX Protobuf for reconstruction even if runtime fails
            if(NOT ONNX_FOUND)
                set(ENABLE_ONNX OFF) # Disable ONNX entirely if neither is found
            endif()
        endif()
    endif()
endif()
# --- End ONNX Configuration ---

# Check for TensorFlow if enabled
if(ENABLE_TENSORFLOW)
    # Use local TensorFlow installation
    set(TENSORFLOW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libtensorflow")
    if(EXISTS "${TENSORFLOW_ROOT}/include" AND EXISTS "${TENSORFLOW_ROOT}/lib")
        set(TensorFlow_FOUND TRUE)
        set(TensorFlow_INCLUDE_DIRS "${TENSORFLOW_ROOT}/include")
        set(TensorFlow_LIBRARIES "${TENSORFLOW_ROOT}/lib/libtensorflow.so" "${TENSORFLOW_ROOT}/lib/libtensorflow_framework.so")
        
        add_definitions(-DENABLE_TENSORFLOW)
        list(APPEND EXTRA_LIBS ${TensorFlow_LIBRARIES})
        list(APPEND EXTRA_INCLUDES ${TensorFlow_INCLUDE_DIRS})
        
        # Make sure we include the TensorFlow headers in all targets
        include_directories(${TensorFlow_INCLUDE_DIRS})
        
        message(STATUS "Using local TensorFlow installation at ${TENSORFLOW_ROOT}")
    else()
        message(WARNING "Local TensorFlow not found at ${TENSORFLOW_ROOT}. TensorFlow model conversion support will be disabled.")
        set(ENABLE_TENSORFLOW OFF)
    endif()
endif()

# Check for PyTorch if enabled
if(ENABLE_PYTORCH)
    # Use local PyTorch installation
    set(TORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/libtorch")
    if(EXISTS "${TORCH_ROOT}/include" AND EXISTS "${TORCH_ROOT}/lib")
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_ROOT}")
        find_package(Torch PATHS ${TORCH_ROOT} NO_DEFAULT_PATH)
        
        if(Torch_FOUND)
            add_definitions(-DENABLE_PYTORCH)
            list(APPEND EXTRA_LIBS ${TORCH_LIBRARIES})
            list(APPEND EXTRA_INCLUDES ${TORCH_INCLUDE_DIRS})
            message(STATUS "Using local PyTorch installation at ${TORCH_ROOT}")
        else()
            message(WARNING "Failed to find PyTorch at ${TORCH_ROOT}. PyTorch model conversion support will be disabled.")
            set(ENABLE_PYTORCH OFF)
        endif()
    else()
        message(WARNING "Local PyTorch not found at ${TORCH_ROOT}. PyTorch model conversion support will be disabled.")
        set(ENABLE_PYTORCH OFF)
    endif()
endif()

# Add library target
if(BUILD_FIRMWARE)
    add_library(${PROJECT_NAME} STATIC ${COMMON_SOURCES})
else()
    add_library(${PROJECT_NAME} ${COMMON_SOURCES} ${UI_SOURCES})
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ai_compression/onnx_proto
        ${EXTRA_INCLUDES} # Keep this for other includes
        ${Protobuf_INCLUDE_DIRS} # Explicitly add protobuf includes
        ${ONNX_INCLUDE_DIRS}     # Explicitly add ONNX includes
)

# Link libraries based on build type
if(BUILD_DESKTOP)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Qt6::Widgets
            Qt6::Widgets
            ZLIB::ZLIB
            ${Protobuf_LIBRARIES} # Link Protobuf libraries directly
            ${EXTRA_LIBS} # Contains ONNX and ONNXRuntime libs if found
    )
else()
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            ZLIB::ZLIB
            ${Protobuf_LIBRARIES} # Link Protobuf libraries directly
            ${EXTRA_LIBS} # Contains ONNX and ONNXRuntime libs if found
    )
endif()

# CLI Tool executable
find_package(ZLIB REQUIRED)
add_executable(cortexsdr_cli src/cli_main.cpp)
target_link_libraries(cortexsdr_cli PRIVATE ${PROJECT_NAME} ZLIB::ZLIB)

# Model Converter CLI Tool
add_executable(cortexsdr_model_converter src/model_converter_cli.cpp)
target_link_libraries(cortexsdr_model_converter PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})

# Add ONNX Runtime include directories to the model converter
if(ENABLE_ONNX)
    target_include_directories(cortexsdr_model_converter PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
endif()

# C API Test Tool
add_executable(test_c_api src/test_c_api.cpp)
target_link_libraries(test_c_api PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})

# AI Compression CLI Tool
add_executable(cortexsdr_ai_compression_cli src/ai_compression_cli.cpp)
target_link_libraries(cortexsdr_ai_compression_cli PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})

# Add tests unless building for firmware (which might have different constraints)
if(NOT BUILD_FIRMWARE)
    add_subdirectory(test)
endif()

# Firmware binary target
if(BUILD_FIRMWARE)
    # Create firmware binary
    add_executable(${PROJECT_NAME}_firmware src/firmware_main.cpp)
    target_link_libraries(${PROJECT_NAME}_firmware PRIVATE ${PROJECT_NAME})
    
    # Create hex file for flashing
    add_custom_command(TARGET ${PROJECT_NAME}_firmware POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:${PROJECT_NAME}_firmware> ${PROJECT_NAME}_firmware.hex
        COMMENT "Creating HEX file for firmware"
    )
endif()

# Python wrapper configuration
if(BUILD_PYTHON_WRAPPER)
    # Find Python and pybind11
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    
    # Create Python module
    pybind11_add_module(cortexsdr_py python/cortexsdr.cpp)
    target_link_libraries(cortexsdr_py PRIVATE ${PROJECT_NAME})
    set_target_properties(cortexsdr_py PROPERTIES OUTPUT_NAME cortexsdr)
    
    # Install Python module
    install(TARGETS cortexsdr_py 
            DESTINATION ${Python_SITEARCH}/cortexsdr)
    
    # Copy example file
    install(FILES python/example.py
            DESTINATION ${Python_SITEARCH}/cortexsdr/examples)
endif()

# Library configuration
if(BUILD_LIBRARY)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    
    # Generate version file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${CORTEXSDR_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    
    # Configure package config file
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )
    
    # Install targets
    install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    # Install headers (from include/ directory for public API)
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.hpp"
    )
    
    # Install export targets
    install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )
    
    # Install config files
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )
endif()
