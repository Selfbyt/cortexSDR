cmake_minimum_required(VERSION 3.10)
project(cortexsdr VERSION 1.0.0 LANGUAGES CXX)

# Build type options
option(BUILD_FIRMWARE "Build as firmware for embedded devices" OFF)
option(BUILD_DESKTOP "Build as desktop application" OFF)
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
    
    # Include firmware configuration
    include(${CMAKE_CURRENT_SOURCE_DIR}/firmware/config/firmware_config.cmake)
endif()

# Desktop application settings removed

# Define source files
set(COMMON_SOURCES
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
    src/ai_compression/parsers/TensorFlowModelParser.cpp
    src/ai_compression/parsers/PyTorchModelParser.cpp
    src/ai_compression/parsers/CoreMLModelParser.cpp
    src/ai_compression/parsers/HDF5ModelParser.cpp
    src/ai_compression/parsers/ModelParserFactory.cpp
    src/ai_compression/utils/ModelConverter.cpp
    src/ai_compression/utils/sha256.cpp
    src/ai_compression/api/c_api.cpp
    src/ai_compression/onnx_proto/onnx.pb.cc
    src/ai_compression/SparseInferenceEngine.cpp
    # Optimized Kernels
    src/ai_compression/kernels/blas_kernels.cpp
    src/ai_compression/kernels/simd_kernels.cpp
    src/ai_compression/kernels/conv_kernels.cpp
    src/ai_compression/kernels/fused_kernels.cpp
    src/ai_compression/kernels/quantized_kernels.cpp
    src/ai_compression/kernels/attention_kernels.cpp
    src/ai_compression/kernels/sparse_kernels.cpp
)

# Remove desktop UI and application sources

# Find required packages
find_package(ZLIB REQUIRED)
find_package(Protobuf REQUIRED) # Protobuf is often a dependency for ONNX
find_package(nlohmann_json REQUIRED)

# libzip (for reading ZIP-based PyTorch checkpoints like consolidated.xx.pth)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZIP REQUIRED libzip)
if(LIBZIP_FOUND)
    list(APPEND EXTRA_LIBS ${LIBZIP_LIBRARIES})
    list(APPEND EXTRA_INCLUDES ${LIBZIP_INCLUDE_DIRS})
    message(STATUS "Found libzip: ${LIBZIP_LIBRARIES}")
    message(STATUS "libzip include dirs: ${LIBZIP_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "libzip not found. Please install libzip (e.g., 'dnf install libzip libzip-devel').")
endif()

# Optional model format support
# Note: TensorFlow is OFF by default to avoid heavy C++ dependency setup.
# To enable, configure with: -DENABLE_TENSORFLOW=ON and set TENSORFLOW_SOURCE_DIR to your TF source path.
option(ENABLE_GGUF "Enable GGUF model support" ON)
option(ENABLE_ONNX "Enable ONNX model support" ON)
option(ENABLE_TENSORFLOW "Enable TensorFlow model conversion support" OFF)
option(ENABLE_PYTORCH "Enable PyTorch model conversion support" ON)
option(ENABLE_COREML "Enable CoreML model support" ON)
option(ENABLE_HDF5 "Enable HDF5/Keras model support" ON)

# Performance optimization options
option(ENABLE_BLAS "Enable BLAS acceleration for linear algebra" ON)
option(USE_MKL "Use Intel MKL for BLAS (highest performance)" OFF)
option(USE_OPENBLAS "Use OpenBLAS for BLAS" ON)
option(ENABLE_SIMD "Enable SIMD optimizations (SSE/AVX)" ON)

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

# --- HDF5 Configuration ---
if(ENABLE_HDF5)
    find_package(HDF5 REQUIRED)
    if(HDF5_FOUND)
        add_definitions(-DENABLE_HDF5)
        list(APPEND EXTRA_LIBS ${HDF5_LIBRARIES})
        list(APPEND EXTRA_INCLUDES ${HDF5_INCLUDE_DIRS})
        message(STATUS "Found HDF5: ${HDF5_LIBRARIES}")
        message(STATUS "HDF5 include dirs: ${HDF5_INCLUDE_DIRS}")
    else()
        message(WARNING "HDF5 not found. HDF5/Keras model support will be disabled.")
        set(ENABLE_HDF5 OFF)
    endif()
endif()
# --- End HDF5 Configuration ---

# --- BLAS Configuration for Performance ---
if(ENABLE_BLAS)
    if(USE_MKL)
        # Intel MKL provides best performance
        find_package(MKL)
        if(MKL_FOUND)
            add_definitions(-DUSE_MKL -DENABLE_BLAS)
            list(APPEND EXTRA_LIBS ${MKL_LIBRARIES})
            list(APPEND EXTRA_INCLUDES ${MKL_INCLUDE_DIRS})
            message(STATUS "Using Intel MKL for BLAS: ${MKL_LIBRARIES}")
            set(BLAS_FOUND TRUE)
        else()
            message(WARNING "Intel MKL not found, trying OpenBLAS...")
            set(USE_MKL OFF)
            set(USE_OPENBLAS ON)
        endif()
    endif()
    
    if(USE_OPENBLAS AND NOT BLAS_FOUND)
        # OpenBLAS is a good open-source alternative
        find_package(OpenBLAS)
        if(NOT OpenBLAS_FOUND)
            # Try pkg-config
            pkg_check_modules(OpenBLAS openblas)
        endif()
        
        if(OpenBLAS_FOUND)
            add_definitions(-DUSE_OPENBLAS -DENABLE_BLAS)
            list(APPEND EXTRA_LIBS ${OpenBLAS_LIBRARIES})
            list(APPEND EXTRA_INCLUDES ${OpenBLAS_INCLUDE_DIRS})
            link_directories(${OpenBLAS_LIBRARY_DIRS})
            message(STATUS "Using OpenBLAS: ${OpenBLAS_LIBRARIES}")
            set(BLAS_FOUND TRUE)
        else()
            message(WARNING "OpenBLAS not found, falling back to standard BLAS...")
        endif()
    endif()
    
    if(NOT BLAS_FOUND)
        # Fall back to standard BLAS/LAPACK
        find_package(BLAS)
        if(BLAS_FOUND)
            add_definitions(-DENABLE_BLAS)
            list(APPEND EXTRA_LIBS ${BLAS_LIBRARIES})
            message(STATUS "Using standard BLAS: ${BLAS_LIBRARIES}")
            set(BLAS_FOUND TRUE)
        else()
            message(WARNING "No BLAS library found. Performance will be degraded.")
            message(WARNING "Install OpenBLAS with: sudo dnf install openblas openblas-devel")
            set(ENABLE_BLAS OFF)
        endif()
    endif()
endif()
# --- End BLAS Configuration ---

# Check for TensorFlow if enabled
if(ENABLE_TENSORFLOW)
    # Allow user to specify TensorFlow source directory via environment variable or CMake variable
    if(NOT DEFINED TENSORFLOW_SOURCE_DIR)
        if(DEFINED ENV{TENSORFLOW_SOURCE_DIR})
            set(TENSORFLOW_SOURCE_DIR "$ENV{TENSORFLOW_SOURCE_DIR}")
        else()
            # Try common locations
            set(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow")
        endif()
    endif()
    
    # Check if TensorFlow source directory exists
    if(EXISTS "${TENSORFLOW_SOURCE_DIR}/tensorflow/cc/saved_model/loader.h")
        set(TensorFlow_FOUND TRUE)
        set(TensorFlow_INCLUDE_DIRS 
            "${TENSORFLOW_SOURCE_DIR}"
            "${TENSORFLOW_SOURCE_DIR}/bazel-bin"
            "${TENSORFLOW_SOURCE_DIR}/bazel-tensorflow/external/com_google_protobuf/src"
            "${TENSORFLOW_SOURCE_DIR}/bazel-tensorflow/external/org_tensorflow"
        )
        
        # Try to find TensorFlow libraries
        find_library(TensorFlow_LIBRARY
            NAMES tensorflow_cc tensorflow
            PATHS
                "${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow"
                "${TENSORFLOW_SOURCE_DIR}/bazel-tensorflow/bin/tensorflow"
                /usr/lib
                /usr/lib64
                /usr/local/lib
                /usr/local/lib64
        )
        
        find_library(TensorFlow_FRAMEWORK_LIBRARY
            NAMES tensorflow_framework
            PATHS
                "${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow"
                "${TENSORFLOW_SOURCE_DIR}/bazel-tensorflow/bin/tensorflow"
                /usr/lib
                /usr/lib64
                /usr/local/lib
                /usr/local/lib64
        )
        
        if(TensorFlow_LIBRARY)
            set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
            if(TensorFlow_FRAMEWORK_LIBRARY)
                list(APPEND TensorFlow_LIBRARIES ${TensorFlow_FRAMEWORK_LIBRARY})
            endif()
        else()
            # Fallback to system libraries if available
            find_library(TensorFlow_LIBRARY
                NAMES tensorflow_cc tensorflow
                PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64
            )
            if(TensorFlow_LIBRARY)
                set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
            endif()
        endif()
        
        add_definitions(-DENABLE_TENSORFLOW)
        list(APPEND EXTRA_LIBS ${TensorFlow_LIBRARIES})
        list(APPEND EXTRA_INCLUDES ${TensorFlow_INCLUDE_DIRS})
        
        # Make sure we include the TensorFlow headers in all targets
        include_directories(${TensorFlow_INCLUDE_DIRS})
        
        message(STATUS "Using TensorFlow source directory at ${TENSORFLOW_SOURCE_DIR}")
        message(STATUS "TensorFlow include directories: ${TensorFlow_INCLUDE_DIRS}")
        message(STATUS "TensorFlow libraries: ${TensorFlow_LIBRARIES}")
    else()
        message(WARNING "TensorFlow source directory not found at ${TENSORFLOW_SOURCE_DIR}")
        message(WARNING "Set TENSORFLOW_SOURCE_DIR environment variable or CMake variable to point to your TensorFlow source directory")
        message(WARNING "TensorFlow model conversion support will be disabled.")
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

# Define SDK sources
set(SDK_SOURCES
    src/ai_compression/api/cortex_sdk.cpp
    src/ai_compression/api/c_api.cpp
    src/ai_compression/SparseInferenceEngine.cpp
    src/ai_compression/parsers/ONNXModelParser.cpp
    src/ai_compression/parsers/TensorFlowModelParser.cpp
    src/ai_compression/parsers/PyTorchModelParser.cpp
    src/ai_compression/parsers/CoreMLModelParser.cpp
    src/ai_compression/parsers/HDF5ModelParser.cpp
    src/ai_compression/parsers/ModelParserFactory.cpp
    src/ai_compression/onnx_proto/onnx.pb.cc
    src/ai_compression/core/AIDecompressor.cpp
    src/ai_compression/core/AICompressor.cpp
    src/ai_compression/core/ArchiveConstants.cpp
    src/ai_compression/streaming/StreamingCompressor.cpp
    src/ai_compression/utils/ModelConverter.cpp
    src/ai_compression/strategies/NumericalRLE.cpp
    src/ai_compression/strategies/GzipStrategy.cpp
    src/ai_compression/utils/sha256.cpp
    src/ai_compression/strategies/AdaptiveSDRStrategy.cpp
    src/ai_compression/strategies/MetadataSDRStrategy.cpp
    src/ai_compression/strategies/QuantizedTensorStrategy.cpp
    src/ai_compression/strategies/SDRIndexStorage.cpp
)

# Add main library target
add_library(${PROJECT_NAME} ${COMMON_SOURCES})

# Add SDK library targets (both static and shared)
add_library(${PROJECT_NAME}_sdk STATIC ${SDK_SOURCES})
add_library(${PROJECT_NAME}_sdk_shared SHARED ${SDK_SOURCES})

# Set proper visibility flags for the shared library
set_target_properties(${PROJECT_NAME}_sdk_shared PROPERTIES 
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# Add compiler definition for shared library
target_compile_definitions(${PROJECT_NAME}_sdk_shared PRIVATE CORTEXSDR_BUILDING_SHARED_LIBRARY=1)

# Add include directories for SDK targets
foreach(target ${PROJECT_NAME}_sdk ${PROJECT_NAME}_sdk_shared)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ai_compression
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ai_compression/onnx_proto
            ${EXTRA_INCLUDES}
            ${Protobuf_INCLUDE_DIRS}
            ${ONNX_INCLUDE_DIRS}
    )

    # Link libraries for SDK targets
    target_link_libraries(${target}
        PRIVATE
            ZLIB::ZLIB
            ${Protobuf_LIBRARIES}
            ${EXTRA_LIBS}
    )
endforeach()

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
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ZLIB::ZLIB
        ${Protobuf_LIBRARIES}
        ${EXTRA_LIBS}
)

# CLI Tool executable removed (moved to sdr_core)

# Model Converter CLI Tool
add_executable(cortexsdr_model_converter src/model_converter_cli.cpp)
target_link_libraries(cortexsdr_model_converter PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})

# Add ONNX Runtime include directories to the model converter
if(ENABLE_ONNX)
    target_include_directories(cortexsdr_model_converter PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
endif()

# C API Test Tool removed

# AI Compression CLI Tool
add_executable(cortexsdr_ai_compression_cli src/ai_compression_cli.cpp)
target_link_libraries(cortexsdr_ai_compression_cli PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})

# Text Generation CLI Tool (Ollama-like)
add_executable(cortexsdr_text_cli src/cortexsdr_cli.cpp)
target_link_libraries(cortexsdr_text_cli PRIVATE ${PROJECT_NAME} ${PROJECT_NAME}_sdk ${EXTRA_LIBS})

# Desktop application target removed

# Tests removed

# Firmware binary target
if(BUILD_FIRMWARE)
    # Include the firmware-specific CMakeLists.txt
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/firmware)
    
    # Create a symbolic link to the firmware binary in the main build directory
    add_custom_command(TARGET cortexsdr_firmware_${CORTEX_TARGET_PLATFORM} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}>
            ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_firmware
        COMMENT "Creating symbolic link to firmware binary"
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
