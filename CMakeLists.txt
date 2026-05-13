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

# CPU-oriented performance defaults
if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/Oi> $<$<CONFIG:Release>:/Ot> $<$<CONFIG:Release>:/GL> $<$<CONFIG:Release>:/fp:fast>)
    add_compile_options($<$<CONFIG:RelWithDebInfo>:/O2> $<$<CONFIG:RelWithDebInfo>:/Oi> $<$<CONFIG:RelWithDebInfo>:/Ot> $<$<CONFIG:RelWithDebInfo>:/GL> $<$<CONFIG:RelWithDebInfo>:/fp:fast>)
    add_link_options($<$<CONFIG:Release>:/LTCG> $<$<CONFIG:RelWithDebInfo>:/LTCG>)
else()
    add_compile_options(-O3 -ffast-math)
endif()

# OpenMP — used for the per-tile encode in HSDR fit. Each tile's MP encode is
# independent, so scattering tiles across cores gives near-linear speedup.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found — enabling parallel encode")
    set(CORTEXSDR_OPENMP_TARGET OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found — falling back to single-threaded encode")
    set(CORTEXSDR_OPENMP_TARGET "")
endif()

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
    src/ai_compression/strategies/SparseWeightReconstruction.cpp
    src/ai_compression/strategies/MetadataSDRStrategy.cpp
    src/ai_compression/strategies/AdaptiveSDRStrategy.cpp
    src/ai_compression/strategies/QuantizedTensorStrategy.cpp
    src/ai_compression/strategies/HierarchicalSDRStrategy.cpp
    src/ai_compression/strategies/DequantToFP32Strategy.cpp
    src/ai_compression/streaming/StreamingCompressor.cpp
    src/ai_compression/parsers/GGUFModelParser.cpp
    src/ai_compression/parsers/ONNXModelParser.cpp
    src/ai_compression/parsers/TensorFlowModelParser.cpp
    src/ai_compression/parsers/PyTorchModelParser.cpp
    src/ai_compression/parsers/HDF5ModelParser.cpp
    src/ai_compression/parsers/ModelParserFactory.cpp
    src/ai_compression/LLMTokenizer.cpp
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
    src/ai_compression/kernels/flash_attention.cpp
)

# Remove desktop UI and application sources

# Find required packages
find_package(ZLIB REQUIRED)
find_package(Protobuf REQUIRED) # Protobuf is often a dependency for ONNX
find_package(nlohmann_json REQUIRED)
if(MSVC)
    # Normalize protobuf link inputs to import libraries (.lib) on Windows.
    # Some package configurations may expose runtime DLL paths, which the linker cannot consume directly.
    set(_PROTOBUF_LIBS_FIXED "")
    foreach(_plib ${Protobuf_LIBRARIES})
        if(_plib MATCHES ".*/debug/bin/.*\\.dll$")
            string(REPLACE "/debug/bin/" "/debug/lib/" _plib_lib "${_plib}")
            string(REGEX REPLACE "\\.dll$" ".lib" _plib_lib "${_plib_lib}")
            list(APPEND _PROTOBUF_LIBS_FIXED "${_plib_lib}")
        elseif(_plib MATCHES ".*/bin/.*\\.dll$")
            string(REPLACE "/bin/" "/lib/" _plib_lib "${_plib}")
            string(REGEX REPLACE "\\.dll$" ".lib" _plib_lib "${_plib_lib}")
            list(APPEND _PROTOBUF_LIBS_FIXED "${_plib_lib}")
        else()
            list(APPEND _PROTOBUF_LIBS_FIXED "${_plib}")
        endif()
    endforeach()
    set(Protobuf_LIBRARIES ${_PROTOBUF_LIBS_FIXED})
endif()
set(PROTOBUF_LINK_LIBS ${Protobuf_LIBRARIES})
if(TARGET protobuf::libprotobuf)
    set(PROTOBUF_LINK_LIBS protobuf::libprotobuf)
elseif(TARGET Protobuf::libprotobuf)
    set(PROTOBUF_LINK_LIBS Protobuf::libprotobuf)
endif()

# libzip (for reading ZIP-based PyTorch checkpoints like consolidated.xx.pth)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZIP REQUIRED libzip)
if(LIBZIP_FOUND)
    if(LIBZIP_LIBRARY_DIRS)
        link_directories(${LIBZIP_LIBRARY_DIRS})
        if(EXISTS "${LIBZIP_LIBRARY_DIRS}/../debug/lib")
            link_directories("${LIBZIP_LIBRARY_DIRS}/../debug/lib")
        endif()
    endif()
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
option(ENABLE_TENSORFLOW "Enable TensorFlow model parsing support" OFF)
option(ENABLE_PYTORCH "Enable PyTorch model parsing support" ON)
option(ENABLE_HDF5 "Enable HDF5/Keras model support" ON)

# Performance optimization options
option(ENABLE_BLAS "Enable BLAS acceleration for linear algebra" ON)
option(USE_MKL "Use Intel MKL for BLAS (highest performance)" OFF)
option(USE_OPENBLAS "Use OpenBLAS for BLAS" ON)
option(ENABLE_SIMD "Enable SIMD optimizations (SSE/AVX)" ON)
set(CORTEXSDR_SIMD_LEVEL "AUTO" CACHE STRING "SIMD target for CPU builds: AUTO, AVX2, SSE4_1, SSE2")

# Set the module path to include our custom FindONNXRuntime.cmake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- ONNX Configuration ---
# ONNX protobuf support is required in this project.
set(ENABLE_ONNX_PROTOBUF ON)
add_definitions(-DENABLE_ONNX_PROTOBUF)
if(ENABLE_ONNX)
    # 1. Find ONNX Protobuf library (for model building/manipulation)
    find_package(ONNX) # Standard CMake find module for ONNX protobuf library
    if(ONNX_FOUND)
        message(STATUS "Found ONNX Protobuf library: ${ONNX_LIBRARIES}")
        message(STATUS "ONNX Protobuf includes: ${ONNX_INCLUDE_DIRS}")
        list(APPEND EXTRA_LIBS ${ONNX_LIBRARIES})
        list(APPEND EXTRA_INCLUDES ${ONNX_INCLUDE_DIRS})
    else()
        message(WARNING "ONNX CMake package not found. Continuing with bundled ONNX protobuf sources.")
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
    if(MSVC)
        # Help CMake find vcpkg-installed BLAS packages on Windows even when a
        # Config package is not provided.
        set(_VCPKG_X64_ROOT "C:/tools/vcpkg/installed/x64-windows")
        if(EXISTS "${_VCPKG_X64_ROOT}/include")
            list(APPEND CMAKE_PREFIX_PATH "${_VCPKG_X64_ROOT}")
        endif()
    endif()

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
        if(NOT OpenBLAS_FOUND)
            find_path(OpenBLAS_INCLUDE_DIRS
                NAMES cblas.h
                PATHS
                    "C:/tools/vcpkg/installed/x64-windows/include"
                    /usr/include
                    /usr/local/include
            )
            find_library(OpenBLAS_LIBRARIES
                NAMES openblas libopenblas
                PATHS
                    "C:/tools/vcpkg/installed/x64-windows/lib"
                    "C:/tools/vcpkg/installed/x64-windows/debug/lib"
                    /usr/lib
                    /usr/lib64
                    /usr/local/lib
                    /usr/local/lib64
            )
            if(OpenBLAS_INCLUDE_DIRS AND OpenBLAS_LIBRARIES)
                set(OpenBLAS_FOUND TRUE)
                set(OpenBLAS_LIBRARY_DIRS "")
            endif()
        endif()
        
        if(OpenBLAS_FOUND)
            add_definitions(-DUSE_OPENBLAS -DENABLE_BLAS)
            list(APPEND EXTRA_LIBS ${OpenBLAS_LIBRARIES})
            list(APPEND EXTRA_INCLUDES ${OpenBLAS_INCLUDE_DIRS})
            if(OpenBLAS_LIBRARY_DIRS)
                link_directories(${OpenBLAS_LIBRARY_DIRS})
            endif()
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

# --- SIMD Configuration ---
if(ENABLE_SIMD)
    if(MSVC)
        if(CORTEXSDR_SIMD_LEVEL STREQUAL "AUTO" OR CORTEXSDR_SIMD_LEVEL STREQUAL "AVX2")
            add_compile_options(/arch:AVX2)
            add_compile_definitions(CORTEXSDR_SIMD_AVX2=1)
            message(STATUS "SIMD target: AVX2")
        elseif(CORTEXSDR_SIMD_LEVEL STREQUAL "SSE4_1")
            add_compile_options(/arch:SSE2)
            add_compile_definitions(CORTEXSDR_SIMD_SSE41=1)
            message(STATUS "SIMD target: SSE4.1")
        else()
            add_compile_options(/arch:SSE2)
            add_compile_definitions(CORTEXSDR_SIMD_SSE2=1)
            message(STATUS "SIMD target: SSE2")
        endif()
    else()
        if(CORTEXSDR_SIMD_LEVEL STREQUAL "AUTO" OR CORTEXSDR_SIMD_LEVEL STREQUAL "AVX2")
            add_compile_options(-mavx2 -mfma)
            add_compile_definitions(CORTEXSDR_SIMD_AVX2=1)
            message(STATUS "SIMD target: AVX2")
        elseif(CORTEXSDR_SIMD_LEVEL STREQUAL "SSE4_1")
            add_compile_options(-msse4.1)
            add_compile_definitions(CORTEXSDR_SIMD_SSE41=1)
            message(STATUS "SIMD target: SSE4.1")
        else()
            add_compile_options(-msse2)
            add_compile_definitions(CORTEXSDR_SIMD_SSE2=1)
            message(STATUS "SIMD target: SSE2")
        endif()
    endif()
endif()

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
        message(WARNING "TensorFlow model parsing support will be disabled.")
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
            message(WARNING "Failed to find PyTorch at ${TORCH_ROOT}. PyTorch model parsing support will be disabled.")
            set(ENABLE_PYTORCH OFF)
        endif()
    else()
        message(WARNING "Local PyTorch not found at ${TORCH_ROOT}. PyTorch model parsing support will be disabled.")
        set(ENABLE_PYTORCH OFF)
    endif()
endif()

# Define SDK sources
set(SDK_SOURCES
    src/ai_compression/api/cortex_sdk.cpp
    src/ai_compression/api/c_api.cpp
    src/ai_compression/SparseInferenceEngine.cpp
    src/ai_compression/parsers/GGUFModelParser.cpp
    src/ai_compression/parsers/ONNXModelParser.cpp
    src/ai_compression/parsers/TensorFlowModelParser.cpp
    src/ai_compression/parsers/PyTorchModelParser.cpp
    src/ai_compression/parsers/HDF5ModelParser.cpp
    src/ai_compression/parsers/ModelParserFactory.cpp
    src/ai_compression/onnx_proto/onnx.pb.cc
    src/ai_compression/core/AIDecompressor.cpp
    src/ai_compression/core/AICompressor.cpp
    src/ai_compression/core/ArchiveConstants.cpp
    src/ai_compression/streaming/StreamingCompressor.cpp
    src/ai_compression/strategies/NumericalRLE.cpp
    src/ai_compression/strategies/GzipStrategy.cpp
    src/ai_compression/utils/sha256.cpp
    src/ai_compression/strategies/AdaptiveSDRStrategy.cpp
    src/ai_compression/strategies/MetadataSDRStrategy.cpp
    src/ai_compression/strategies/QuantizedTensorStrategy.cpp
    src/ai_compression/strategies/HierarchicalSDRStrategy.cpp
    src/ai_compression/strategies/SDRIndexStorage.cpp
    src/ai_compression/strategies/SparseWeightReconstruction.cpp
    src/ai_compression/kernels/blas_kernels.cpp
    src/ai_compression/kernels/simd_kernels.cpp
    src/ai_compression/kernels/conv_kernels.cpp
    src/ai_compression/kernels/fused_kernels.cpp
    src/ai_compression/kernels/quantized_kernels.cpp
    src/ai_compression/kernels/attention_kernels.cpp
    src/ai_compression/kernels/sparse_kernels.cpp
    src/ai_compression/kernels/flash_attention.cpp
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
target_compile_definitions(${PROJECT_NAME} PUBLIC CORTEXSDR_STATIC=1)
target_compile_definitions(${PROJECT_NAME}_sdk PUBLIC CORTEXSDR_STATIC=1)

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
            ${PROTOBUF_LINK_LIBS}
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
        ${PROTOBUF_LINK_LIBS}
        ${EXTRA_LIBS}
        ${CORTEXSDR_OPENMP_TARGET}
)

# CLI Tool executable removed (moved to sdr_core)

# C API Test Tool removed

# AI Compression CLI Tool
add_executable(cortexsdr_ai_compression_cli src/ai_compression_cli.cpp)
target_link_libraries(cortexsdr_ai_compression_cli PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(cortexsdr_ai_compression_cli PRIVATE CORTEXSDR_STATIC=1)

# Text Generation CLI Tool (Ollama-like)
add_executable(cortexsdr_text_cli src/cortexsdr_cli.cpp)
target_link_libraries(cortexsdr_text_cli PRIVATE ${PROJECT_NAME} ${PROJECT_NAME}_sdk ${EXTRA_LIBS})
target_compile_definitions(cortexsdr_text_cli PRIVATE CORTEXSDR_STATIC=1)

# Benchmark CLI Tool
add_executable(cortexsdr_bench src/cortexsdr_bench.cpp)
target_link_libraries(cortexsdr_bench PRIVATE ${PROJECT_NAME} ${PROJECT_NAME}_sdk ${EXTRA_LIBS})
target_compile_definitions(cortexsdr_bench PRIVATE CORTEXSDR_STATIC=1)

# HSDR (V4b) round-trip smoke test
add_executable(test_hsdr_roundtrip src/test_hsdr_roundtrip.cpp)
target_link_libraries(test_hsdr_roundtrip PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsdr_roundtrip PRIVATE CORTEXSDR_STATIC=1)

# HSDR archive end-to-end test (compress → write archive → read back → fused matmul)
add_executable(test_hsdr_archive_e2e src/test_hsdr_archive_e2e.cpp)
target_link_libraries(test_hsdr_archive_e2e PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsdr_archive_e2e PRIVATE CORTEXSDR_STATIC=1)

# HSDR shared-dictionary test (cross-layer dict, encode + decode + fused matmul)
add_executable(test_hsdr_shared_dict src/test_hsdr_shared_dict.cpp)
target_link_libraries(test_hsdr_shared_dict PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsdr_shared_dict PRIVATE CORTEXSDR_STATIC=1)

# HSDR protection-policy test (hybrid FP16 orchestration, boundary-MLP rule)
add_executable(test_hsdr_protection src/test_hsdr_protection.cpp)
target_link_libraries(test_hsdr_protection PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsdr_protection PRIVATE CORTEXSDR_STATIC=1)

# HSDR multi-pass pipeline test (auto-group by role, shared dict per group)
add_executable(test_hsdr_pipeline src/test_hsdr_pipeline.cpp)
target_link_libraries(test_hsdr_pipeline PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsdr_pipeline PRIVATE CORTEXSDR_STATIC=1)

# HSDR pipeline disk-roundtrip test (writeToFile / readFromFile / verify)
add_executable(test_hsdr_pipeline_disk src/test_hsdr_pipeline_disk.cpp)
target_link_libraries(test_hsdr_pipeline_disk PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsdr_pipeline_disk PRIVATE CORTEXSDR_STATIC=1)

# HSDAReader end-to-end test (name-indexed lookup over .hsda)
add_executable(test_hsda_reader src/test_hsda_reader.cpp)
target_link_libraries(test_hsda_reader PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsda_reader PRIVATE CORTEXSDR_STATIC=1)

# HSDR multi-dtype input test (FP16, BF16, INT8 dequantise paths)
add_executable(test_hsdr_dtypes src/test_hsdr_dtypes.cpp)
target_link_libraries(test_hsdr_dtypes PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsdr_dtypes PRIVATE CORTEXSDR_STATIC=1)

# Inference engine matmul-routing test (HSDR→fused, others→dense fallback)
add_executable(test_inference_matmul_routing src/test_inference_matmul_routing.cpp)
target_link_libraries(test_inference_matmul_routing PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_inference_matmul_routing PRIVATE CORTEXSDR_STATIC=1)

# K-quant hand-computed unit tests (Q4_K / Q5_K / Q6_K bit-layout validation)
add_executable(test_hsdr_kquants src/test_hsdr_kquants.cpp)
target_link_libraries(test_hsdr_kquants PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(test_hsdr_kquants PRIVATE CORTEXSDR_STATIC=1)

# HSDR fit-speed benchmark (default vs --fast vs tile-cap; sanity tuning data)
add_executable(bench_hsdr_fit_speed src/bench_hsdr_fit_speed.cpp)
target_link_libraries(bench_hsdr_fit_speed PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(bench_hsdr_fit_speed PRIVATE CORTEXSDR_STATIC=1)

# HSDA archive inspector — open .hsda, list dicts/segments, sanity matmul
add_executable(tool_hsda_inspect src/tool_hsda_inspect.cpp)
target_link_libraries(tool_hsda_inspect PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(tool_hsda_inspect PRIVATE CORTEXSDR_STATIC=1)

# .sdr archive inspector — strategy_id + compressed_size per segment
add_executable(tool_sdr_inspect src/tool_sdr_inspect.cpp)
target_compile_definitions(tool_sdr_inspect PRIVATE CORTEXSDR_STATIC=1)

# Dequant-stats tool: dumps first Q4_K block of a chosen weight to validate parser/dequant
add_executable(tool_dequant_stats src/tool_dequant_stats.cpp)
target_link_libraries(tool_dequant_stats PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(tool_dequant_stats PRIVATE CORTEXSDR_STATIC=1)

# K-SVD per-tensor diagnostic: compress one named weight + report W stats
add_executable(tool_ksvd_diag src/tool_ksvd_diag.cpp)
target_link_libraries(tool_ksvd_diag PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(tool_ksvd_diag PRIVATE CORTEXSDR_STATIC=1)

# Shared-dict synthetic repro — exercises the pooled K-SVD path on synthetic
# Qwen-shaped tensors to isolate the NaN-atom bug from real-data noise.
add_executable(tool_shared_dict_diag src/tool_shared_dict_diag.cpp)
target_link_libraries(tool_shared_dict_diag PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(tool_shared_dict_diag PRIVATE CORTEXSDR_STATIC=1)

# Reconstruction-quality tool: per-segment NMSE / cosine between original
# weights and HSDR-decompressed weights. Layer-level inference-readiness check.
add_executable(tool_recon_quality src/tool_recon_quality.cpp)
target_link_libraries(tool_recon_quality PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(tool_recon_quality PRIVATE CORTEXSDR_STATIC=1)

# SDR Size Analyzer Tool
add_executable(analyze_sdr_size analyze_sdr_size.cpp)
# No dependencies needed - standalone tool

# Error Calculator Tool
add_executable(calculate_error calculate_error.cpp)
target_link_libraries(calculate_error PRIVATE ${PROJECT_NAME} ${EXTRA_LIBS})
target_compile_definitions(calculate_error PRIVATE CORTEXSDR_STATIC=1)
target_include_directories(calculate_error PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

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
