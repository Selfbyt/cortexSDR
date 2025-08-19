cmake_minimum_required(VERSION 3.20)
project(cortexsdr_sdk_python LANGUAGES CXX)

# Options
set(CORTEXSDR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src" CACHE PATH "Path to CortexSDR C/C++ headers")
set(CORTEXSDR_LIB_DIR "" CACHE PATH "Path to directory containing libcortexsdr_sdk*.so")

# Allow environment overrides (always override if provided)
if(DEFINED ENV{CORTEXSDR_LIB_DIR})
  set(CORTEXSDR_LIB_DIR "$ENV{CORTEXSDR_LIB_DIR}" CACHE PATH "Path to directory containing libcortexsdr_sdk*.so" FORCE)
endif()
if(DEFINED ENV{CORTEXSDR_INCLUDE_DIR})
  set(CORTEXSDR_INCLUDE_DIR "$ENV{CORTEXSDR_INCLUDE_DIR}" CACHE PATH "Path to CortexSDR C/C++ headers" FORCE)
endif()

# Python and pybind11
find_package(Python COMPONENTS Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Locate CortexSDR SDK shared library
if(NOT CORTEXSDR_LIB)
  if(CORTEXSDR_LIB_DIR)
    find_library(CORTEXSDR_LIB
      NAMES cortexsdr_sdk_shared cortexsdr_sdk libcortexsdr_sdk_shared libcortexsdr_sdk libcortexsdr
      HINTS ${CORTEXSDR_LIB_DIR}
    )
  else()
    find_library(CORTEXSDR_LIB
      NAMES cortexsdr_sdk_shared cortexsdr_sdk libcortexsdr_sdk_shared libcortexsdr_sdk libcortexsdr
      PATHS /usr/local/lib /usr/lib
    )
  endif()
endif()


if(NOT CORTEXSDR_LIB)
  message(FATAL_ERROR "Could not find CortexSDR SDK shared library. Set CORTEXSDR_LIB_DIR or CORTEXSDR_LIB.")
endif()

message(STATUS "Using CortexSDR lib: ${CORTEXSDR_LIB}")

# Locate CortexSDR headers (cortex_sdk.h)
find_path(CORTEXSDR_INCLUDE_DIR_RES
  NAMES ai_compression/api/cortex_sdk.h
  HINTS ${CORTEXSDR_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        /usr/local/include
        /usr/include
)

if(NOT CORTEXSDR_INCLUDE_DIR_RES)
  message(FATAL_ERROR "Could not find cortex_sdk.h. Set CORTEXSDR_INCLUDE_DIR to the SDK 'src' directory (containing ai_compression/api/cortex_sdk.h)")
endif()

message(STATUS "Using CortexSDR include: ${CORTEXSDR_INCLUDE_DIR_RES}")

# Build the Python extension module
pybind11_add_library(_bindings MODULE
  cortexsdr_sdk/_bindings.cpp
  ${CORTEXSDR_INCLUDE_DIR_RES}/ai_compression/utils/sha256.cpp
)

target_include_directories(_bindings PRIVATE ${CORTEXSDR_INCLUDE_DIR_RES})

target_link_libraries(_bindings PRIVATE ${CORTEXSDR_LIB})

# Use C++17 and instruct scikit-build-core where to place the module in the wheel
set_target_properties(_bindings PROPERTIES
  CXX_STANDARD 17
  OUTPUT_NAME "_bindings"
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${SKBUILD_PLATLIB_DIR}/cortexsdr_sdk"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

# Install the extension module into the Python package directory inside the wheel
install(TARGETS _bindings
  LIBRARY DESTINATION cortexsdr_sdk
  ARCHIVE DESTINATION cortexsdr_sdk
  RUNTIME DESTINATION cortexsdr_sdk
)

# Bundle the SDK shared library into the wheel next to the extension so RPATH $ORIGIN resolves it
install(FILES ${CORTEXSDR_LIB} DESTINATION cortexsdr_sdk)
