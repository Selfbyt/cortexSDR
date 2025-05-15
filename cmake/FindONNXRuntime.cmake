# FindONNXRuntime.cmake
# Find the ONNXRuntime library
#
# This will define:
# ONNXRuntime_FOUND - True if ONNXRuntime was found
# ONNXRuntime_INCLUDE_DIRS - ONNXRuntime include directories
# ONNXRuntime_LIBRARIES - ONNXRuntime libraries

# Try to find the ONNXRuntime package using pkg-config
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_ONNXRuntime QUIET onnxruntime)
endif()

# Find the include directory
find_path(ONNXRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS
    ${PC_ONNXRuntime_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    /usr/include/onnxruntime
    /usr/local/include/onnxruntime
  PATH_SUFFIXES onnxruntime/core/session
)

# Find the library
find_library(ONNXRuntime_LIBRARY
  NAMES onnxruntime
  PATHS
    ${PC_ONNXRuntime_LIBRARY_DIRS}
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
)

# Set variables for standard find_package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
)

# Set output variables
if(ONNXRuntime_FOUND)
  set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})
  set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
endif()

# Mark variables as advanced
mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
