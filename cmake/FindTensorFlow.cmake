# FindTensorFlow.cmake
# Find the TensorFlow library
#
# This will define:
# TensorFlow_FOUND - True if TensorFlow was found
# TensorFlow_INCLUDE_DIRS - TensorFlow include directories
# TensorFlow_LIBRARIES - TensorFlow libraries

# Try to find the TensorFlow package using pkg-config
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_TensorFlow QUIET tensorflow)
endif()

# Check for local installation first
if(TENSORFLOW_ROOT)
  set(TensorFlow_INCLUDE_DIRS "${TENSORFLOW_ROOT}/include")
  
  # Find the libraries
  find_library(TensorFlow_LIBRARY
    NAMES tensorflow
    PATHS ${TENSORFLOW_ROOT}/lib
    NO_DEFAULT_PATH
  )
  
  find_library(TensorFlow_FRAMEWORK_LIBRARY
    NAMES tensorflow_framework
    PATHS ${TENSORFLOW_ROOT}/lib
    NO_DEFAULT_PATH
  )
  
  if(TensorFlow_LIBRARY)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    if(TensorFlow_FRAMEWORK_LIBRARY)
      list(APPEND TensorFlow_LIBRARIES ${TensorFlow_FRAMEWORK_LIBRARY})
    endif()
  endif()
else()
  # Find the include directory
  find_path(TensorFlow_INCLUDE_DIR
    NAMES tensorflow/c/c_api.h
    PATHS
      ${PC_TensorFlow_INCLUDE_DIRS}
      /usr/include
      /usr/local/include
      /opt/tensorflow/include
  )

  # Find the libraries
  find_library(TensorFlow_LIBRARY
    NAMES tensorflow
    PATHS
      ${PC_TensorFlow_LIBRARY_DIRS}
      /usr/lib
      /usr/lib64
      /usr/local/lib
      /usr/local/lib64
      /opt/tensorflow/lib
  )
  
  find_library(TensorFlow_FRAMEWORK_LIBRARY
    NAMES tensorflow_framework
    PATHS
      ${PC_TensorFlow_LIBRARY_DIRS}
      /usr/lib
      /usr/lib64
      /usr/local/lib
      /usr/local/lib64
      /opt/tensorflow/lib
  )
  
  if(TensorFlow_LIBRARY)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    if(TensorFlow_FRAMEWORK_LIBRARY)
      list(APPEND TensorFlow_LIBRARIES ${TensorFlow_FRAMEWORK_LIBRARY})
    endif()
  endif()
  
  if(TensorFlow_INCLUDE_DIR)
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
  endif()
endif()

# Set variables for standard find_package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow
  REQUIRED_VARS TensorFlow_LIBRARY TensorFlow_INCLUDE_DIRS
)

# Mark variables as advanced
mark_as_advanced(
  TensorFlow_INCLUDE_DIR
  TensorFlow_LIBRARY
  TensorFlow_FRAMEWORK_LIBRARY
)
