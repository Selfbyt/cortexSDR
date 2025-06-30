# Findnlohmann_json.cmake
# Find the nlohmann_json library
#
# This module defines
# nlohmann_json_FOUND - True if nlohmann_json was found
# nlohmann_json_INCLUDE_DIRS - The nlohmann_json include directory

# Try to find the header
find_path(nlohmann_json_INCLUDE_DIR
  NAMES nlohmann/json.hpp
  PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    /usr/include
    /usr/local/include
)

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nlohmann_json DEFAULT_MSG nlohmann_json_INCLUDE_DIR)

# Set output variables
if(nlohmann_json_FOUND)
  set(nlohmann_json_INCLUDE_DIRS ${nlohmann_json_INCLUDE_DIR})
endif()

# Hide internal variables
mark_as_advanced(nlohmann_json_INCLUDE_DIR)
