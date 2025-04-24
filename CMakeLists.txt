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
    src/encoders/AudioEncoding.cpp
    src/encoders/DateTimeEncoding.cpp
    src/encoders/ImageEncoding.cpp
    src/encoders/NumberEncoding.cpp
    src/encoders/SpecialCharacterEncoding.cpp
    src/encoders/VideoEncoding.cpp
    src/encoders/WordEncoding.cpp
    src/cortexSDR.cpp
)

# Add UI sources only for desktop build
if(BUILD_DESKTOP)
    set(UI_SOURCES
        src/SDRWindow.cpp
    )
endif()

# Library target
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
)

# Link libraries based on build type
if(BUILD_DESKTOP)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Qt6::Widgets
    )
endif()

# Test UI only for desktop build
if(BUILD_DESKTOP)
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
    
    # Install headers
    install(DIRECTORY src/
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
