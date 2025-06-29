# CortexSDR Firmware CMake Configuration
cmake_minimum_required(VERSION 3.10)

# Include firmware configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/config/firmware_config.cmake)

# Define firmware sources
set(FIRMWARE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cortex_firmware.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/firmware_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cortex_resource_monitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cortex_resource_monitor_impl.cpp
)

# Define SDK sources needed by firmware
set(SDK_SOURCES
    ${CMAKE_SOURCE_DIR}/src/ai_compression/AICompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/AIDecompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/ArchiveConstants.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/StreamingCompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/ModelConverter.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/SparseInferenceEngine.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/SDRModelLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/NumericalRLE.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/GzipStrategy.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/AdaptiveSDRStrategy.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/MetadataSDRStrategy.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/QuantizedTensorStrategy.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/SDRIndexStorage.cpp
    ${CMAKE_SOURCE_DIR}/src/ai_compression/c_api.cpp
)

# Add firmware library target
add_library(cortex_firmware STATIC
    ${FIRMWARE_SOURCES}
    ${SDK_SOURCES}
)

# Add firmware executable target
add_executable(cortexsdr_firmware_${CORTEX_TARGET_PLATFORM} 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/firmware_main.cpp
)

# Link libraries are set below with other dependencies

# Add executable for firmware test
add_executable(firmware_test src/cortex_firmware_test.cpp)
target_link_libraries(firmware_test cortex_firmware)

# Add executable for resource monitor test
add_executable(resource_monitor_test src/cortex_resource_monitor_test.cpp)
target_link_libraries(resource_monitor_test cortex_firmware)

# Set include directories for library
target_include_directories(cortex_firmware PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

# Set include directories for executable
target_include_directories(cortexsdr_firmware_${CORTEX_TARGET_PLATFORM} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/ai_compression
)

# Set compiler flags
target_compile_features(cortexsdr_firmware_${CORTEX_TARGET_PLATFORM} PRIVATE cxx_std_17)

# Link libraries
target_link_libraries(cortexsdr_firmware_${CORTEX_TARGET_PLATFORM} PRIVATE
    cortex_firmware           # Our firmware library
    ${CMAKE_THREAD_LIBS_INIT}  # For std::thread
    z                          # For zlib/gzip compression
)

# For embedded targets, add linker script if needed
if(NOT CORTEX_TARGET_PLATFORM STREQUAL "SERVER_X86_64")
    # Uncomment and adjust for your specific embedded target
    #set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/config/stm32f4_flash.ld)
    #target_link_options(cortexsdr_firmware_${CORTEX_TARGET_PLATFORM} PRIVATE -T${LINKER_SCRIPT})
endif()

# Post-build commands
if(NOT CORTEX_TARGET_PLATFORM STREQUAL "SERVER_X86_64")
    # For embedded targets, generate binary and hex files
    add_custom_command(TARGET cortexsdr_firmware_${CORTEX_TARGET_PLATFORM} POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O binary 
            $<TARGET_FILE:cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}> 
            $<TARGET_FILE:cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}>.bin
        COMMAND ${CMAKE_OBJCOPY} -O ihex 
            $<TARGET_FILE:cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}> 
            $<TARGET_FILE:cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}>.hex
        COMMENT "Generating binary and hex files"
    )
endif()

# Print firmware size after build
add_custom_command(TARGET cortexsdr_firmware_${CORTEX_TARGET_PLATFORM} POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}>
    COMMENT "Firmware size:"
)

# Install targets
install(TARGETS cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}
    RUNTIME DESTINATION bin
)

# If this is an embedded target, also install the binary and hex files
if(NOT CORTEX_TARGET_PLATFORM STREQUAL "SERVER_X86_64")
    install(FILES
        $<TARGET_FILE:cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}>.bin
        $<TARGET_FILE:cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}>.hex
        DESTINATION bin
    )
endif()
