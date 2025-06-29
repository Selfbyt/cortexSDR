# CortexSDR Firmware Configuration
# This file configures the build system for both embedded targets and server-side firmware

# Set firmware build flag
set(BUILD_FIRMWARE ON)
set(BUILD_DESKTOP OFF)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Optimize for size in release builds
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -ffunction-sections -fdata-sections")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
else()
    # Debug build with symbols
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
endif()

# Target platform configuration
# Uncomment and modify the appropriate section for your target

# === ARM Cortex-M4 Configuration ===
#set(CMAKE_SYSTEM_NAME Generic)
#set(CMAKE_SYSTEM_PROCESSOR arm)
#set(CMAKE_C_COMPILER arm-none-eabi-gcc)
#set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
#set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
#set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
#set(CMAKE_OBJDUMP arm-none-eabi-objdump)
#set(CMAKE_SIZE arm-none-eabi-size)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -specs=nosys.specs -specs=nano.specs")
#set(CORTEX_TARGET_PLATFORM "ARM_CORTEX_M4")

# === RISC-V Configuration ===
#set(CMAKE_SYSTEM_NAME Generic)
#set(CMAKE_SYSTEM_PROCESSOR riscv)
#set(CMAKE_C_COMPILER riscv64-unknown-elf-gcc)
#set(CMAKE_CXX_COMPILER riscv64-unknown-elf-g++)
#set(CMAKE_ASM_COMPILER riscv64-unknown-elf-gcc)
#set(CMAKE_OBJCOPY riscv64-unknown-elf-objcopy)
#set(CMAKE_OBJDUMP riscv64-unknown-elf-objdump)
#set(CMAKE_SIZE riscv64-unknown-elf-size)
#set(CORTEX_TARGET_PLATFORM "RISC_V")

# === Server-Side Firmware (x86_64) ===
# This is the default configuration for server-side firmware
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CORTEX_TARGET_PLATFORM "SERVER_X86_64")

# Memory layout - adjust for your target device
# For embedded targets, uncomment and adjust these values
#set(FLASH_ORIGIN 0x08000000)
#set(FLASH_SIZE 512K)
#set(RAM_ORIGIN 0x20000000)
#set(RAM_SIZE 128K)

# Additional firmware-specific definitions
add_compile_definitions(
    CORTEX_SDR_FIRMWARE
    FIRMWARE_VERSION="1.0.0"
    MEMORY_LIMIT_KB=512
    TARGET_PLATFORM="${CORTEX_TARGET_PLATFORM}"
)

# Include paths for firmware
include_directories(
    ${CMAKE_SOURCE_DIR}/firmware/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ai_compression
)

# Set output file names
set(FIRMWARE_OUTPUT_NAME "cortexsdr_firmware_${CORTEX_TARGET_PLATFORM}")
