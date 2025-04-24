# Firmware configuration for cortexSDR
# This file sets up the toolchain and build options for embedded targets

# Set firmware build flag
set(BUILD_FIRMWARE ON)
set(BUILD_DESKTOP OFF)

# Optimization for size
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os")

# Embedded target options - uncomment and modify for your specific target
# For ARM Cortex-M4
#set(CMAKE_SYSTEM_NAME Generic)
#set(CMAKE_SYSTEM_PROCESSOR arm)
#set(CMAKE_C_COMPILER arm-none-eabi-gcc)
#set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
#set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
#set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
#set(CMAKE_OBJDUMP arm-none-eabi-objdump)
#set(CMAKE_SIZE arm-none-eabi-size)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -specs=nosys.specs -specs=nano.specs")

# Memory layout - adjust for your target device
#set(FLASH_ORIGIN 0x08000000)
#set(FLASH_SIZE 512K)
#set(RAM_ORIGIN 0x20000000)
#set(RAM_SIZE 128K)

# Additional firmware-specific definitions
add_compile_definitions(
    CORTEX_SDR_FIRMWARE
    FIRMWARE_VERSION="1.0.0"
    MEMORY_LIMIT_KB=512
)
