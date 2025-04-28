# Qt is already found in the parent CMakeLists.txt

# Enable Qt features (these should be inherited from parent, but adding for clarity)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(sdr_test
    test.cpp # Updated path
)

target_link_libraries(sdr_test
    PRIVATE
        cortexsdr
        Qt6::Widgets
)

# Added src directory to include paths for the test target.
target_include_directories(sdr_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
