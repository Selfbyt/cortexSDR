# Qt is already found in the parent CMakeLists.txt

# Enable Qt features (these should be inherited from parent, but adding for clarity)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(sdr_test
    ../src/test.cpp
)

target_link_libraries(sdr_test
    PRIVATE
        cortexsdr
        Qt6::Widgets
)
