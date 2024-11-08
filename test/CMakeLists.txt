find_package(Qt6 COMPONENTS Widgets REQUIRED)

add_executable(sdr_test
    SDRWindow.cpp
    test.cpp
)

target_link_libraries(sdr_test
    PRIVATE
        cortexsdr
        Qt6::Widgets
)
