# Find GTest package
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)

# Add test executable
add_executable(test_compressed_inference
    test_compressed_inference.cpp
)

# Add include directories
target_include_directories(test_compressed_inference
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

# Link against our library and GTest
target_link_libraries(test_compressed_inference
    PRIVATE
    cortexsdr
    nlohmann_json::nlohmann_json
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    pthread
)

# Add test to CTest
add_test(NAME test_compressed_inference COMMAND test_compressed_inference)

# Add GPT-2 text generation example
add_executable(test_gpt2_text_generation
    test_gpt2_text_generation.cpp
)
target_include_directories(test_gpt2_text_generation
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_gpt2_text_generation
    PRIVATE
    cortexsdr
    nlohmann_json::nlohmann_json
    pthread
)
# Optionally, add as a test if you want to run it with CTest
# add_test(NAME test_gpt2_text_generation COMMAND test_gpt2_text_generation)