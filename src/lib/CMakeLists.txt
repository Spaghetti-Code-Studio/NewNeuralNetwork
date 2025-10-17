add_library(NewNeuralNetwork 
    "math/FloatMatrix.cpp"
    "math/RowMajorFloatMatrixIterator.cpp"
    "math/ColumnMajorFloatMatrixIterator.cpp"
)

target_include_directories(NewNeuralNetwork
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/core"
        "${CMAKE_CURRENT_SOURCE_DIR}/math"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/result"
)

if (INCLUDE_TESTS STREQUAL "ON")
    add_executable(MathUnitTests "math/tests/FloatMatrix.test.cpp")
    target_link_libraries(MathUnitTests PRIVATE Catch2::Catch2WithMain NewNeuralNetwork)

    enable_testing()
    add_test(NAME MathUnitTests COMMAND MathUnitTests)
endif()