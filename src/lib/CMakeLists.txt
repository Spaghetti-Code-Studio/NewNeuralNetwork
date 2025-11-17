add_library(NewNeuralNetwork 
    "math/FloatMatrix.cpp"
    "math/RowMajorFloatMatrixIterator.cpp"
    "math/ColumnMajorFloatMatrixIterator.cpp"
    "core/DenseLayer.cpp"
    "core/ReLU.cpp"
    "core/LeakyReLU.cpp"
    "core/NeuralNetwork.cpp"
    "core/MSE.cpp"
    "core/SoftmaxDenseOutputLayer.cpp"
    "core/Softmax.cpp"
    "core/CrossEntropyWithSoftmax.cpp"
)

target_include_directories(NewNeuralNetwork
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/core"
        "${CMAKE_CURRENT_SOURCE_DIR}/math"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/result"
)

if (INCLUDE_TESTS STREQUAL "ON")
    add_executable(FloatMatrixUnitTests "math/tests/FloatMatrix.test.cpp")
    target_link_libraries(FloatMatrixUnitTests PRIVATE Catch2::Catch2WithMain NewNeuralNetwork)

    add_executable(NeuralNetworkWorkflowsUnitTests "core/tests/NeuralNetworkWorkflows.test.cpp")
    target_link_libraries(NeuralNetworkWorkflowsUnitTests PRIVATE Catch2::Catch2WithMain NewNeuralNetwork)

    enable_testing()
    add_test(NAME FloatMatrixUnitTests COMMAND FloatMatrixUnitTests)
    add_test(NAME NeuralNetworkWorkflowsUnitTests COMMAND NeuralNetworkWorkflowsUnitTests)
endif()