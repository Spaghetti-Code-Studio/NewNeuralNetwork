add_library(NewNeuralNetwork
    "math/Math.cpp"
)

target_include_directories(NewNeuralNetwork
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/core"
        "${CMAKE_CURRENT_SOURCE_DIR}/math"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/result"
)