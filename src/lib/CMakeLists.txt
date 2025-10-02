add_library(NewNeuralNetwork Math.cpp)

target_include_directories(NewNeuralNetwork
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
