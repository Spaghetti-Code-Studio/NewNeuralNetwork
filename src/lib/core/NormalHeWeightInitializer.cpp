#include "FloatMatrix.hpp"
#include "NormalHeWeightInitializer.hpp"

nnn::NormalHeWeightInitializer::NormalHeWeightInitializer() : m_rng(std::random_device{}()) {}

nnn::NormalHeWeightInitializer::NormalHeWeightInitializer(unsigned int seed) : m_rng(seed) {}

nnn::FloatMatrix nnn::NormalHeWeightInitializer::Initialize(size_t row, size_t col) {
  double stdDeviation = std::sqrt(2.0 / row);
  std::normal_distribution<float> distribution(0.0, stdDeviation);

  auto weights = FloatMatrix(row, col);

  for (size_t i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j++) {
      weights(i, j) = distribution(m_rng);
    }
  }
  return weights;
}