#include "NormalGlorotWeightInitializer.hpp"

nnn::NormalGlorotWeightInitializer::NormalGlorotWeightInitializer(): m_rng(std::random_device{}()) {}

nnn::NormalGlorotWeightInitializer::NormalGlorotWeightInitializer(unsigned int seed): m_rng(seed) {}

nnn::FloatMatrix nnn::NormalGlorotWeightInitializer::Initialize(size_t n, size_t m) {
    double stdDeviation = std::sqrt(2.0 / (n + m));
    std::normal_distribution<float> distribution(0.0, stdDeviation);

    auto weights = FloatMatrix(n,m);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            weights(i,j) = distribution(m_rng);
        }
    }
    return weights;
}