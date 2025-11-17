#include "FloatMatrix.hpp"
#include "NormalHeWeightInitializer.hpp"


nnn::NormalHeWeightInitializer::NormalHeWeightInitializer(): m_rng(std::random_device{}()) {}

nnn::NormalHeWeightInitializer::NormalHeWeightInitializer(unsigned int seed): m_rng(seed) {}

nnn::FloatMatrix nnn::NormalHeWeightInitializer::Initialize(size_t n, size_t m) {
    double stdDeviation = std::sqrt(2.0 / n);
    std::normal_distribution<float> distribution(0.0, stdDeviation);

    auto weights = FloatMatrix(n,m);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            weights(i,j) = distribution(m_rng);
        }
    }
    return weights;
}