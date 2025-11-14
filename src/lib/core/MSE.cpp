#include "MSE.hpp"

nnn::FloatMatrix nnn::MSE::Loss(const FloatMatrix& actual, const FloatMatrix& expected) {
    return (actual - expected) * 2.0f;
}