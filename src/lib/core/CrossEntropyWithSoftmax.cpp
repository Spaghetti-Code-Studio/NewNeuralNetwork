#include "CrossEntropyWithSoftmax.hpp"

nnn::FloatMatrix nnn::CrossEntropyWithSoftmax::Loss(const FloatMatrix& actual, const FloatMatrix& expected) {
  return actual - expected;
}